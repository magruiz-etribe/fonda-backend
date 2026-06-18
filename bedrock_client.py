from __future__ import annotations

import base64
import json
import logging
import re
import time
from typing import Any, Final

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, ReadTimeoutError

import config
import timing

logger = logging.getLogger(__name__)


class BedrockError(Exception):
    pass


_RETRYABLE: Final[tuple[type[BaseException], ...]] = (
    ClientError,
    ReadTimeoutError,
    BotoCoreError,
)

_RETRY_BACKOFF_S: Final[float] = 0.4

_client = boto3.client(
    "bedrock-runtime",
    region_name=config.AWS_REGION,
    config=Config(
        read_timeout=config.BEDROCK_READ_TIMEOUT_S,
        connect_timeout=config.BEDROCK_CONNECT_TIMEOUT_S,
        # We handle retry semantics manually (1 extra attempt) per spec.
        retries={"max_attempts": 1, "mode": "standard"},
    ),
)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def converse(
    model_id: str,
    system: str,
    messages: list[dict[str, Any]],
    inference_config: dict[str, Any] | None = None,
    tool_config: dict[str, Any] | None = None,
    *,
    return_full: bool = False,
    stage: str | None = None,
) -> str | dict[str, Any]:
    return _invoke(
        model_id,
        system,
        messages,
        inference_config,
        tool_config,
        return_full=return_full,
        stage=stage,
        structured=False,
    )


def structured_tool_config(
    name: str,
    schema: dict[str, Any],
    description: str = "",
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Build a Nova toolConfig that forces schema-valid JSON via constrained decoding."""
    tool_spec: dict[str, Any] = {
        "name": name,
        "description": description or f"Structured output: {name}",
        "inputSchema": {"json": schema},
    }
    if strict:
        tool_spec["strict"] = True
    return {
        "tools": [{"toolSpec": tool_spec}],
        "toolChoice": {"tool": {"name": name}},
    }


def converse_json(
    model_id: str,
    system: str,
    messages: list[dict[str, Any]],
    *,
    schema: dict[str, Any],
    tool_name: str,
    tool_description: str = "",
    inference_config: dict[str, Any] | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    """Call Bedrock with a forced tool schema and return the parsed tool input dict."""
    structured_error: BaseException | None = None

    for strict in (True, False):
        tool_config = structured_tool_config(
            tool_name,
            schema,
            tool_description,
            strict=strict,
        )
        try:
            resp = _invoke(
                model_id,
                system,
                messages,
                inference_config,
                tool_config,
                return_full=True,
                stage=stage,
                structured=True,
            )
        except BedrockError as e:
            structured_error = e
            if strict:
                logger.warning(
                    "converse_json_strict_failed",
                    extra={
                        "tool_name": tool_name,
                        "stage": stage or "",
                        "error": str(e)[:300],
                    },
                )
                continue
            break

        if not isinstance(resp, dict):
            structured_error = BedrockError("structured converse returned unexpected type")
            break

        parsed = _try_parse_structured_response(resp, tool_name, stage)
        if parsed is not None:
            return parsed

        structured_error = BedrockError(f"structured output missing toolUse for {tool_name}")
        logger.warning(
            "converse_json_tool_use_missing",
            extra={
                "tool_name": tool_name,
                "stage": stage or "",
                "stop_reason": resp.get("stopReason"),
                "strict": strict,
            },
        )
        if strict:
            continue
        break

    logger.info(
        "converse_json_fallback_text",
        extra={"tool_name": tool_name, "stage": stage or ""},
    )
    try:
        text = _invoke(
            model_id,
            system,
            messages,
            inference_config,
            None,
            return_full=False,
            stage=stage,
            structured=False,
        )
        parsed = parse_json_lenient(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:
        structured_error = structured_error or e

    raise BedrockError(
        f"structured output failed for {tool_name}: {_format_bedrock_error(structured_error)}"
    ) from structured_error


def _try_parse_structured_response(
    resp: dict[str, Any],
    tool_name: str,
    stage: str | None,
) -> dict[str, Any] | None:
    try:
        return _extract_tool_input(resp, tool_name)
    except BedrockError:
        pass

    try:
        text = _extract_text(resp)
        parsed = parse_json_lenient(text)
        if isinstance(parsed, dict):
            logger.info(
                "converse_json_parsed_text_fallback",
                extra={"tool_name": tool_name, "stage": stage or ""},
            )
            return parsed
    except (BedrockError, json.JSONDecodeError):
        pass
    return None


def converse_with_image(
    model_id: str,
    system: str,
    prompt: str,
    image_b64: str,
    inference_config: dict[str, Any] | None = None,
) -> str:
    image_bytes = base64.b64decode(image_b64, validate=True)
    fmt = _detect_image_format(image_bytes)
    messages = [
        {
            "role": "user",
            "content": [
                {"image": {"format": fmt, "source": {"bytes": image_bytes}}},
                {"text": prompt},
            ],
        }
    ]
    return _invoke(model_id, system, messages, inference_config)


def parse_json_strict(text: str) -> Any:
    s = _prepare_json_text(text)
    return json.loads(s)


def parse_json_lenient(text: str) -> Any:
    """Parse JSON from LLM output, tolerating common formatting mistakes."""
    last_error: Exception | None = None
    seen: set[str] = set()
    for candidate in _json_parse_candidates(text):
        for variant in _json_repair_variants(candidate):
            if variant in seen:
                continue
            seen.add(variant)
            try:
                return json.loads(variant)
            except json.JSONDecodeError as e:
                last_error = e
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("no JSON found", text or "", 0)


def _prepare_json_text(text: str) -> str:
    s = (text or "").strip()
    m = _JSON_FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()
    if not s.startswith(("{", "[")):
        start = min(
            (i for i in (s.find("{"), s.find("[")) if i != -1),
            default=-1,
        )
        if start != -1:
            s = s[start:]
    return s


def _json_parse_candidates(text: str) -> list[str]:
    s = (text or "").strip()
    candidates: list[str] = []
    for item in (s, _prepare_json_text(s)):
        if item and item not in candidates:
            candidates.append(item)
    balanced = _extract_balanced_json_object(s)
    if balanced and balanced not in candidates:
        candidates.append(balanced)
    m = _JSON_FENCE_RE.search(s)
    if m:
        fenced = m.group(1).strip()
        balanced_fenced = _extract_balanced_json_object(fenced)
        for item in (fenced, balanced_fenced):
            if item and item not in candidates:
                candidates.append(item)
    return candidates


def _json_repair_variants(text: str) -> list[str]:
    variants: list[str] = []
    for base in (text, _fix_json_string_newlines(text)):
        for variant in (base, _fix_trailing_commas(base), _fix_trailing_commas(_fix_json_string_newlines(text))):
            if variant and variant not in variants:
                variants.append(variant)
    return variants


def _extract_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _fix_json_string_newlines(text: str) -> str:
    result: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            result.append(ch)
            escape = False
            continue
        if ch == "\\" and in_string:
            result.append(ch)
            escape = True
            continue
        if ch == '"':
            result.append(ch)
            in_string = not in_string
            continue
        if in_string and ch == "\n":
            result.append("\\n")
            continue
        if in_string and ch == "\r":
            continue
        if in_string and ch == "\t":
            result.append("\\t")
            continue
        result.append(ch)
    return "".join(result)


def _fix_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _format_bedrock_error(exc: BaseException) -> str:
    if isinstance(exc, ClientError):
        err = exc.response.get("Error", {})
        code = err.get("Code", type(exc).__name__)
        message = err.get("Message", str(exc))
        return f"{code}: {message}"
    return str(exc)


def extract_grounding_log_data(resp: dict[str, Any]) -> dict[str, Any]:
    """Parse Nova Web Grounding tool calls and citations from a converse response."""
    content: list[Any] = []
    try:
        content = resp["output"]["message"]["content"]
    except (KeyError, TypeError):
        return {"queries": [], "citations": [], "text": ""}

    queries: list[str] = []
    citations: list[dict[str, str]] = []
    text_parts: list[str] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        if text := block.get("text"):
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
        if tool_use := block.get("toolUse"):
            if not isinstance(tool_use, dict):
                continue
            if tool_use.get("name") == "nova_grounding":
                inp = tool_use.get("input") or {}
                if isinstance(inp, dict):
                    if q := inp.get("query"):
                        queries.append(str(q))
        if cite_block := block.get("citationsContent"):
            if not isinstance(cite_block, dict):
                continue
            for citation in cite_block.get("citations") or []:
                if not isinstance(citation, dict):
                    continue
                web = (citation.get("location") or {}).get("web") or {}
                if not isinstance(web, dict):
                    continue
                url = str(web.get("url") or "")
                domain = str(web.get("domain") or "")
                if url or domain:
                    citations.append({"url": url, "domain": domain})

    return {
        "queries": queries,
        "citations": citations,
        "text": "\n".join(text_parts),
    }


def _coerce_tool_input(inp: Any) -> dict[str, Any] | None:
    if isinstance(inp, dict) and inp:
        return inp
    if isinstance(inp, str) and inp.strip():
        try:
            parsed = parse_json_lenient(inp)
        except (json.JSONDecodeError, ValueError):
            return None
        if isinstance(parsed, dict) and parsed:
            return parsed
    return None


def _extract_tool_input(resp: dict[str, Any], tool_name: str) -> dict[str, Any]:
    try:
        content = resp["output"]["message"]["content"]
    except (KeyError, TypeError) as e:
        raise BedrockError(f"unexpected response shape: {e}") from e

    for block in content:
        if not isinstance(block, dict):
            continue
        tool_use = block.get("toolUse")
        if not isinstance(tool_use, dict):
            continue
        if tool_use.get("name") != tool_name:
            continue
        if data := _coerce_tool_input(tool_use.get("input")):
            return data
        raise BedrockError(f"toolUse input empty or invalid for {tool_name}")
    raise BedrockError(f"toolUse block not found for {tool_name}")


def _invoke(
    model_id: str,
    system: str,
    messages: list[dict[str, Any]],
    inference_config: dict[str, Any] | None,
    tool_config: dict[str, Any] | None = None,
    *,
    return_full: bool = False,
    stage: str | None = None,
    structured: bool = False,
) -> str | dict[str, Any]:
    kwargs: dict[str, Any] = {
        "modelId": model_id,
        "system": [{"text": system}],
        "messages": messages,
    }
    if inference_config:
        kwargs["inferenceConfig"] = inference_config

    last: BaseException | None = None
    invoke_start = time.perf_counter()
    for attempt in (1, 2):
        try:
            resp = _client.converse(**kwargs)
            invoke_ms = (time.perf_counter() - invoke_start) * 1000
            if stage:
                timing.record_llm(stage, invoke_ms, model_id=model_id, attempt=attempt)
            if return_full:
                logger.info(
                    "bedrock_converse_ok",
                    extra={
                        "attempt": attempt,
                        "model_id": model_id,
                        "stage": stage or "",
                        "duration_ms": round(invoke_ms, 2),
                        "stop_reason": resp.get("stopReason"),
                        "grounding": bool(tool_config) and not structured,
                        "structured": structured,
                    },
                )
                return resp
            text = _extract_text(resp)
            logger.info(
                "bedrock_converse_ok",
                extra={
                    "attempt": attempt,
                    "model_id": model_id,
                    "stage": stage or "",
                    "duration_ms": round(invoke_ms, 2),
                    "reply_len": len(text),
                    "stop_reason": resp.get("stopReason"),
                    "grounding": bool(tool_config) and not structured,
                    "structured": structured,
                },
            )
            return text
        except _RETRYABLE as e:
            if isinstance(e, ClientError) and e.response["Error"]["Code"] == "ValidationException":
                raise BedrockError(f"bedrock validation error (not retryable): {e}") from e
            last = e
            logger.warning(
                "bedrock_converse_retryable_error",
                extra={
                    "attempt": attempt,
                    "model_id": model_id,
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            )
            if attempt == 2:
                break
            time.sleep(_RETRY_BACKOFF_S)

    raise BedrockError(f"bedrock converse failed after retry: {last}") from last


def _extract_text(resp: dict[str, Any]) -> str:
    try:
        content = resp["output"]["message"]["content"]
    except (KeyError, TypeError) as e:
        raise BedrockError(f"unexpected response shape: {e}") from e

    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    if not parts:
        raise BedrockError("response had no text content")
    return "".join(parts).strip()


def _detect_image_format(b: bytes) -> str:
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if b[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if b[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "webp"
    return "jpeg"
