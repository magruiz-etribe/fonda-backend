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
) -> str | dict[str, Any]:
    return _invoke(
        model_id,
        system,
        messages,
        inference_config,
        tool_config,
        return_full=return_full,
    )


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
    s = (text or "").strip()
    m = _JSON_FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()
    # Fallback: try to slice the first balanced JSON object/array if the
    # model added stray prose around it.
    if not s.startswith(("{", "[")):
        start = min(
            (i for i in (s.find("{"), s.find("[")) if i != -1),
            default=-1,
        )
        if start != -1:
            s = s[start:]
    return json.loads(s)


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


def _invoke(
    model_id: str,
    system: str,
    messages: list[dict[str, Any]],
    inference_config: dict[str, Any] | None,
    tool_config: dict[str, Any] | None = None,
    *,
    return_full: bool = False,
) -> str | dict[str, Any]:
    kwargs: dict[str, Any] = {
        "modelId": model_id,
        "messages": messages,
    }
    if system.strip():
        kwargs["system"] = [{"text": system}]
    if inference_config:
        kwargs["inferenceConfig"] = inference_config
    if tool_config:
        kwargs["toolConfig"] = tool_config

    last: BaseException | None = None
    for attempt in (1, 2):
        try:
            resp = _client.converse(**kwargs)
            if return_full:
                logger.info(
                    "bedrock_converse_ok",
                    extra={
                        "attempt": attempt,
                        "model_id": model_id,
                        "stop_reason": resp.get("stopReason"),
                        "grounding": bool(tool_config),
                    },
                )
                return resp
            text = _extract_text(resp)
            logger.info(
                "bedrock_converse_ok",
                extra={
                    "attempt": attempt,
                    "model_id": model_id,
                    "reply_len": len(text),
                    "stop_reason": resp.get("stopReason"),
                    "grounding": bool(tool_config),
                },
            )
            return text
        except _RETRYABLE as e:
            last = e
            logger.warning(
                "bedrock_converse_retryable_error model_id=%s attempt=%s error=%s",
                model_id,
                attempt,
                _format_bedrock_error(e),
            )
            if attempt == 2:
                break
            time.sleep(_RETRY_BACKOFF_S)

    raise BedrockError(f"bedrock converse failed after retry: {_format_bedrock_error(last)}") from last


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
