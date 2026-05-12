from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Final

import bedrock_client
import config
from classifier import ClassifierResult
from prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_PROMPT: Final[str] = "generation_system.txt"

_FALLBACK: Final[dict[str, Any]] = {
    "response": ["Disculpa, tuve un problema. ¿Puedes repetir tu mensaje? 😊"],
    "current_dishes": [],
    "buttons": [],
}


@dataclass
class GenResult:
    response: list[str] = field(default_factory=list)
    current_dishes: list[str] = field(default_factory=list)
    buttons: list[str] = field(default_factory=list)


def generate(
    cr: ClassifierResult,
    message: str,
    kb_context: str,
    history: list[dict[str, str]],
) -> GenResult:
    system = load_prompt(_PROMPT)
    user_text = _build_user_text(cr, message, kb_context, history)
    messages = [{"role": "user", "content": [{"text": user_text}]}]

    try:
        raw = bedrock_client.converse(
            config.NOVA_LITE_MODEL_ID,
            system,
            messages,
            inference_config={
                "maxTokens": config.GEN_MAX_TOKENS,
                "temperature": 0.5,
            },
        )
    except bedrock_client.BedrockError as e:
        logger.warning("generation_bedrock_error", extra={"error": str(e)})
        return GenResult(**_FALLBACK)

    return _parse(raw)


def _build_user_text(
    cr: ClassifierResult,
    message: str,
    kb_context: str,
    history: list[dict[str, str]],
) -> str:
    hist_lines: list[str] = []
    for h in history:
        role = "usuario" if h.get("role") == "user" else "agente"
        text = str(h.get("text", "")).strip().replace("\n", " ")
        if len(text) > 400:
            text = text[:397] + "…"
        hist_lines.append(f"  {role}: {text}")
    hist_block = "\n".join(hist_lines) if hist_lines else "(sin historial)"

    kb_block = kb_context.strip() or "(sin contexto KB — platillo personalizado o tema sin ficha)"

    pending = f'"{cr.pending_variant_for}"' if cr.pending_variant_for else "null"

    return (
        f"Intención: {cr.intent}\n"
        f"Platillos en contexto: {cr.current_dishes}\n"
        f"translate_now: {str(cr.translate_now).lower()}\n"
        f"pending_variant_for: {pending}\n\n"
        f"Contexto KB:\n{kb_block}\n\n"
        f"Historial:\n{hist_block}\n\n"
        f"Mensaje del usuario: \"{message}\"\n\n"
        "Devuelve únicamente el JSON."
    )


def _parse(raw: str) -> GenResult:
    try:
        data = bedrock_client.parse_json_strict(raw)
    except Exception as e:
        logger.warning(
            "generation_parse_error",
            extra={"error": str(e), "raw": raw[:300]},
        )
        return GenResult(**_FALLBACK)

    if not isinstance(data, dict):
        return GenResult(**_FALLBACK)

    raw_response = data.get("response") or []
    response: list[str] = []
    if isinstance(raw_response, list):
        for r in raw_response:
            if isinstance(r, str) and r.strip():
                response.append(r.strip())
    if not response:
        logger.warning("generation_empty_response", extra={"raw": str(data)[:200]})
        return GenResult(**_FALLBACK)

    raw_dishes = data.get("current_dishes") or []
    dishes: list[str] = []
    if isinstance(raw_dishes, list):
        for d in raw_dishes:
            if isinstance(d, str) and d.strip():
                dishes.append(d.strip().lower())

    raw_buttons = data.get("buttons") or []
    buttons: list[str] = []
    if isinstance(raw_buttons, list):
        for b in raw_buttons:
            if isinstance(b, str) and b.strip():
                buttons.append(b.strip())

    logger.info(
        "generation_ok",
        extra={
            "bubbles": len(response),
            "buttons": len(buttons),
            "current_dishes_out": dishes,
        },
    )

    return GenResult(response=response, current_dishes=dishes, buttons=buttons)
