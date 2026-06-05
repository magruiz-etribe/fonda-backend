from __future__ import annotations

import logging
import re
from typing import Any, Final

import bedrock_client
import config

logger = logging.getLogger(__name__)

_WEB_SEARCH_SYSTEM: Final[str] = (
    "Eres un asistente de gastronomía para fonderos de CDMX. "
    "Buscas con el nombre EXACTO que usa el fondero (puede ser coloquial, regional o un apodo local del menú). "
    "Interpreta siempre el término en contexto de comida mexicana y fondas de la Ciudad de México. "
    "Si el nombre es un apodo (ej. 'orejas de elefante'), identifica a qué platillo se refiere en ese contexto. "
    "Responde en español, breve y en listas. "
    "No incluyas historia, origen, preparación, técnicas de cocción ni curiosidades."
)

_GROUNDING_TOOL: Final[dict[str, Any]] = {
    "tools": [{"systemTool": {"name": "nova_grounding"}}]
}

_SHORT_REPLY_RE: Final[re.Pattern[str]] = re.compile(
    r"^(✅|❌|listo|ok|sí|si\b|no\b|claro|ninguno|eso es todo|correcto|exacto)(\b|!)",
    re.IGNORECASE,
)


def _is_short_reply(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if len(t) <= 2:
        return True
    if t.startswith("✅") or t.startswith("❌"):
        return True
    return bool(_SHORT_REPLY_RE.match(t))


def resolve_search_query(
    message: str,
    history: list[dict[str, str]],
    dish_context: dict | None = None,
) -> str | None:
    """Phrase to search: what the fondero said, not the canonical KB entity."""
    msg = message.strip()
    if msg and not _is_short_reply(msg):
        return msg

    dc = dish_context or {}
    if phrase := dc.get("search_phrase"):
        if isinstance(phrase, str) and phrase.strip():
            return phrase.strip()

    for turn in reversed(history):
        if turn.get("role") != "user":
            continue
        text = str(turn.get("text", "")).strip()
        if text and not _is_short_reply(text):
            return text

    return None


def _build_search_prompt(platillo: str) -> str:
    return (
        f"En contexto de comida mexicana en fondas de CDMX, busca el platillo o preparación "
        f"que la gente conoce como '{platillo}' (como lo piden en el menú, aunque sea nombre coloquial). "
        "Si es un apodo local, identifica primero a qué platillo corresponde en fondas mexicanas. "
        "Responde SOLO con:\n"
        f"1. Ingredientes base y extras visibles típicos de '{platillo}' en CDMX "
        "(proteína, salsas, guarniciones, complementos)\n"
        f"2. Variantes de '{platillo}' más comunes en CDMX\n"
        "Prioriza la versión capitalina; variantes regionales solo si también se encuentran en CDMX. "
        "Omite por completo: historia, origen, pasos de preparación, tiempos de cocción y anécdotas."
    )


def search_platillo(platillo: str) -> dict[str, Any] | None:
    """Search the web for a dish phrase via Nova Web Grounding and log the results."""
    if not config.WEB_GROUNDING_ENABLED:
        return None

    platillo = platillo.strip()
    if not platillo:
        return None

    prompt = _build_search_prompt(platillo)
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    try:
        resp = bedrock_client.converse(
            config.NOVA_2_LITE_MODEL_ID,
            _WEB_SEARCH_SYSTEM,
            messages,
            inference_config={
                "maxTokens": config.WEB_GROUNDING_MAX_TOKENS,
                "temperature": 0.2,
            },
            tool_config=_GROUNDING_TOOL,
            return_full=True,
        )
    except bedrock_client.BedrockError as e:
        logger.warning(
            "web_grounding_error query=%s model=%s error=%s",
            platillo,
            config.NOVA_2_LITE_MODEL_ID,
            e,
        )
        return None
    except Exception as e:
        logger.exception(
            "web_grounding_error query=%s model=%s unexpected=%s",
            platillo,
            config.NOVA_2_LITE_MODEL_ID,
            e,
        )
        return None

    if not isinstance(resp, dict):
        logger.warning(
            "web_grounding_error query=%s model=%s error=unexpected response type %s",
            platillo,
            config.NOVA_2_LITE_MODEL_ID,
            type(resp).__name__,
        )
        return None

    details = bedrock_client.extract_grounding_log_data(resp)
    logger.info(
        "web_grounding_platillo query=%s queries=%s citations=%s summary=%s",
        platillo,
        details["queries"],
        details["citations"],
        details["text"][:5000],
    )
    return details
