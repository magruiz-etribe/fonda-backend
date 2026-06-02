from __future__ import annotations

import logging
from typing import Any, Final

import bedrock_client
import config

logger = logging.getLogger(__name__)

_WEB_SEARCH_SYSTEM: Final[str] = (
    "Eres un asistente de gastronomía mexicana. "
    "Solo extraes ingredientes y variantes de platillos. "
    "Responde en español, breve y en listas. "
    "No incluyas historia, origen, preparación, técnicas de cocción ni curiosidades."
)

_GROUNDING_TOOL: Final[dict[str, Any]] = {
    "tools": [{"systemTool": {"name": "nova_grounding"}}]
}


def _build_search_prompt(platillo: str) -> str:
    return (
        f"Busca información factual del platillo mexicano '{platillo}'. "
        "Responde SOLO con:\n"
        "1. Ingredientes base y extras visibles típicos (proteína, salsas, guarniciones, complementos)\n"
        "2. Variantes más comunes (ej. rojo/verde/negro, con pollo/con cerdo, suizas/motuleñas)\n"
        "Omite por completo: historia, origen, pasos de preparación, tiempos de cocción y anécdotas."
    )


def search_platillo(platillo: str) -> dict[str, Any] | None:
    """Search the web for a Mexican dish via Nova Web Grounding and log the results."""
    if not config.WEB_GROUNDING_ENABLED:
        return None

    platillo = platillo.strip()
    if not platillo or platillo == "custom":
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
            "web_grounding_error platillo=%s model=%s error=%s",
            platillo,
            config.NOVA_2_LITE_MODEL_ID,
            e,
        )
        return None
    except Exception as e:
        logger.exception(
            "web_grounding_error platillo=%s model=%s unexpected=%s",
            platillo,
            config.NOVA_2_LITE_MODEL_ID,
            e,
        )
        return None

    if not isinstance(resp, dict):
        logger.warning(
            "web_grounding_error platillo=%s model=%s error=unexpected response type %s",
            platillo,
            config.NOVA_2_LITE_MODEL_ID,
            type(resp).__name__,
        )
        return None

    details = bedrock_client.extract_grounding_log_data(resp)
    logger.info(
        "web_grounding_platillo platillo=%s queries=%s citations=%s summary=%s",
        platillo,
        details["queries"],
        details["citations"],
        details["text"][:5000],
    )
    return details
