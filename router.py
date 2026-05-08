from __future__ import annotations

import logging
from typing import Final

import classifier as cls_module
import generation as gen_module
import retrieval
from generation import GenResult

logger = logging.getLogger(__name__)

_FALLBACK_RESULT: Final[GenResult] = GenResult(
    response=["Disculpa, tuve un problema procesando tu mensaje. ¿Puedes intentarlo de nuevo? 😊"],
    current_dishes=[],
    buttons=[],
)


def handle(
    message: str,
    current_dishes: list[str],
    history: list[dict[str, str]],
) -> GenResult:
    try:
        cr = cls_module.classify(message, current_dishes, history)
        kb_context = _get_kb_context(cr)
        return gen_module.generate(cr, message, kb_context, history)
    except Exception as e:
        logger.exception("router_unhandled_exception", extra={"error": str(e)})
        return _FALLBACK_RESULT


def _get_kb_context(cr: cls_module.ClassifierResult) -> str:
    if cr.intent == "traduccion":
        return retrieval.get_context_for_dishes(cr.current_dishes)
    if cr.intent in ("maps", "higiene"):
        return retrieval.get_static(cr.intent)  # type: ignore[arg-type]
    return ""
