from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final

import bedrock_client
import config
from prompt_loader import load_prompt
from retrieval import get_entities_index

logger = logging.getLogger(__name__)

_PROMPT: Final[str] = "classifier_system.txt"
_VALID_INTENTS: Final[frozenset[str]] = frozenset(
    {"traduccion", "maps", "higiene", "fallback"}
)


@dataclass
class ClassifierResult:
    intent: str
    current_dishes: list[str] = field(default_factory=list)
    translate_now: bool = False


def classify(
    message: str,
    current_dishes: list[str],
    history: list[dict[str, str]],
) -> ClassifierResult:
    entities_index = get_entities_index()
    user_text = _build_user_text(message, current_dishes, history, entities_index)
    system = load_prompt(_PROMPT)
    messages = [{"role": "user", "content": [{"text": user_text}]}]

    try:
        raw = bedrock_client.converse(
            config.NOVA_LITE_MODEL_ID,
            system,
            messages,
            inference_config={
                "maxTokens": config.CLASSIFIER_MAX_TOKENS,
                "temperature": 0.0,
            },
        )
    except bedrock_client.BedrockError as e:
        logger.warning("classifier_bedrock_error", extra={"error": str(e)})
        return ClassifierResult(intent="fallback", current_dishes=current_dishes)

    return _parse(raw, current_dishes)


def _build_user_text(
    message: str,
    current_dishes: list[str],
    history: list[dict[str, str]],
    entities_index: dict[str, str],
) -> str:
    hist_lines: list[str] = []
    for h in history:
        role = "usuario" if h.get("role") == "user" else "agente"
        text = str(h.get("text", "")).strip().replace("\n", " ")
        if len(text) > 300:
            text = text[:297] + "…"
        hist_lines.append(f"  {role}: {text}")
    hist_block = "\n".join(hist_lines) if hist_lines else "(sin historial)"

    canonicals = sorted(set(entities_index.values()))
    entities_block = ", ".join(canonicals) if canonicals else "(ninguno)"

    return (
        f"Platillos en contexto actual (current_dishes): {current_dishes}\n\n"
        f"Entidades canónicas en KB: {entities_block}\n\n"
        f"Historial (cronológico, más antiguo arriba):\n{hist_block}\n\n"
        f"Mensaje actual del usuario: \"{message}\"\n\n"
        "Devuelve únicamente el JSON."
    )


def _parse(raw: str, current_dishes: list[str]) -> ClassifierResult:
    try:
        data = bedrock_client.parse_json_strict(raw)
    except Exception as e:
        logger.warning(
            "classifier_parse_error",
            extra={"error": str(e), "raw": raw[:200]},
        )
        return ClassifierResult(intent="fallback", current_dishes=current_dishes)

    if not isinstance(data, dict):
        return ClassifierResult(intent="fallback", current_dishes=current_dishes)

    intent = data.get("intent", "fallback")
    if intent not in _VALID_INTENTS:
        logger.warning(
            "classifier_unknown_intent",
            extra={"raw_intent": str(intent)[:64]},
        )
        intent = "fallback"

    if intent != "traduccion":
        return ClassifierResult(intent=intent, current_dishes=[])

    raw_dishes = data.get("current_dishes") or []
    dishes: list[str] = []
    if isinstance(raw_dishes, list):
        for d in raw_dishes:
            if not isinstance(d, str):
                continue
            d_clean = d.strip().lower()
            if d_clean:
                dishes.append(d_clean)

    translate_now = bool(data.get("translate_now", False))

    logger.info(
        "classifier_result",
        extra={
            "intent": intent,
            "current_dishes": dishes,
            "translate_now": translate_now,
            "reasoning": str(data.get("reasoning", ""))[:500],
        },
    )

    return ClassifierResult(
        intent=intent,
        current_dishes=dishes,
        translate_now=translate_now,
    )
