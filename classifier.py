from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import bedrock_client
import config
from intents import INTENTS
from state import AgentState

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    intent: str
    intent_changed: bool


_SYSTEM: Final[str] = (
    "Eres un clasificador de intenciones para un agente que ayuda a fonderos de CDMX. "
    "Recibes un resumen del estado de la conversación, los últimos turnos y el "
    "último mensaje del usuario. "
    "Devuelves SOLO un JSON con esta forma exacta y nada más: "
    '{"intent": "<una de las intenciones>", "intent_changed": <true|false>}. '
    "Sin markdown, sin explicaciones, sin texto extra. "
    "REGLA DE CONTINUIDAD (importante): si el state.intent no es null, "
    "MANTÉN esa intención salvo que el último mensaje contenga señal CLARA y EXPLÍCITA "
    "de cambio de tema (cambio de tópico evidente, palabras como 'ahora ayúdame con...', "
    "tema completamente distinto al actual). En duda, conserva la intención actual. "
    "intent_changed es true únicamente si la intención elegida difiere de state.intent."
)


def classify(
    state: AgentState,
    message: str,
    history: list[dict[str, str]],
    image_summary: str | None,
) -> ClassificationResult:
    user_text = _build_user_text(state, message, history, image_summary)
    messages = [{"role": "user", "content": [{"text": user_text}]}]

    try:
        raw = bedrock_client.converse(
            config.NOVA_MICRO_MODEL_ID,
            _SYSTEM,
            messages,
            inference_config={
                "maxTokens": config.CLASSIFIER_MAX_TOKENS,
                "temperature": 0.0,
            },
        )
    except bedrock_client.BedrockError as e:
        logger.warning("classifier_bedrock_error", extra={"error": str(e)})
        return ClassificationResult(intent=state.intent or "fallback", intent_changed=False)

    return _parse(raw, state.intent)


def _build_user_text(
    state: AgentState,
    message: str,
    history: list[dict[str, str]],
    image_summary: str | None,
) -> str:
    intents_block = "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())

    history_block = (
        "\n".join(f"- {h.get('role', '?')}: {h.get('text', '')}" for h in history)
        or "(sin historial)"
    )

    cd = state.current_dish
    state_block = (
        f"intent={state.intent} | mode={state.mode} | "
        f"current_dish={cd.entity if cd else None}"
    )

    image_block = (
        f"Resumen de imagen recién analizada: {image_summary}\n\n"
        if image_summary
        else ""
    )

    return (
        f"Intenciones disponibles:\n{intents_block}\n\n"
        f"Estado actual:\n{state_block}\n\n"
        f"Últimos turnos (máx {config.MAX_HISTORY_TURNS}):\n{history_block}\n\n"
        f"{image_block}"
        f"Último mensaje del usuario:\n\"{message}\"\n\n"
        "Responde solo con el JSON especificado."
    )


def _parse(raw: str, current_intent: str | None) -> ClassificationResult:
    try:
        data = bedrock_client.parse_json_strict(raw)
    except Exception as e:
        logger.warning("classifier_parse_error", extra={"error": str(e), "raw": raw[:200]})
        return ClassificationResult(intent=current_intent or "fallback", intent_changed=False)

    if not isinstance(data, dict):
        return ClassificationResult(intent=current_intent or "fallback", intent_changed=False)

    intent = data.get("intent")
    if intent not in INTENTS:
        logger.warning("classifier_unknown_intent", extra={"raw_intent": str(intent)[:64]})
        intent = "fallback"

    raw_changed = data.get("intent_changed", False)
    intent_changed = bool(raw_changed) and intent != current_intent

    return ClassificationResult(intent=intent, intent_changed=intent_changed)
