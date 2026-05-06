from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final, Literal

import bedrock_client
import config
from intents import INTENTS
from retrieval import get_entities_index
from state import CUSTOM_ENTITY, AgentState

logger = logging.getLogger(__name__)


UserSignal = Literal["approve", "reject", "want_translation"]
_VALID_SIGNALS: Final[frozenset[str]] = frozenset({"approve", "reject", "want_translation"})


@dataclass
class TurnDecision:
    """Decisión completa por turno: intención + (si aplica) extracción de
    cambios de estado para `traducir`. Un solo Nova Micro call hace ambos
    pasos para minimizar latencia."""

    intent: str
    intent_changed: bool
    dish_change: bool = False
    new_entity: str | None = None
    variant: str | None = None
    ingredients_added: list[str] = field(default_factory=list)
    user_signal: UserSignal | None = None


_SYSTEM: Final[str] = (
    "Eres un analizador de turnos para un agente que ayuda a fonderos de CDMX. "
    "Tu trabajo tiene DOS pasos.\n\n"
    "PASO 1 — Intención.\n"
    "Identifica la intención del último mensaje entre las disponibles. "
    "REGLA DE CONTINUIDAD (importante): si state.intent != null, MANTÉN esa "
    "intención salvo que el último mensaje contenga señal CLARA y EXPLÍCITA de "
    "cambio de tema. En duda, conserva la intención actual. intent_changed=true "
    "solo si la intención elegida difiere de state.intent.\n\n"
    "PASO 2 — Solo si intent='traducir', extrae los cambios de estado de la "
    "traducción a partir del último mensaje.\n"
    "- dish_change: true si el usuario está cambiando a un platillo DIFERENTE del "
    "que ya estaba trabajando. Si está dando más info del mismo platillo "
    "(ingredientes, variante, etc.), false.\n"
    "- new_entity: el alias canónico exacto del índice. Si el platillo no está en "
    "el índice usa '__custom__'. Si en este turno el usuario no menciona un "
    "platillo nuevo y ya hay uno en curso, devuelve null.\n"
    "- variant: el subtipo del platillo, implícito o explícito. Ejemplos:\n"
    "    'arroz rojo' -> new_entity='arroz', variant='rojo'\n"
    "    'mole poblano' -> new_entity='mole', variant='poblano'\n"
    "    'mole verde de mi tía' -> new_entity='mole', variant='verde'\n"
    "  Si no aplica, null.\n"
    "- ingredients_added: lista de ingredientes NUEVOS que el usuario agrega EN "
    "ESTE TURNO. Normaliza a snake_case y singular cuando aplique. Si no agregó "
    "ingredientes en este turno, devuelve []. NO repitas los ya presentes en "
    "current_dish.user_ingredients.\n"
    "  Ejemplos:\n"
    "    'le pongo jitomate, cebolla y caldo de pollo' -> "
    "['jitomate','cebolla','caldo_de_pollo']\n"
    "    'ya está, tradúcelo' -> [] (no agrega ingredientes)\n"
    "- user_signal:\n"
    "  - 'approve': el usuario aprueba la última traducción.\n"
    "  - 'reject': la rechaza.\n"
    "  - 'want_translation': pide traducir ya o señala que terminó de dar info.\n"
    "  - null si el mensaje es información, pregunta, off-topic o no es claro.\n"
    "  IMPORTANTE: 'no mucho' o 'no muchas cosas' NO son rechazo; son respuestas "
    "a una pregunta abierta. user_signal solo aplica a aprobación/rechazo de la "
    "última TRADUCCIÓN si existe.\n\n"
    "Si intent != 'traducir': dish_change=false, new_entity=null, variant=null, "
    "ingredients_added=[], user_signal=null.\n\n"
    "OUTPUT: SOLO este JSON, sin markdown, sin texto extra:\n"
    "{\n"
    '  "intent": "<intent>",\n'
    '  "intent_changed": <bool>,\n'
    '  "dish_change": <bool>,\n'
    '  "new_entity": "<canonical>" | "__custom__" | null,\n'
    '  "variant": "<variante>" | null,\n'
    '  "ingredients_added": ["<ing1>", ...],\n'
    '  "user_signal": "approve" | "reject" | "want_translation" | null\n'
    "}\n"
    "NO inventes datos. Si dudas, devuelve null/[]/false."
)


def classify(
    state: AgentState,
    message: str,
    history: list[dict[str, str]],
    image_summary: str | None,
) -> TurnDecision:
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
        return TurnDecision(intent=state.intent or "fallback", intent_changed=False)

    return _parse(raw, state.intent)


def _build_user_text(
    state: AgentState,
    message: str,
    history: list[dict[str, str]],
    image_summary: str | None,
) -> str:
    intents_block = "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())

    index = get_entities_index()
    aliases_block = (
        "\n".join(f"- {alias} -> {canonical}" for alias, canonical in index.items())
        if index
        else "(índice vacío)"
    )

    history_block = (
        "\n".join(f"- {h.get('role', '?')}: {h.get('text', '')}" for h in history)
        or "(sin historial)"
    )

    cd = state.current_dish
    last_completed = state.completed[-1] if state.completed else None
    state_block = (
        f"intent={state.intent} | mode={state.mode}\n"
        f"current_dish.entity = {cd.entity if cd else 'null'}\n"
        f"current_dish.variant = {cd.variant if cd else 'null'}\n"
        f"current_dish.user_ingredients = {cd.user_ingredients if cd else []}\n"
        f"last_completed_translation_en = "
        f"{last_completed.translation_en if last_completed else 'null'}\n"
        f"last_completed_approved = "
        f"{last_completed.approved if last_completed else 'null'}"
    )

    image_block = (
        f"Resumen de imagen recién analizada: {image_summary}\n\n"
        if image_summary
        else ""
    )

    return (
        f"Intenciones disponibles:\n{intents_block}\n\n"
        f"Índice de entidades canónicas (KB de platillos):\n{aliases_block}\n\n"
        f"Estado actual:\n{state_block}\n\n"
        f"Últimos turnos (máx {config.MAX_HISTORY_TURNS}):\n{history_block}\n\n"
        f"{image_block}"
        f"Último mensaje del usuario:\n\"{message}\"\n\n"
        "Devuelve solo el JSON especificado."
    )


def _parse(raw: str, current_intent: str | None) -> TurnDecision:
    try:
        data = bedrock_client.parse_json_strict(raw)
    except Exception as e:
        logger.warning("classifier_parse_error", extra={"error": str(e), "raw": raw[:200]})
        return TurnDecision(intent=current_intent or "fallback", intent_changed=False)

    if not isinstance(data, dict):
        return TurnDecision(intent=current_intent or "fallback", intent_changed=False)

    intent = data.get("intent")
    if intent not in INTENTS:
        logger.warning("classifier_unknown_intent", extra={"raw_intent": str(intent)[:64]})
        intent = "fallback"

    intent_changed = bool(data.get("intent_changed", False)) and intent != current_intent

    if intent != "traducir":
        return TurnDecision(intent=intent, intent_changed=intent_changed)

    canonicals = set((get_entities_index() or {}).values())

    new_entity = data.get("new_entity")
    if isinstance(new_entity, str):
        if new_entity != CUSTOM_ENTITY and new_entity not in canonicals:
            new_entity = None
    else:
        new_entity = None

    variant = data.get("variant")
    if not isinstance(variant, str) or not variant.strip():
        variant = None
    else:
        variant = variant.strip().lower()

    raw_ings = data.get("ingredients_added") or []
    ingredients: list[str] = []
    if isinstance(raw_ings, list):
        for i in raw_ings:
            if not isinstance(i, (str, int, float)):
                continue
            s = str(i).strip().lower()
            if s and s not in ingredients:
                ingredients.append(s)

    signal = data.get("user_signal")
    if not isinstance(signal, str) or signal not in _VALID_SIGNALS:
        signal = None

    return TurnDecision(
        intent=intent,
        intent_changed=intent_changed,
        dish_change=bool(data.get("dish_change", False)),
        new_entity=new_entity,
        variant=variant,
        ingredients_added=ingredients,
        user_signal=signal,
    )
