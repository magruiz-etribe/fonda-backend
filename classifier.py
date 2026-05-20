from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final

import bedrock_client
import config
from prompt_loader import load_prompt
from retrieval import get_dish_data, get_entities_index, get_entities_with_variants

logger = logging.getLogger(__name__)

_PROMPT: Final[str] = "classifier_system.txt"
_VALID_INTENTS: Final[frozenset[str]] = frozenset(
    {"traduccion", "maps", "higiene", "fallback"}
)


@dataclass
class PendingSlot:
    entity: str
    slot_name: str  # "variant" | "filling" | "sauce" | "protein" | "accompaniment"
    options: list[str] = field(default_factory=list)


@dataclass
class ClassifierResult:
    intent: str
    current_dishes: list[str] = field(default_factory=list)
    translate_now: bool = False
    pending_slots: list[PendingSlot] = field(default_factory=list)
    resolved_variants: dict[str, str] = field(default_factory=dict)
    extra_user_ingredients: list[str] = field(default_factory=list)

    @property
    def pending_variant_for(self) -> str | None:
        """Backward-compat property: first pending variant slot entity, or None."""
        for slot in self.pending_slots:
            if slot.slot_name == "variant":
                return slot.entity
        return None


def classify(
    message: str,
    current_dishes: list[str],
    history: list[dict[str, str]],
) -> ClassifierResult:
    entities_index = get_entities_index()
    entities_with_variants = get_entities_with_variants()
    user_text = _build_user_text(message, current_dishes, history, entities_index, entities_with_variants)
    system = load_prompt(_PROMPT)
    messages = [{"role": "user", "content": [{"text": user_text}]}]

    try:
        raw = bedrock_client.converse(
            config.NOVA_2_LITE_MODEL_ID,
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
    entities_with_variants: list[str],
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
    variants_block = ", ".join(entities_with_variants) if entities_with_variants else "(ninguno)"

    return (
        f"Platillos en contexto actual (current_dishes): {current_dishes}\n\n"
        f"Entidades canónicas en KB: {entities_block}\n\n"
        f"Entidades con variantes en KB: {variants_block}\n\n"
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

    # Parse pending_slots — options are code-populated from YAML, not from the LLM
    raw_slots = data.get("pending_slots") or []
    pending_slots: list[PendingSlot] = []
    if isinstance(raw_slots, list):
        for slot_data in raw_slots:
            if not isinstance(slot_data, dict):
                continue
            entity = slot_data.get("entity", "")
            if not isinstance(entity, str) or not entity.strip():
                continue
            entity = entity.strip().lower()
            slot_name = str(slot_data.get("slot_name", "variant")).lower().strip()
            options: list[str] = []
            if slot_name == "variant":
                dish_data = get_dish_data(entity)
                if dish_data:
                    options = list(dish_data.get("variants", {}).keys())
            pending_slots.append(PendingSlot(entity=entity, slot_name=slot_name, options=options))

    # Parse resolved_variants
    raw_rv = data.get("resolved_variants") or {}
    resolved_variants: dict[str, str] = {}
    if isinstance(raw_rv, dict):
        for k, v in raw_rv.items():
            if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                resolved_variants[k.lower().strip()] = v.lower().strip()

    # Parse extra_user_ingredients
    raw_eui = data.get("extra_user_ingredients") or []
    extra_user_ingredients: list[str] = []
    if isinstance(raw_eui, list):
        for ing in raw_eui:
            if isinstance(ing, str) and ing.strip():
                extra_user_ingredients.append(ing.strip().lower())

    logger.info(
        "classifier_result",
        extra={
            "intent": intent,
            "current_dishes": dishes,
            "translate_now": translate_now,
            "pending_slots": [(s.entity, s.slot_name) for s in pending_slots],
            "resolved_variants": resolved_variants,
            "reasoning": str(data.get("reasoning", ""))[:500],
        },
    )

    return ClassifierResult(
        intent=intent,
        current_dishes=dishes,
        translate_now=translate_now,
        pending_slots=pending_slots,
        resolved_variants=resolved_variants,
        extra_user_ingredients=extra_user_ingredients,
    )
