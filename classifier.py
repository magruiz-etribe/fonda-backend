from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Final

import bedrock_client
import config
from prompt_loader import load_prompt
from retrieval import (
    get_dish_data,
    get_entities_index,
    get_entities_with_variants,
    get_variant_keys_for_slot,
)

logger = logging.getLogger(__name__)

_CLASSIFIER_PROMPT: Final[str] = "classifier_system.txt"
_EXTRACTOR_PROMPT: Final[str] = "extractor_system.txt"

_VALID_INTENTS: Final[frozenset[str]] = frozenset({
    "traduccion", "maps", "yelp", "tripadvisor", "higiene",
    "fundacion_placemaking", "menu_del_dia", "organizaciones_participantes",
    "primera_edicion", "talleres", "beneficios_negocio", "contacto",
    "fallback",
})
_VALID_PLATFORMS: Final[frozenset[str]] = frozenset({"google_maps", "yelp", "tripadvisor"})


@dataclass
class PendingSlot:
    entity: str
    slot_name: str  # "variant" | "filling" | "sauce"
    options: list[str] = field(default_factory=list)


@dataclass
class ClassifierResult:
    intent: str
    current_dishes: list[str] = field(default_factory=list)
    translate_now: bool = False
    pending_slots: list[PendingSlot] = field(default_factory=list)
    resolved_variants: dict[str, str] = field(default_factory=dict)
    extra_user_ingredients: list[str] = field(default_factory=list)
    platform: str = ""

    @property
    def pending_variant_for(self) -> str | None:
        for slot in self.pending_slots:
            if slot.slot_name == "variant":
                return slot.entity
        return None


def classify(
    message: str,
    current_dishes: list[str],
    history: list[dict[str, str]],
    dish_context: dict | None = None,
) -> ClassifierResult:
    intent, platform = _classify_intent(message, history)

    if intent == "traduccion":
        return _extract_traduccion(message, current_dishes, history, dish_context, intent)

    return ClassifierResult(intent=intent, current_dishes=[], platform=platform)


# ── Stage 1: intent classification ───────────────────────────────────────────

def _classify_intent(
    message: str,
    history: list[dict[str, str]],
) -> tuple[str, str]:
    user_text = _build_classifier_text(message, history)
    system = load_prompt(_CLASSIFIER_PROMPT)
    messages = [{"role": "user", "content": [{"text": user_text}]}]

    try:
        raw = bedrock_client.converse(
            config.NOVA_2_LITE_MODEL_ID,
            system,
            messages,
            inference_config={"maxTokens": 256, "temperature": 0.0},
        )
    except bedrock_client.BedrockError as e:
        logger.warning("classifier_bedrock_error", extra={"error": str(e)})
        return "fallback", ""

    try:
        data = bedrock_client.parse_json_strict(raw)
    except Exception as e:
        logger.warning("classifier_parse_error", extra={"error": str(e), "raw": raw[:200]})
        return "fallback", ""

    intent = str(data.get("intent", "fallback")).strip().lower()
    if intent not in _VALID_INTENTS:
        logger.warning("classifier_unknown_intent", extra={"raw_intent": intent[:64]})
        intent = "fallback"

    platform = ""
    if intent == "maps":
        raw_platform = str(data.get("platform", "")).strip().lower()
        platform = raw_platform if raw_platform in _VALID_PLATFORMS else ""

    logger.info("classifier_intent", extra={"intent": intent, "platform": platform,
                                            "reasoning": str(data.get("reasoning", ""))[:300]})
    return intent, platform


def _build_classifier_text(message: str, history: list[dict[str, str]]) -> str:
    hist_lines: list[str] = []
    for h in history[-6:]:
        role = "usuario" if h.get("role") == "user" else "agente"
        text = str(h.get("text", "")).strip().replace("\n", " ")
        if len(text) > 300:
            text = text[:297] + "…"
        hist_lines.append(f"  {role}: {text}")
    hist_block = "\n".join(hist_lines) if hist_lines else "(sin historial)"

    return (
        f"Historial (cronológico, más antiguo arriba):\n{hist_block}\n\n"
        f"Mensaje actual del usuario: \"{message}\"\n\n"
        "Devuelve únicamente el JSON."
    )


# ── Stage 2: traduccion extraction ───────────────────────────────────────────

def _extract_traduccion(
    message: str,
    current_dishes: list[str],
    history: list[dict[str, str]],
    dish_context: dict | None,
    intent: str,
) -> ClassifierResult:
    entities_index = get_entities_index()
    entities_with_variants = get_entities_with_variants()
    user_text = _build_extractor_text(
        message, current_dishes, history, entities_index, entities_with_variants, dish_context
    )
    system = load_prompt(_EXTRACTOR_PROMPT)
    messages = [{"role": "user", "content": [{"text": user_text}]}]

    try:
        raw = bedrock_client.converse(
            config.NOVA_2_LITE_MODEL_ID,
            system,
            messages,
            inference_config={"maxTokens": config.CLASSIFIER_MAX_TOKENS, "temperature": 0.0},
        )
    except bedrock_client.BedrockError as e:
        logger.warning("extractor_bedrock_error", extra={"error": str(e)})
        return ClassifierResult(intent=intent, current_dishes=current_dishes)

    return _parse_extraction(raw, current_dishes, intent)


def _build_extractor_text(
    message: str,
    current_dishes: list[str],
    history: list[dict[str, str]],
    entities_index: dict[str, str],
    entities_with_variants: list[str],
    dish_context: dict | None = None,
) -> str:
    hist_lines: list[str] = []
    for h in history[-6:]:
        role = "usuario" if h.get("role") == "user" else "agente"
        text = str(h.get("text", "")).strip().replace("\n", " ")
        if len(text) > 300:
            text = text[:297] + "…"
        hist_lines.append(f"  {role}: {text}")
    hist_block = "\n".join(hist_lines) if hist_lines else "(sin historial)"

    canonicals = sorted(set(entities_index.values()))
    entities_block = ", ".join(canonicals) if canonicals else "(ninguno)"
    variants_block = ", ".join(entities_with_variants) if entities_with_variants else "(ninguno)"

    dish_ctx_block = ""
    if dish_context:
        dc = dish_context
        rv = json.dumps(dc.get("resolved_variants") or {}, ensure_ascii=False)
        extras = ", ".join(dc.get("extra_ingredients") or []) or "(ninguno)"
        dish_ctx_block = (
            "[CONTEXTO DEL PLATILLO EN CURSO — fuente de verdad]\n"
            f"Platillo principal: {dc.get('main_dish', '')}\n"
            f"Variantes confirmadas: {rv}\n"
            f"Ingredientes extras confirmados: {extras}\n\n"
        )

    return (
        f"{dish_ctx_block}"
        f"Platillos en contexto actual (current_dishes): {current_dishes}\n\n"
        f"Entidades canónicas en KB: {entities_block}\n\n"
        f"Entidades con variantes en KB: {variants_block}\n\n"
        f"Historial (cronológico, más antiguo arriba):\n{hist_block}\n\n"
        f"Mensaje actual del usuario: \"{message}\"\n\n"
        "Devuelve únicamente el JSON."
    )


def _parse_extraction(raw: str, current_dishes: list[str], intent: str) -> ClassifierResult:
    try:
        data = bedrock_client.parse_json_strict(raw)
    except Exception as e:
        logger.warning("extractor_parse_error", extra={"error": str(e), "raw": raw[:200]})
        return ClassifierResult(intent=intent, current_dishes=current_dishes)

    if not isinstance(data, dict):
        return ClassifierResult(intent=intent, current_dishes=current_dishes)

    raw_dishes = data.get("current_dishes") or []
    dishes: list[str] = []
    if isinstance(raw_dishes, list):
        for d in raw_dishes:
            if isinstance(d, str):
                d_clean = d.strip().lower()
                if d_clean:
                    dishes.append(d_clean)

    translate_now = bool(data.get("translate_now", False))

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
                options = get_variant_keys_for_slot(entity)
            pending_slots.append(PendingSlot(entity=entity, slot_name=slot_name, options=options))

    raw_rv = data.get("resolved_variants") or {}
    resolved_variants: dict[str, str] = {}
    if isinstance(raw_rv, dict):
        for k, v in raw_rv.items():
            if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                resolved_variants[k.lower().strip()] = v.lower().strip()

    raw_eui = data.get("extra_user_ingredients") or []
    extra_user_ingredients: list[str] = []
    if isinstance(raw_eui, list):
        for ing in raw_eui:
            if isinstance(ing, str) and ing.strip():
                extra_user_ingredients.append(ing.strip().lower())

    logger.info(
        "extractor_result",
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
