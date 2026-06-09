from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final

import bedrock_client
import config
import llm_schemas
import timing
from prompt_loader import load_prompt
from retrieval import get_entities_index

logger = logging.getLogger(__name__)

_CLASSIFIER_PROMPT: Final[str] = "classifier_system.txt"
_EXTRACTOR_PROMPT: Final[str] = "extractor_system.txt"

_VALID_INTENTS: Final[frozenset[str]] = frozenset({
    "traduccion", "maps", "yelp", "tripadvisor", "higiene",
    "fundacion_placemaking", "menu_del_dia", "organizaciones_participantes",
    "primera_edicion", "talleres", "beneficios_negocio", "contacto",
    "etribe", "out_of_domain", "fallback",
})
_VALID_PLATFORMS: Final[frozenset[str]] = frozenset({"google_maps", "yelp", "tripadvisor"})


@dataclass
class PendingSlot:
    """Kept for backward compat — non-traduccion path in generation._build_user_text."""
    entity: str
    slot_name: str
    options: list[str] = field(default_factory=list)


@dataclass
class ClassifierResult:
    intent: str
    # New traduccion fields
    current_dish: str = ""
    companions: list[str] = field(default_factory=list)
    # Legacy fields — used by non-traduccion generation path
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
    current_dish: str,
    history: list[dict[str, str]],
    dish_status: str | None = None,
) -> ClassifierResult:
    intent, platform = _classify_intent(message, history)

    if intent == "traduccion":
        return _extract_traduccion(message, current_dish, history, intent, dish_status)

    return ClassifierResult(intent=intent, platform=platform)


# ── Stage 1: intent classification ───────────────────────────────────────────

def _classify_intent(
    message: str,
    history: list[dict[str, str]],
) -> tuple[str, str]:
    user_text = _build_classifier_text(message, history)
    system = load_prompt(_CLASSIFIER_PROMPT)
    messages = [{"role": "user", "content": [{"text": user_text}]}]

    try:
        data = bedrock_client.converse_json(
            config.NOVA_2_LITE_MODEL_ID,
            system,
            messages,
            schema=llm_schemas.CLASSIFIER_INTENT,
            tool_name="classify_intent",
            tool_description="Classify the user message intent",
            inference_config={"maxTokens": 256, "temperature": 0.0},
            stage="classifier_intent",
        )
    except bedrock_client.BedrockError as e:
        logger.warning("classifier_bedrock_error", extra={"error": str(e)})
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
    current_dish: str,
    history: list[dict[str, str]],
    intent: str,
    dish_status: str | None = None,
) -> ClassifierResult:
    with timing.stage("classifier.kb_load"):
        entities_index = get_entities_index()
        user_text = _build_extractor_text(message, current_dish, history, entities_index, dish_status)
    system = load_prompt(_EXTRACTOR_PROMPT)
    messages = [{"role": "user", "content": [{"text": user_text}]}]

    try:
        data = bedrock_client.converse_json(
            config.NOVA_2_LITE_MODEL_ID,
            system,
            messages,
            schema=llm_schemas.EXTRACTOR_TRADUCCION,
            tool_name="extract_traduccion",
            tool_description="Extract dish and companions from the user message",
            inference_config={"maxTokens": config.CLASSIFIER_MAX_TOKENS, "temperature": 0.0},
            stage="extractor",
        )
    except bedrock_client.BedrockError as e:
        logger.warning("extractor_bedrock_error", extra={"error": str(e)})
        return ClassifierResult(intent=intent, current_dish=current_dish)

    return _parse_extraction_data(data, current_dish, intent)


def _build_extractor_text(
    message: str,
    current_dish: str,
    history: list[dict[str, str]],
    entities_index: dict[str, str],
    dish_status: str | None = None,
) -> str:
    hist_lines: list[str] = []
    for h in history[-6:]:
        role = "usuario" if h.get("role") == "user" else "agente"
        text = str(h.get("text", "")).strip().replace("\n", " ")
        if len(text) > 300:
            text = text[:297] + "…"
        hist_lines.append(f"  {role}: {text}")
    hist_block = "\n".join(hist_lines) if hist_lines else "(sin historial)"

    # Build canonical → sorted aliases map for the extractor
    canonical_to_aliases: dict[str, list[str]] = {}
    for alias, canonical in entities_index.items():
        canonical_to_aliases.setdefault(canonical, [])
        if alias != canonical:
            canonical_to_aliases[canonical].append(alias)
    entities_lines: list[str] = []
    for canonical in sorted(canonical_to_aliases.keys()):
        aliases = sorted(canonical_to_aliases[canonical])
        alias_str = ", ".join(aliases) if aliases else ""
        entities_lines.append(
            f"  {canonical}: {alias_str}" if alias_str else f"  {canonical}"
        )
    entities_block = "\n".join(entities_lines) if entities_lines else "  (ninguno)"

    current_dish_block = ""
    if current_dish:
        current_dish_block = (
            f"Platillo en curso (NO lo cambies salvo que el usuario mencione uno diferente): "
            f"{current_dish}\n\n"
        )

    # When there is an active translation flow, warn the extractor to be conservative
    # about changing the current dish — the user may be answering a variable question
    # even if the most recent history turn was a digression (general question answer).
    active_flow_hint = ""
    if current_dish and dish_status in ("EXTRACTING", "CONFIRMING_FLAGS", "EDITING"):
        active_flow_hint = (
            f"AVISO: el sistema está en etapa {dish_status} recopilando datos para '{current_dish}'. "
            f"Si el mensaje parece ser una respuesta sobre ingredientes, preparación o variantes "
            f"(ej: 'de pollo', 'con queso', 'rojo', 'sin relleno') → devuelve current_dish = \"\" "
            f"para conservar el platillo en curso. Solo devuelve un nuevo platillo si el usuario "
            f"claramente quiere cambiar de tema.\n\n"
        )

    return (
        f"{active_flow_hint}"
        f"{current_dish_block}"
        f"Platillos en KB (canónico: alias1, alias2, …):\n{entities_block}\n\n"
        f"Historial (cronológico, más antiguo arriba):\n{hist_block}\n\n"
        f"Mensaje actual del usuario: \"{message}\"\n\n"
        "Devuelve únicamente el JSON."
    )


def _parse_extraction_data(data: dict, current_dish: str, intent: str) -> ClassifierResult:
    if not isinstance(data, dict):
        return ClassifierResult(intent=intent, current_dish=current_dish)

    new_dish = str(data.get("current_dish", "")).strip().lower()

    raw_companions = data.get("companions") or []
    companions: list[str] = []
    if isinstance(raw_companions, list):
        for c in raw_companions:
            if isinstance(c, str) and c.strip():
                companions.append(c.strip().lower())

    logger.info(
        "extractor_result",
        extra={
            "intent": intent,
            "current_dish": new_dish,
            "companions": companions,
            "reasoning": str(data.get("reasoning", ""))[:300],
        },
    )

    return ClassifierResult(
        intent=intent,
        current_dish=new_dish,
        companions=companions,
        current_dishes=[new_dish] if new_dish else [],
    )


def _parse_extraction(raw: str, current_dish: str, intent: str) -> ClassifierResult:
    """Backward-compatible wrapper for tests that pass raw JSON strings."""
    try:
        data = bedrock_client.parse_json_strict(raw)
    except Exception as e:
        logger.warning("extractor_parse_error", extra={"error": str(e), "raw": raw[:200]})
        return ClassifierResult(intent=intent, current_dish=current_dish)
    return _parse_extraction_data(data, current_dish, intent)
