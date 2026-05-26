from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Any, Final

import classifier as cls_module
import flag_llm
import generation as gen_module
import retrieval
from generation import GenResult

logger = logging.getLogger(__name__)

_FALLBACK_RESULT: Final[GenResult] = GenResult(
    response=["Disculpa, tuve un problema procesando tu mensaje. ¿Puedes intentarlo de nuevo? 😊"],
    current_dishes=[],
    buttons=[],
)

_MAPS_PAGE_LINK: Final[dict[str, Any]] = {
    "label": "Regístrate en Google Maps",
    "url": "https://business.google.com/es-all/business-profile/?ppsrc=GPDA2",
    "type": "page",
}

_MAPS_PDF_LINK: Final[dict[str, Any]] = {
    "label": "Guía: Menú del Día en Google Maps",
    "url": "https://d1b1gcigbjwv2n.cloudfront.net/Men%C3%BA%20del%20D%C3%ADa%20-%20Google%20Maps.pdf",
    "type": "pdf",
}

_YELP_PDF_LINK: Final[dict[str, Any]] = {
    "label": "Guía: Menú del Día en Yelp",
    "url": "https://d1b1gcigbjwv2n.cloudfront.net/Men%C3%BA%20del%20D%C3%ADa%20-%20Yelp.pdf",
    "type": "pdf",
}

_TRIPADVISOR_PDF_LINK: Final[dict[str, Any]] = {
    "label": "Guía: Menú del Día en TripAdvisor",
    "url": "https://d1b1gcigbjwv2n.cloudfront.net/Men%C3%BA%20del%20D%C3%ADa%20-%20Tripadvisor.pdf",
    "type": "pdf",
}

_PLATFORM_LINKS: Final[dict[str, list[dict]]] = {
    "google_maps": [_MAPS_PAGE_LINK, _MAPS_PDF_LINK],
    "yelp": [_YELP_PDF_LINK],
    "tripadvisor": [_TRIPADVISOR_PDF_LINK],
}

# Phrases that unambiguously mean "done, nothing to add" in A1 context.
_COMPLETION_PHRASES: frozenset[str] = frozenset({
    "✅ listo, eso es todo!",
    "✅ listo, eso es todo",
    "listo, eso es todo!",
    "listo, eso es todo",
    "listo",
    "eso es todo",
    "claro",
    "ninguno",
    "correcto",
    "exacto",
    "no",
    "si",
    "sí",
    "ok",
    "✅",
    "❌",
})

_CONFIRM_BTNS: Final[list[str]] = ["✅ Sí, contiene alguno", "❌ No, ninguno de esos"]


def handle(
    message: str,
    current_dishes: list[str],
    history: list[dict[str, str]],
    confirmation_state: dict | None = None,
) -> GenResult:
    try:
        cr = cls_module.classify(message, current_dishes, history)
        cr = _enrich_classifier_from_conversation(cr, message, history)
        kb_context = _get_kb_context(cr)
        dish_flags: dict = {}
        conf_state_for_gen = None
        trigger_info_for_gen = None
        if cr.intent == "traduccion" and cr.current_dishes:
            dish_flags = _compute_dish_flags(cr, message, history, kb_context)
            logger.info("computed_flags", extra={"flags": dish_flags, "dishes": cr.current_dishes})
            kb_context = _append_flags_to_context(kb_context, dish_flags)
            conf_state_for_gen = confirmation_state
            trigger_info_for_gen = {
                k: dish_flags.get(k, [])
                for k in ("allergen_triggers", "gluten_triggers", "spicy_triggers")
            }
        if (cr.intent == "traduccion" and cr.current_dishes
                and not cr.pending_slots and not cr.translate_now
                and confirmation_state is not None):
            short = _try_short_circuit(cr, confirmation_state, trigger_info_for_gen or {}, message)
            if short is not None:
                short.intent = cr.intent
                short.flags = _clean_flags(dish_flags)
                return short

        result = gen_module.generate(cr, message, kb_context, history, conf_state_for_gen, trigger_info_for_gen, cr.platform)
        result.intent = cr.intent
        result.flags = _clean_flags(dish_flags)
        if cr.intent == "maps":
            result.links = _PLATFORM_LINKS.get(cr.platform, [])
        if cr.translate_now:
            result.current_dishes = []  # always clear after translation (LLM sometimes forgets)
            result.menu_entry = _build_menu_entry(result, history, dish_flags)
        return result
    except Exception as e:
        logger.exception("router_unhandled_exception", extra={"error": str(e)})
        return _FALLBACK_RESULT


def _is_pure_completion(message: str) -> bool:
    """True when the message is an unambiguous 'done / that's all' with no ingredient additions."""
    normalized = message.strip().rstrip("!.?, ").lower()
    return normalized in _COMPLETION_PHRASES


def _is_yes(message: str) -> bool:
    """True for affirmative A2/A3/A4 responses, False for negative."""
    s = message.strip().lower()
    return not (s.startswith("❌") or re.match(r"^(no\b|ninguno)", s))


def _fmt_trigger_list(triggers: list[str]) -> str:
    return ", ".join(t.replace("_", " ") for t in triggers)


def _ask_a2(triggers: list[str]) -> str:
    tl = _fmt_trigger_list(triggers)
    return (
        f"He detectado que tu platillo puede contener ingredientes alérgenos: **{tl}**. "
        "¿Confirmas que tu platillo tiene al menos uno de estos? 🌿"
    )


def _ask_a3(triggers: list[str]) -> str:
    tl = _fmt_trigger_list(triggers)
    return (
        f"He detectado que tu platillo puede contener algunos de estos ingredientes: **{tl}**. "
        "¿Confirmas que tu platillo tiene al menos uno? 🌾"
    )


def _ask_a4(triggers: list[str]) -> str:
    tl = _fmt_trigger_list(triggers)
    return (
        f"He detectado que tu platillo puede contener algunos de estos ingredientes: **{tl}**. "
        "¿Confirmas que tu platillo tiene al menos uno? 🌶️"
    )


def _try_short_circuit(
    cr: cls_module.ClassifierResult,
    confirmation_state: dict,
    trigger_info: dict,
    message: str,
) -> GenResult | None:
    """Return a deterministic GenResult for A1–A4 confirmation stages, or None to call LLM."""
    cs = confirmation_state
    allergen_tr: list[str] = trigger_info.get("allergen_triggers") or []
    gluten_tr: list[str] = trigger_info.get("gluten_triggers") or []
    spicy_tr: list[str] = trigger_info.get("spicy_triggers") or []

    # ── A1 PROCESA RESPUESTA ─────────────────────────────────────────────────
    # User responded with an unambiguous "done" to the A1 completeness question.
    if cs.get("completeness_confirmed") is None and _is_pure_completion(message):
        if allergen_tr:
            return GenResult(
                response=[_ask_a2(allergen_tr)],
                current_dishes=cr.current_dishes,
                buttons=_CONFIRM_BTNS,
                completeness_confirmed=True,
            )
        if gluten_tr:
            return GenResult(
                response=[_ask_a3(gluten_tr)],
                current_dishes=cr.current_dishes,
                buttons=_CONFIRM_BTNS,
                completeness_confirmed=True,
            )
        if spicy_tr:
            return GenResult(
                response=[_ask_a4(spicy_tr)],
                current_dishes=cr.current_dishes,
                buttons=_CONFIRM_BTNS,
                completeness_confirmed=True,
            )
        # No triggers at all → fall through to LLM for ETAPA B (Spanish description).
        return None

    # Stages A2–A4 only apply once completeness is confirmed.
    if not cs.get("completeness_confirmed"):
        return None

    allergens_handled = (not allergen_tr) or (cs.get("allergens_confirmed") is not None)
    gluten_handled = (not gluten_tr) or (cs.get("gluten_confirmed") is not None)
    is_responding = gen_module._is_confirmation(message)

    # ── A2 ───────────────────────────────────────────────────────────────────
    if allergen_tr and cs.get("allergens_confirmed") is None:
        if not is_responding:
            return GenResult(
                response=[_ask_a2(allergen_tr)],
                current_dishes=cr.current_dishes,
                buttons=_CONFIRM_BTNS,
            )
        confirmed = _is_yes(message)
        if gluten_tr:
            return GenResult(
                response=[_ask_a3(gluten_tr)],
                current_dishes=cr.current_dishes,
                buttons=_CONFIRM_BTNS,
                allergens_confirmed=confirmed,
            )
        if spicy_tr:
            return GenResult(
                response=[_ask_a4(spicy_tr)],
                current_dishes=cr.current_dishes,
                buttons=_CONFIRM_BTNS,
                allergens_confirmed=confirmed,
            )
        return None  # next is ETAPA B — let LLM generate description

    # ── A3 ───────────────────────────────────────────────────────────────────
    if allergens_handled and gluten_tr and cs.get("gluten_confirmed") is None:
        if not is_responding:
            return GenResult(
                response=[_ask_a3(gluten_tr)],
                current_dishes=cr.current_dishes,
                buttons=_CONFIRM_BTNS,
            )
        confirmed = _is_yes(message)
        if spicy_tr:
            return GenResult(
                response=[_ask_a4(spicy_tr)],
                current_dishes=cr.current_dishes,
                buttons=_CONFIRM_BTNS,
                gluten_confirmed=confirmed,
            )
        return None  # next is ETAPA B — let LLM generate description

    # ── A4 ───────────────────────────────────────────────────────────────────
    if allergens_handled and gluten_handled and spicy_tr and cs.get("spicy_confirmed") is None:
        if not is_responding:
            return GenResult(
                response=[_ask_a4(spicy_tr)],
                current_dishes=cr.current_dishes,
                buttons=_CONFIRM_BTNS,
            )
        # A4 PROCESA RESPUESTA → ETAPA B: let LLM generate description.
        return None

    return None


def _get_kb_context(cr: cls_module.ClassifierResult) -> str:
    if cr.intent == "traduccion":
        return retrieval.get_context_for_dishes(cr.current_dishes)
    if cr.intent in ("maps", "higiene"):
        return retrieval.get_static(cr.intent)  # type: ignore[arg-type]
    return ""


def _enrich_classifier_from_conversation(
    cr: cls_module.ClassifierResult,
    message: str,
    history: list[dict[str, str]],
) -> cls_module.ClassifierResult:
    """Fill resolved_variants from conversation and drop redundant variant slots."""
    if cr.intent != "traduccion" or not cr.current_dishes:
        return cr

    conversation = retrieval.conversation_text(message, history)
    resolved = retrieval.resolve_variants_from_conversation(
        cr.current_dishes,
        cr.resolved_variants,
        conversation,
    )
    pending = [
        slot
        for slot in cr.pending_slots
        if not (slot.slot_name == "variant" and slot.entity in resolved)
    ]
    if resolved == cr.resolved_variants and pending == cr.pending_slots:
        return cr

    logger.info(
        "classifier_enriched",
        extra={
            "resolved_variants": resolved,
            "pending_slots": [(s.entity, s.slot_name) for s in pending],
        },
    )
    return replace(cr, resolved_variants=resolved, pending_slots=pending)


def _compute_dish_flags(
    cr: cls_module.ClassifierResult,
    message: str,
    history: list[dict[str, str]],
    kb_context: str,
) -> dict:
    """Compute dietary flags using LLM analysis of all available dish context."""
    conversation = retrieval.conversation_text(message, history)
    return flag_llm.compute_flags_llm(
        cr.current_dishes,
        cr.resolved_variants,
        cr.extra_user_ingredients,
        conversation,
        kb_context,
    )


def _clean_flags(flags: dict) -> dict:
    """Return the 5 presentable flags, stripping internal trigger lists."""
    return {
        "allergens": bool(flags.get("allergens")),
        "gluten_free": bool(flags.get("gluten_free", True)),
        "vegetarian": bool(flags.get("vegetarian", True)),
        "vegan": bool(flags.get("vegan", True)),
        "spicy_level": flags.get("spicy_level", "none"),
    }


def _append_flags_to_context(ctx: str, dish_flags: dict) -> str:
    """Appends general dietary flags summary to the KB context string."""
    if not dish_flags:
        return ctx
    lines = [
        "\n## Banderas dietéticas (calculadas automáticamente)",
        f"- Tiene alérgenos: {'Sí' if dish_flags.get('allergens') else 'No'}",
        f"- Sin gluten: {'Sí' if dish_flags.get('gluten_free') else 'No'}",
        f"- Vegetariano: {'Sí' if dish_flags.get('vegetarian') else 'No'}",
        f"- Vegano: {'Sí' if dish_flags.get('vegan') else 'No'}",
        f"- Nivel picante: {dish_flags.get('spicy_level', 'none')}",
    ]
    return ctx + "\n" + "\n".join(lines)


_CARD_RE = re.compile(r'^\*\*(.+?)\*\*(.*)', re.DOTALL)


def _extract_card_parts(bubble: str) -> tuple[str, str]:
    # Isolate the card paragraph — stop before any \n\n (confirmation question / second bubble)
    card_text = bubble.strip().split('\n\n')[0]
    m = _CARD_RE.match(card_text)
    if not m:
        return ("", "")
    name = m.group(1).strip()
    remainder = m.group(2)
    # Description may be on next line ("**Title** emoji\nDesc") or same line ("**Title** emoji Desc")
    if '\n' in remainder:
        desc = remainder.split('\n', 1)[1].strip()
    else:
        # Skip leading non-letter chars (emoji, spaces) to reach the description
        first_letter = re.search(r'[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]', remainder)
        desc = remainder[first_letter.start():].strip() if first_letter else ""
    return (name, desc)


def _find_last_spanish_card(history: list[dict[str, str]]) -> str:
    for turn in reversed(history):
        if turn.get("role") == "agent":
            text = turn.get("text", "")
            if text.strip().startswith("**"):
                return text
    return ""


def _build_menu_entry(
    result: GenResult,
    history: list[dict[str, str]],
    dish_flags: dict,
) -> dict | None:
    if not result.response:
        return None
    name_en, description_en = _extract_card_parts(result.response[0])
    if not name_en:
        return None
    es_card = _find_last_spanish_card(history)
    name_es, description_es = _extract_card_parts(es_card)
    return {
        "name_es": name_es,
        "name_en": name_en,
        "description_es": description_es,
        "description_en": description_en,
        "flags": _clean_flags(dish_flags),
    }
