from __future__ import annotations

import logging
import re
from typing import Any, Final

import classifier as cls_module
import flags as flags_module
import generation as gen_module
import retrieval
from generation import GenResult

logger = logging.getLogger(__name__)

_FALLBACK_RESULT: Final[GenResult] = GenResult(
    response=["Disculpa, tuve un problema procesando tu mensaje. ¿Puedes intentarlo de nuevo? 😊"],
    current_dishes=[],
    buttons=[],
)

_MAPS_LINK: Final[dict[str, Any]] = {
    "label": "Regístrate en Google Maps",
    "url": "https://business.google.com/es-all/business-profile/?ppsrc=GPDA2",
}


def handle(
    message: str,
    current_dishes: list[str],
    history: list[dict[str, str]],
    confirmation_state: dict | None = None,
) -> GenResult:
    try:
        cr = cls_module.classify(message, current_dishes, history)
        kb_context = _get_kb_context(cr)
        dish_flags: dict = {}
        conf_state_for_gen = None
        trigger_info_for_gen = None
        if cr.intent == "traduccion" and cr.current_dishes:
            dish_flags = _compute_dish_flags(cr)
            logger.info("computed_flags", extra={"flags": dish_flags, "dishes": cr.current_dishes})
            kb_context = _append_flags_to_context(kb_context, dish_flags)
            conf_state_for_gen = confirmation_state
            trigger_info_for_gen = {
                k: dish_flags.get(k, [])
                for k in ("allergen_triggers", "gluten_triggers", "spicy_triggers")
            }
        result = gen_module.generate(cr, message, kb_context, history, conf_state_for_gen, trigger_info_for_gen)
        result.flags = _clean_flags(dish_flags)
        if cr.intent == "maps":
            result.link = _MAPS_LINK
        if cr.translate_now and not result.current_dishes:
            result.menu_entry = _build_menu_entry(result, history, dish_flags)
        return result
    except Exception as e:
        logger.exception("router_unhandled_exception", extra={"error": str(e)})
        return _FALLBACK_RESULT


def _get_kb_context(cr: cls_module.ClassifierResult) -> str:
    if cr.intent == "traduccion":
        return retrieval.get_context_for_dishes(cr.current_dishes)
    if cr.intent in ("maps", "higiene"):
        return retrieval.get_static(cr.intent)  # type: ignore[arg-type]
    return ""


def _compute_dish_flags(cr: cls_module.ClassifierResult) -> dict:
    """Compute dietary flags from KB ingredients + resolved variant + fondero extras."""
    all_ingr: list[str] = []
    for dish in cr.current_dishes:
        data = retrieval.get_dish_data(dish)
        if not data:
            continue
        all_ingr += data.get("base_ingredients", [])
        variant_key = cr.resolved_variants.get(dish)
        if variant_key:
            variant = data.get("variants", {}).get(variant_key, {})
            all_ingr += variant.get("extra_ingredients", [])
    return flags_module.compute_flags(all_ingr, list(cr.extra_user_ingredients))


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


_CARD_RE = re.compile(r'^\*\*(.+?)\*\*[^\n]*\n?(.*)', re.DOTALL)


def _extract_card_parts(bubble: str) -> tuple[str, str]:
    m = _CARD_RE.match(bubble.strip())
    if not m:
        return ("", "")
    return (m.group(1).strip(), m.group(2).strip())


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
