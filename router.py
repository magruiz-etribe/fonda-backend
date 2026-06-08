from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Final

import classifier as cls_module
import flag_llm
import generation as gen_module
import retrieval
import timing
from generation import GenResult

logger = logging.getLogger(__name__)

_FALLBACK_RESULT: Final[GenResult] = GenResult(
    response=["Disculpa, tuve un problema procesando tu mensaje. ¿Puedes intentarlo de nuevo? 😊"],
    current_dishes=[],
    buttons=[],
)

_OUT_OF_DOMAIN_RESPONSE: Final[str] = (
    "No puedo ayudarte con ese tema. Soy Huevito y estoy aquí para apoyarte con tu fonda: "
    "adaptar platillos al inglés, registro en plataformas, higiene en cocina y el programa Menú del Día. "
    "¿Te ayudo con algo de eso? 😊"
)


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

_VARIANT_EMOJI: Final[dict[str, str]] = {
    "rojo": "🟥",
    "negro": "⚫",
    "verde": "🌿",
    "blanco": "⬜",
    "poblano": "🫑",
    "amarillo": "🟡",
    "picante": "🌶️",
}


def handle(
    message: str,
    current_dishes: list[str],
    history: list[dict[str, str]],
    confirmation_state: dict | None = None,
    dish_context: dict | None = None,
) -> GenResult:
    try:
        with timing.stage("router.classify"):
            cr = cls_module.classify(message, current_dishes, history, dish_context)
        if cr.intent == "out_of_domain":
            return GenResult(
                response=[_OUT_OF_DOMAIN_RESPONSE],
                current_dishes=list(current_dishes),
                buttons=[],
                intent="out_of_domain",
            )
        with timing.stage("router.merge_variants"):
            cr = _merge_persisted_variants(cr, dish_context)
        with timing.stage("router.enrich_conversation"):
            cr = _enrich_classifier_from_conversation(cr, message, history)
        with timing.stage("router.kb_context"):
            kb_context, kb_links = _get_kb_context(cr, message, history)
        dish_flags: dict = {}
        conf_state_for_gen = None
        trigger_info_for_gen = None
        if cr.intent == "traduccion" and cr.current_dishes:
            with timing.stage("router.flags"):
                dish_flags = _compute_dish_flags(cr, message, history, kb_context)
                dish_flags = _normalize_flags(dish_flags)
            logger.info("computed_flags", extra={"flags": dish_flags, "dishes": cr.current_dishes})
            kb_context = _append_flags_to_context(kb_context, dish_flags)
            conf_state_for_gen = confirmation_state
            trigger_info_for_gen = {
                k: dish_flags.get(k, [])
                for k in ("allergen_triggers", "gluten_triggers", "spicy_triggers")
            }
        short: GenResult | None = None
        with timing.stage("router.short_circuit"):
            if (cr.intent == "traduccion" and cr.current_dishes
                    and not cr.pending_slots and not cr.translate_now
                    and confirmation_state is not None):
                short = _try_short_circuit(cr, confirmation_state, trigger_info_for_gen or {}, message)
        if short is not None:
            short.intent = cr.intent
            short.flags = _clean_flags(dish_flags)
            return short

        with timing.stage("router.generation"):
            result = gen_module.generate(
                cr, message, kb_context, history, conf_state_for_gen,
                trigger_info_for_gen, cr.platform, dish_context,
            )
        result = _ensure_pending_slot_buttons(result, cr)
        result = _guard_etapa_b_integrity(result, confirmation_state)
        if cr.translate_now:
            result = _ensure_english_translation(result, dish_context, history)
        result.intent = cr.intent
        result.flags = _clean_flags(dish_flags)
        result.resolved_variants = cr.resolved_variants
        result.extra_user_ingredients = cr.extra_user_ingredients
        result.links = kb_links
        if cr.translate_now:
            with timing.stage("router.menu_entry"):
                result.current_dishes = []
                result.menu_entry = _build_menu_entry(result, history, dish_flags, dish_context)
        return result
    except Exception as e:
        logger.exception("router_unhandled_exception", extra={"error": str(e)})
        return _FALLBACK_RESULT


def _ensure_english_translation(
    result: GenResult,
    dish_context: dict | None,
    history: list[dict[str, str]],
) -> GenResult:
    """Re-translate the menu card when ETAPA C returns Spanish in the English slot."""
    if not result.response:
        return result

    name_en, description_en = _extract_card_parts(result.response[0])
    if not description_en or not gen_module.looks_spanish(description_en):
        return result

    last_es = (dish_context or {}).get("last_description_es") or ""
    es_card = last_es if last_es else _find_last_spanish_card(history)
    name_es, description_es = _extract_card_parts(es_card)
    if not description_es:
        description_es = description_en

    translated = gen_module.translate_menu_card(
        name_es=name_es,
        description_es=description_es,
        name_en_hint=name_en,
    )
    if not translated:
        logger.warning(
            "english_translation_unresolved",
            extra={"name_en_hint": name_en, "description_es": description_es[:200]},
        )
        return result

    new_card = f"**{translated['name_en']}**\n{translated['description_en']}"
    new_response = [new_card, *result.response[1:]]
    logger.info(
        "english_translation_retry_applied",
        extra={"name_en": translated["name_en"]},
    )
    return replace(result, response=new_response)


def _guard_etapa_b_integrity(
    result: GenResult,
    confirmation_state: dict | None,
) -> GenResult:
    """Detect and repair malformed ETAPA B: '✅ Adaptar al inglés' present but card missing.

    This happens when the LLM jumps ahead and emits the '¿Te parece bien?' bubble
    without the required '**Nombre**\\nDescripcion' card as the first bubble.
    We strip the ETAPA B artifacts so the user sees only the valid bubble (e.g. the
    A1 question) with the correct button restored.
    """
    if "✅ Adaptar al inglés" not in result.buttons:
        return result
    if result.response and result.response[0].strip().startswith("**"):
        return result  # card present — ETAPA B is valid

    logger.warning(
        "etapa_b_card_missing",
        extra={
            "bubbles": len(result.response),
            "first_bubble_prefix": result.response[0][:120] if result.response else "",
        },
    )
    # Keep only bubbles that are NOT the "¿Te parece bien?" confirmation text
    safe_response = [
        b for b in result.response
        if "¿Te parece bien?" not in b and "Adaptar al inglés" not in b
    ]
    cs = confirmation_state or {}
    if cs.get("completeness_confirmed") is None:
        safe_buttons: list[str] = ["✅ Listo, eso es todo!"]
    else:
        safe_buttons = []
    if not safe_response:
        safe_response = ["Disculpa, tuve un problema. ¿Puedes intentarlo de nuevo? 😊"]
        safe_buttons = []
    return replace(result, response=safe_response, buttons=safe_buttons, completeness_confirmed=None)


def _ensure_pending_slot_buttons(
    result: GenResult,
    cr: cls_module.ClassifierResult,
) -> GenResult:
    """Inject variant/slot buttons server-side when the LLM omits them."""
    if not cr.pending_slots:
        return result
    slot = cr.pending_slots[0]
    if not slot.options:
        return result

    buttons = _format_slot_buttons(slot)
    if not buttons:
        return result
    if result.buttons == buttons:
        return result

    if result.buttons:
        logger.warning(
            "slot_buttons_overridden",
            extra={
                "entity": slot.entity,
                "slot_name": slot.slot_name,
                "llm_buttons": result.buttons,
                "injected_buttons": buttons,
            },
        )
    else:
        logger.info(
            "slot_buttons_injected",
            extra={
                "entity": slot.entity,
                "slot_name": slot.slot_name,
                "count": len(buttons),
            },
        )
    return replace(result, buttons=buttons)


def _format_slot_buttons(slot: cls_module.PendingSlot) -> list[str]:
    data = retrieval.get_dish_data(slot.entity) or {}
    variants = data.get("variants") or {}
    buttons: list[str] = []
    for key in slot.options:
        variant = variants.get(key) if isinstance(variants.get(key), dict) else {}
        label = _option_display_label(key, variant)
        emoji = _VARIANT_EMOJI.get(key, "")
        buttons.append(f"{emoji} {label}".strip() if emoji else label)
    return buttons


def _option_display_label(key: str, variant: dict) -> str:
    if name_es := variant.get("name_es"):
        return str(name_es).strip()
    return key.replace("_", " ").title()


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
        "¿Confirmas que tu platillo tiene al menos uno de estos? 🌿 Usa los botones para responder 👇"
    )


def _ask_a3(triggers: list[str]) -> str:
    tl = _fmt_trigger_list(triggers)
    return (
        f"He detectado que tu platillo puede contener algunos de estos ingredientes: **{tl}**. "
        "¿Confirmas que tu platillo tiene al menos uno? 🌾 Usa los botones para responder 👇"
    )


def _ask_a4(triggers: list[str]) -> str:
    tl = _fmt_trigger_list(triggers)
    return (
        f"He detectado que tu platillo puede contener algunos de estos ingredientes: **{tl}**. "
        "¿Confirmas que tu platillo tiene al menos uno? 🌶️ Usa los botones para responder 👇"
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

    # ── A1 — completamente determinista, el LLM no interviene en esta etapa ──
    if cs.get("completeness_confirmed") is None:
        if _is_pure_completion(message):
            # Usuario confirmó que ya no agrega nada → avanzar a la siguiente etapa
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
            # Sin triggers → LLM genera ETAPA B
            return None
        if cr.extra_user_ingredients:
            # RAMA AGREGA: el usuario agregó ingredientes mientras A1 estaba pendiente
            return GenResult(
                response=["¡Anotado! 👍 ¿Algo más que quieras incluir? Si ya está completo, presiona el botón 👇"],
                current_dishes=cr.current_dishes,
                buttons=["✅ Listo, eso es todo!"],
                completeness_confirmed=None,
            )
        # A1 HAZ PREGUNTA: respuesta determinista, sin llamar al LLM
        return GenResult(
            response=[
                "Antes de continuar, ¿tu platillo lleva proteína (pollo, carne, huevo...), "
                "guarnición, salsa especial o algún complemento que no hayamos mencionado? 😊 "
                "Si ya está completo, presiona el botón 👇"
            ],
            current_dishes=cr.current_dishes,
            buttons=["✅ Listo, eso es todo!"],
            completeness_confirmed=None,
        )

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


def _merge_persisted_variants(
    cr: cls_module.ClassifierResult,
    dish_context: dict | None,
) -> cls_module.ClassifierResult:
    """Merge persisted resolved_variants from dish_context, letting LLM values take priority."""
    if not dish_context:
        return cr
    persisted = dish_context.get("resolved_variants") or {}
    if not persisted:
        return cr
    merged = {**persisted, **cr.resolved_variants}
    if merged == cr.resolved_variants:
        return cr
    return replace(cr, resolved_variants=merged)


def _get_kb_context(
    cr: cls_module.ClassifierResult,
    message: str,
    history: list[dict[str, str]],
) -> tuple[str, list[dict]]:
    if cr.intent == "traduccion":
        pending_variant_entities = {
            s.entity for s in cr.pending_slots if s.slot_name == "variant"
        }
        conversation = retrieval.conversation_text(message, history)
        ctx = retrieval.get_context_for_dishes(
            cr.current_dishes,
            resolved_variants=cr.resolved_variants,
            pending_variant_entities=pending_variant_entities,
            conversation=conversation,
        )
        return ctx, []
    return retrieval.get_topic(cr.intent, cr.platform or None)


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


def _normalize_flags(flags: dict) -> dict:
    """Ensure flag/trigger consistency: a flag without triggers is not actionable — clear it."""
    flags = dict(flags)
    if not flags.get("allergen_triggers"):
        flags["allergens"] = False
        flags["allergen_triggers"] = []
    if not flags.get("gluten_triggers"):
        flags["gluten_free"] = True
        flags["gluten_triggers"] = []
    if not flags.get("spicy_triggers"):
        flags["spicy_level"] = "none"
        flags["spicy_triggers"] = []
    return flags


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
    dish_context: dict | None = None,
) -> dict | None:
    if not result.response:
        return None
    name_en, description_en = _extract_card_parts(result.response[0])
    if not name_en:
        return None
    last_es = (dish_context or {}).get("last_description_es") or ""
    es_card = last_es if last_es else _find_last_spanish_card(history)
    name_es, description_es = _extract_card_parts(es_card)
    return {
        "name_es": name_es,
        "name_en": name_en,
        "description_es": description_es,
        "description_en": description_en,
        "flags": _clean_flags(dish_flags),
    }
