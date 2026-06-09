from __future__ import annotations

import logging
import re
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

_APPROVAL_RE: re.Pattern[str] = re.compile(
    r"^(✅|sí\b|si\b|guardar|listo|correcto|exacto)", re.IGNORECASE
)
_EDIT_RE: re.Pattern[str] = re.compile(
    r"^(✏️|cambios|cambiar|editar|ajustar|modificar)", re.IGNORECASE
)
_CARD_RE: re.Pattern[str] = re.compile(r"^\*\*(.+?)\*\*(.*)", re.DOTALL)
_ALLERGEN_NOTE_RE: re.Pattern[str] = re.compile(
    r"^\*\((?:Contiene|Contains)\b", re.IGNORECASE
)


def handle(
    message: str,
    session_state: dict,
    history: list[dict[str, str]],
) -> GenResult:
    try:
        current_dish = session_state.get("current_dish") or ""

        with timing.stage("router.classify"):
            cr = cls_module.classify(message, current_dish, history)

        if cr.intent == "out_of_domain":
            return GenResult(
                response=[_OUT_OF_DOMAIN_RESPONSE],
                current_dishes=session_state.get("current_dishes", []),
                buttons=[],
                intent="out_of_domain",
            )

        if cr.intent != "traduccion":
            return _handle_other_intent(cr, message, history)

        return _handle_traduccion(cr, session_state, message, history)

    except Exception as e:
        logger.exception("router_unhandled_exception", extra={"error": str(e)})
        return _FALLBACK_RESULT


# ── Non-translation intents ───────────────────────────────────────────────────

def _handle_other_intent(
    cr: cls_module.ClassifierResult,
    message: str,
    history: list[dict[str, str]],
) -> GenResult:
    with timing.stage("router.kb_context"):
        kb_context, kb_links = retrieval.get_topic(cr.intent, cr.platform or None)

    with timing.stage("router.generation"):
        result = gen_module.generate(cr, message, kb_context, history)

    result.intent = cr.intent
    result.links = kb_links
    return result


# ── Translation state machine ─────────────────────────────────────────────────

def _handle_traduccion(
    cr: cls_module.ClassifierResult,
    session_state: dict,
    message: str,
    history: list[dict[str, str]],
) -> GenResult:
    dish_status = session_state.get("dish_status")
    session_current_dish = session_state.get("current_dish") or ""

    # Detect new dish from classifier
    new_dish_from_cr = bool(cr.current_dish) and cr.current_dish != session_current_dish

    if new_dish_from_cr:
        effective_session: dict = {
            "current_dish": cr.current_dish,
            "companions": cr.companions,
            "dish_status": None,
            "collected_ingredients": [],
            "detected_flags": [],
        }
    else:
        # Once inside a dish flow, freeze companions from session — the extractor
        # can mis-classify ingredients/toppings as companions (e.g. "queso y crema").
        # Only the initial dish detection sets companions; the EXTRACTING LLM handles
        # subsequent ingredient mentions.
        preserve_companions = dish_status is not None
        effective_session = {
            "current_dish": session_current_dish or cr.current_dish,
            "companions": (
                session_state.get("companions", [])
                if preserve_companions
                else (cr.companions or session_state.get("companions", []))
            ),
            "dish_status": dish_status,
            "collected_ingredients": list(session_state.get("collected_ingredients") or []),
            "detected_flags": list(session_state.get("detected_flags") or []),
        }

    effective_dish = effective_session["current_dish"]

    if not effective_dish:
        return GenResult(
            response=["¡Con gusto te ayudo! Cuéntame cómo preparas tu platillo: nombre, proteína, tipo de salsa, relleno, guarniciones... Entre más detalles me des, más completa queda la descripción. 😊"],
            current_dishes=[],
            buttons=[],
            dish_status=None,
            intent="traduccion",
        )

    current_status = effective_session["dish_status"]

    # Custom dish (not in KB): handle before entering the normal state machine
    if effective_dish == "custom" and current_status is None:
        if not cr.custom_dish_known:
            return GenResult(
                response=["Por el momento no tengo información sobre ese platillo para adaptarlo al menú. ¿Tienes otro platillo que quieras poner? 😊"],
                current_dishes=[],
                buttons=[],
                dish_status=None,
                intent="traduccion",
            )
        return GenResult(
            response=["¡Con gusto! No tengo ese platillo en mi base de conocimiento, pero si me describes cómo lo preparas — ingredientes principales, tipo de salsa, proteína, guarnición — yo te ayudo con la descripción para el menú. 😊"],
            current_dishes=["custom"],
            buttons=[],
            dish_status="EXTRACTING",
            collected_ingredients=[],
            detected_flags=[],
            intent="traduccion",
        )

    if current_status is None or current_status == "EXTRACTING":
        return _handle_extracting(effective_session, message, history)
    if current_status == "CONFIRMING_FLAGS":
        return _handle_confirming_flags(effective_session, message, history)
    if current_status == "DRAFTING":
        return _handle_drafting(effective_session, message, history)

    return _handle_extracting(effective_session, message, history)


def _handle_extracting(
    session_state: dict,
    message: str,
    history: list[dict[str, str]],
) -> GenResult:
    current_dish: str = session_state["current_dish"]
    companions: list[str] = session_state["companions"]
    collected: list[str] = list(session_state.get("collected_ingredients") or [])

    kb_data = retrieval.get_dish_data(current_dish) or {}
    variables_requeridas: list[str] = kb_data.get("variables_requeridas") or []

    # Short-circuit only for KB dishes with no variables — not for custom dishes,
    # which need the EXTRACTING LLM to pull collected_ingredients from the description.
    if not variables_requeridas and current_dish != "custom":
        return _transition_to_flags_or_draft(
            current_dish, companions, collected, message, history, kb_data
        )

    with timing.stage("router.generation"):
        result = gen_module.generate_extracting(
            current_dish=current_dish,
            companions=companions,
            collected_ingredients=collected,
            message=message,
            history=history,
            kb_data=kb_data,
        )

    new_collected = result.collected_ingredients if result.collected_ingredients else collected

    if result.variables_complete:
        return _transition_to_flags_or_draft(
            current_dish, companions, new_collected, message, history, kb_data
        )

    result.dish_status = "EXTRACTING"
    result.collected_ingredients = new_collected
    result.current_dishes = [current_dish] + companions
    result.intent = "traduccion"
    return result


def _transition_to_flags_or_draft(
    current_dish: str,
    companions: list[str],
    collected: list[str],
    message: str,
    history: list[dict[str, str]],
    kb_data: dict,
) -> GenResult:
    with timing.stage("router.flags"):
        raw_flags = _compute_flags(current_dish, companions, collected)

    all_detected = _extract_detected_flag_names(raw_flags)
    clean_flags = _clean_flags(raw_flags)

    # Only ask CONFIRMING_FLAGS for triggers that the user has NOT already stated.
    # If the user explicitly said "enfrijoladas con queso y crema", queso/crema are
    # already in collected_ingredients — asking again is redundant and confusing.
    user_stated = {i.replace("_", " ").lower() for i in collected}
    hidden_triggers = [
        t for t in all_detected
        if t.lower() not in user_stated and t.replace(" ", "_").lower() not in user_stated
    ]

    if hidden_triggers:
        with timing.stage("router.generation"):
            result = gen_module.generate_confirming_flags(
                current_dish=current_dish,
                companions=companions,
                collected_ingredients=collected,
                detected_flags=hidden_triggers,
                message=message,
                history=history,
            )
        result.dish_status = "CONFIRMING_FLAGS"
        result.collected_ingredients = collected
        result.detected_flags = all_detected  # store all triggers for the menu entry
        result.current_dishes = [current_dish] + companions
        result.flags = clean_flags
        result.intent = "traduccion"
        return result

    return _start_drafting(
        current_dish, companions, collected, all_detected, clean_flags, message, history, kb_data
    )


def _handle_confirming_flags(
    session_state: dict,
    message: str,
    history: list[dict[str, str]],
) -> GenResult:
    current_dish: str = session_state["current_dish"]
    companions: list[str] = session_state["companions"]
    collected: list[str] = list(session_state.get("collected_ingredients") or [])
    detected_flags: list[str] = list(session_state.get("detected_flags") or [])

    with timing.stage("router.flags"):
        raw_flags = _compute_flags(current_dish, companions, collected)
    clean_flags = _clean_flags(raw_flags)

    # Let the user back out and edit instead of confirming allergens
    if _EDIT_RE.match(message.strip()):
        return GenResult(
            response=["Claro, vamos a ajustarlo. ¿Qué cambiamos? 😊"],
            current_dishes=[current_dish] + companions,
            buttons=[],
            flags=clean_flags,
            dish_status="EXTRACTING",
            collected_ingredients=[],
            detected_flags=[],
            intent="traduccion",
        )

    kb_data = retrieval.get_dish_data(current_dish) or {}
    return _start_drafting(
        current_dish, companions, collected, detected_flags, clean_flags, message, history, kb_data
    )


def _start_drafting(
    current_dish: str,
    companions: list[str],
    collected: list[str],
    detected_flags: list[str],
    clean_flags: dict,
    message: str,
    history: list[dict[str, str]],
    kb_data: dict,
) -> GenResult:
    with timing.stage("router.generation"):
        result = gen_module.generate_drafting(
            current_dish=current_dish,
            companions=companions,
            collected_ingredients=collected,
            detected_flags=detected_flags,
            message=message,
            history=history,
            kb_data=kb_data,
        )
    result.dish_status = "DRAFTING"
    result.collected_ingredients = collected
    result.detected_flags = detected_flags
    result.current_dishes = [current_dish] + companions
    result.flags = clean_flags
    result.intent = "traduccion"
    return result


def _handle_drafting(
    session_state: dict,
    message: str,
    history: list[dict[str, str]],
) -> GenResult:
    current_dish: str = session_state["current_dish"]
    companions: list[str] = session_state["companions"]
    collected: list[str] = list(session_state.get("collected_ingredients") or [])
    detected_flags: list[str] = list(session_state.get("detected_flags") or [])

    with timing.stage("router.flags"):
        raw_flags = _compute_flags(current_dish, companions, collected)
    clean_flags = _clean_flags(raw_flags)

    msg_stripped = message.strip()

    if _APPROVAL_RE.match(msg_stripped):
        menu_entry = _build_menu_entry_from_history(history, clean_flags)
        return GenResult(
            response=["¡Perfecto! Lo guardé en tu menú. ¿Quieres traducir otro platillo? 😊"],
            current_dishes=[],
            buttons=[],
            flags=clean_flags,
            menu_entry=menu_entry,
            save_to_menu=True,
            dish_status=None,
            collected_ingredients=[],
            detected_flags=[],
            intent="traduccion",
        )

    if _EDIT_RE.match(msg_stripped):
        return GenResult(
            response=["Claro, vamos a ajustarlo. ¿Qué cambiamos? 😊"],
            current_dishes=[current_dish] + companions,
            buttons=[],
            flags=clean_flags,
            dish_status="EXTRACTING",
            collected_ingredients=[],
            detected_flags=[],
            intent="traduccion",
        )

    # Any other message → regenerate drafting card
    kb_data = retrieval.get_dish_data(current_dish) or {}
    return _start_drafting(
        current_dish, companions, collected, detected_flags, clean_flags, message, history, kb_data
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_flags(
    current_dish: str,
    companions: list[str],
    collected: list[str],
) -> dict:
    kb_map: dict[str, list[str]] = {}
    for entity in [current_dish] + companions:
        data = retrieval.get_dish_data(entity) or {}
        defaults = [
            str(i).strip()
            for i in (data.get("ingredientes_base_default") or [])
            if str(i).strip()
        ]
        if defaults:
            kb_map[entity] = defaults
    raw = flag_llm.compute_flags_for_dish(
        current_dish=current_dish,
        companions=companions,
        collected_ingredients=collected,
        kb_ingredients_per_dish=kb_map,
    )
    return _normalize_flags(raw)


def _build_all_ingredients(
    current_dish: str,
    companions: list[str],
    collected: list[str],
) -> list[str]:
    ingredients: list[str] = list(collected)
    seen: set[str] = {i.lower() for i in ingredients}

    for entity in [current_dish] + companions:
        data = retrieval.get_dish_data(entity) or {}
        for ing in data.get("ingredientes_base_default") or []:
            ing_lower = str(ing).strip().lower()
            if ing_lower and ing_lower not in seen:
                seen.add(ing_lower)
                ingredients.append(ing_lower)

    return ingredients


def _extract_detected_flag_names(flags: dict) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for key in ("allergen_triggers", "gluten_triggers", "spicy_triggers"):
        for trigger in (flags.get(key) or []):
            name = str(trigger).replace("_", " ").strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _normalize_flags(flags: dict) -> dict:
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
    return {
        "allergens": bool(flags.get("allergens")),
        "gluten_free": bool(flags.get("gluten_free", True)),
        "vegetarian": bool(flags.get("vegetarian", True)),
        "vegan": bool(flags.get("vegan", True)),
        "spicy_level": flags.get("spicy_level", "none"),
    }


def _extract_card_parts(bubble: str) -> tuple[str, str]:
    text = bubble.strip()
    m = _CARD_RE.match(text)
    if not m:
        return "", ""
    name = m.group(1).strip()
    rest = m.group(2)  # everything after **Title** (DOTALL — may span multiple lines)
    desc_lines: list[str] = []
    for line in rest.split("\n"):
        stripped = line.strip()
        if not stripped:
            if not desc_lines:
                continue  # skip leading blank lines between title and description
            break  # first blank line after description = end of this section
        if _ALLERGEN_NOTE_RE.match(stripped):
            continue  # allergen notes live in flags, not in stored description
        if stripped.startswith("---"):
            break  # confirmation separator
        desc_lines.append(stripped)
    return name, "\n".join(desc_lines).strip()


def _build_menu_entry_from_history(
    history: list[dict[str, str]],
    clean_flags: dict,
) -> dict | None:
    for turn in reversed(history):
        if turn.get("role") == "agent":
            text = turn.get("text", "")
            if text.strip().startswith("**"):
                return _parse_bilingual_card(text, clean_flags)
    return None


def _parse_bilingual_card(card: str, flags: dict) -> dict:
    card = card.strip()
    en_start = re.search(r"\n\n\*\*", card)
    if en_start:
        es_text = card[:en_start.start()].strip()
        en_text = card[en_start.start():].strip()
    else:
        es_text = card
        en_text = ""

    name_es, desc_es = _extract_card_parts(es_text)
    name_en, desc_en = _extract_card_parts(en_text) if en_text else ("", "")

    return {
        "name_es": name_es,
        "name_en": name_en,
        "description_es": desc_es,
        "description_en": desc_en,
        "flags": flags,
    }
