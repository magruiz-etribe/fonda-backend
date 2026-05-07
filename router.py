from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Final

import bedrock_client
import classifier
import config
import entity_mapper
import generation
import image_analyzer
import retrieval
import validation
from classifier import TurnDecision
from generation import GenInput
from state import (
    CUSTOM_ENTITY,
    AgentState,
    CompletedDish,
    CurrentDish,
    pause_current,
    resume_paused,
)

logger = logging.getLogger(__name__)


_GENERIC_FALLBACK: Final[str] = config.GENERIC_FALLBACK_REPLY


# Backup determinístico para cuando el classifier no etiqueta como
# approve/reject pero el mensaje del fondero es claramente afirmativo o
# negativo en respuesta a "¿queda conforme?". Conservador a propósito:
# solo aplica cuando hay last_completed pendiente y el mensaje cae 1:1 en
# un patrón corto, para no pisar al classifier en casos ambiguos.
_AFFIRMATIVE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(p) for p in (
        r"^si$",
        r"^sii+$",
        r"^si si$",
        r"^si+ esta bien$",
        r"^si+ estoy conforme$",
        r"^si+,? (esta|esto|asi) bien$",
        r"^si+,? me gusta$",
        r"^si+,? correcto$",
        r"^si+,? perfecto$",
        r"^estoy conforme$",
        r"^esta bien$",
        r"^asi (esta|es)( bien)?$",
        r"^ok$",
        r"^okay$",
        r"^okey$",
        r"^sale$",
        r"^va$",
        r"^andale$",
        r"^claro$",
        r"^perfecto$",
        r"^correcto$",
        r"^me gusta$",
        r"^conforme$",
    )
)

_NEGATIVE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(p) for p in (
        r"^no$",
        r"^no,? cambiale$",
        r"^cambiale$",
        r"^no esta bien$",
        r"^no me gusta$",
        r"^no me convence$",
        r"^ese no es$",
    )
)


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    s = re.sub(r"[¿?¡!.,;:]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _deterministic_signal(message: str) -> str | None:
    norm = _normalize(message)
    if not norm or len(norm) > 32:
        return None
    if any(p.match(norm) for p in _AFFIRMATIVE_PATTERNS):
        return "approve"
    if any(p.match(norm) for p in _NEGATIVE_PATTERNS):
        return "reject"
    return None


def handle(
    message: str,
    state: AgentState,
    image_b64: str | None,
    history: list[dict[str, str]],
) -> tuple[str, AgentState]:
    if image_b64:
        _process_image(image_b64, state)

    image_summary = state.image_analysis.raw if state.image_analysis else None
    decision = classifier.classify(state, message, history, image_summary)

    if decision.intent_changed and state.intent and state.intent != decision.intent:
        logger.info(
            "intent_change",
            extra={"from": state.intent, "to": decision.intent},
        )
        pause_current(state)

    state.intent = decision.intent

    if decision.intent == "traducir":
        _backfill_user_signal(decision, state, message)
        _apply_traducir_updates(decision, state)

    reply = _dispatch(decision.intent, message, state, history)

    if _current_task_finished(decision.intent, state) and state.paused_task is not None:
        resume_msg = resume_paused(state)
        if resume_msg:
            reply = f"{reply}\n\n{resume_msg}"

    return reply, state


def _process_image(image_b64: str, state: AgentState) -> None:
    try:
        analysis = image_analyzer.analyze(image_b64)
    except (image_analyzer.ImageAnalysisError, bedrock_client.BedrockError) as e:
        logger.warning("image_analyzer_failed", extra={"error": str(e)})
        return

    state.image_analysis = analysis

    if analysis.type == "menu" and analysis.detected_dishes:
        state.mode = "menu"
        state.dish_queue = list(analysis.detected_dishes)
        state.current_dish = None
        return

    if analysis.type == "platillo":
        state.mode = "platillo"
        if analysis.detected_dishes:
            entities = entity_mapper.map_dishes(
                " ".join(analysis.detected_dishes),
                image_dishes=analysis.detected_dishes,
            )
            primary = entities[0]
            extras = [e for e in entities[1:] if e != CUSTOM_ENTITY]
            state.current_dish = CurrentDish(
                entity=primary, variant=None, user_ingredients=[]
            )
            if extras:
                state.dish_queue = extras + state.dish_queue


def _backfill_user_signal(
    d: TurnDecision,
    state: AgentState,
    message: str,
) -> None:
    """Si hay traducción pendiente y el classifier no etiquetó approve/reject
    pero el mensaje del fondero es 1:1 una afirmación/negación corta,
    rellenamos el user_signal de forma determinística. Defensa para los
    casos en que Nova Micro se queda con user_signal=null en mensajes
    obvios como 'si' / 'no'."""
    if d.user_signal is not None:
        return
    if not state.completed or state.completed[-1].approved is not None:
        return
    fallback = _deterministic_signal(message)
    if fallback is None:
        return
    logger.info(
        "user_signal_backfill",
        extra={"signal": fallback, "msg": message[:64]},
    )
    d.user_signal = fallback  # type: ignore[assignment]


def _apply_traducir_updates(d: TurnDecision, state: AgentState) -> None:
    """Aplica determinísticamente al state lo que el classifier dictaminó para
    el intent traducir: cambio de platillo, promoción de __custom__, variante,
    ingredientes nuevos, aprobación/rechazo de la última traducción."""
    cd = state.current_dish

    if d.dish_change and d.new_entity:
        if cd is not None and d.new_entity != cd.entity:
            logger.info(
                "dish_change",
                extra={"from": cd.entity, "to": d.new_entity},
            )
        state.current_dish = CurrentDish(
            entity=d.new_entity,
            variant=d.variant,
            user_ingredients=[],
        )
        cd = state.current_dish
    elif cd is None and d.new_entity:
        state.current_dish = CurrentDish(
            entity=d.new_entity,
            variant=d.variant,
            user_ingredients=[],
        )
        cd = state.current_dish
    elif (
        cd is not None
        and cd.entity == CUSTOM_ENTITY
        and d.new_entity
        and d.new_entity != CUSTOM_ENTITY
    ):
        logger.info("current_dish_promoted", extra={"from": cd.entity, "to": d.new_entity})
        cd.entity = d.new_entity
        if d.variant and not cd.variant:
            cd.variant = d.variant

    if cd is not None:
        if d.variant and not cd.variant:
            cd.variant = d.variant
        for ing in d.ingredients_added:
            ing_clean = ing.strip().lower()
            if ing_clean and ing_clean not in cd.user_ingredients:
                cd.user_ingredients.append(ing_clean)

    if state.completed and state.completed[-1].approved is None:
        last = state.completed[-1]
        if d.user_signal == "approve":
            last.approved = True
            _advance_after_approval(state)
        elif d.user_signal == "reject":
            last.approved = False

    if state.current_dish is not None and state.mode is None:
        state.mode = "menu" if state.dish_queue else "platillo"


def _dispatch(
    intent: str,
    message: str,
    state: AgentState,
    history: list[dict[str, str]],
) -> str:
    if intent == "traducir":
        return _handle_traducir(message, state, history)
    if intent == "higiene":
        return _handle_static(message, state, history, "higiene")
    if intent == "maps":
        return _handle_static(message, state, history, "maps")
    return _handle_fallback(message, state, history)


def _handle_traducir(
    message: str,
    state: AgentState,
    history: list[dict[str, str]],
) -> str:
    cd = state.current_dish
    kb = retrieval.get_dish_context(cd.entity) if cd else ""

    visible, meta = _generate_with_validation(
        intent="traducir",
        message=message,
        state=state,
        history=history,
        kb_context=kb,
        user_ingredients=cd.user_ingredients if cd else [],
    )

    if meta and meta.get("final_translation") is True and cd is not None:
        _record_translation(meta, cd, state)

    return visible


def _record_translation(
    meta: dict[str, Any],
    cd: CurrentDish,
    state: AgentState,
) -> None:
    """Persiste la traducción producida por Nova Lite (vía bloque <META>) en
    state.completed con dedup por entidad pendiente."""
    translation_en = str(meta.get("translation_en", "")).strip()
    description_en = str(meta.get("description_en", "")).strip()
    raw_allergens = meta.get("allergens") or []
    allergens: list[str] = []
    if isinstance(raw_allergens, list):
        allergens = [
            str(a).strip()
            for a in raw_allergens
            if isinstance(a, (str, int, float)) and str(a).strip()
        ]

    label = cd.entity if cd.entity != CUSTOM_ENTITY else "platillo personalizado"

    if state.completed and state.completed[-1].approved is None and state.completed[-1].dish == label:
        last = state.completed[-1]
        last.translation_en = translation_en or last.translation_en
        last.description_en = description_en or last.description_en
        last.allergens = allergens or last.allergens
        last.ingredients = list(cd.user_ingredients) or last.ingredients
        return

    state.completed.append(CompletedDish(
        dish=label,
        ingredients=list(cd.user_ingredients),
        translation_en=translation_en,
        description_en=description_en,
        allergens=allergens,
        approved=None,
    ))


def _handle_static(
    message: str,
    state: AgentState,
    history: list[dict[str, str]],
    topic: str,
) -> str:
    kb = retrieval.get_static(topic)  # type: ignore[arg-type]
    visible, _ = _generate_with_validation(
        intent=topic,
        message=message,
        state=state,
        history=history,
        kb_context=kb,
        user_ingredients=[],
    )
    return visible


def _handle_fallback(
    message: str,
    state: AgentState,
    history: list[dict[str, str]],
) -> str:
    visible, _ = _generate_with_validation(
        intent="fallback",
        message=message,
        state=state,
        history=history,
        kb_context="",
        user_ingredients=[],
    )
    return visible


def _generate_with_validation(
    intent: str,
    message: str,
    state: AgentState,
    history: list[dict[str, str]],
    kb_context: str,
    user_ingredients: list[str],
) -> tuple[str, dict[str, Any] | None]:
    gi = GenInput(
        state=state,
        intent=intent,
        message=message,
        kb_context=kb_context,
        history=history,
    )
    try:
        raw = generation.generate(gi)
    except bedrock_client.BedrockError as e:
        logger.warning("generation_failed_first", extra={"error": str(e)})
        return _GENERIC_FALLBACK, None

    visible, meta = generation.parse_and_strip_meta(raw)

    ok, reason = validation.validate(visible, kb_context, user_ingredients)
    if ok:
        return visible, meta

    logger.info("validation_failed_first", extra={"reason": reason})
    gi2 = GenInput(
        state=state,
        intent=intent,
        message=message,
        kb_context=kb_context,
        history=history,
        correction_note=(
            f"La respuesta anterior falló validación: {reason}. "
            f"Responde en español, sin caracteres extraños, y declara solo "
            f"alérgenos respaldados por el contexto del KB o los ingredientes del fondero."
        ),
    )
    try:
        raw2 = generation.generate(gi2)
    except bedrock_client.BedrockError as e:
        logger.warning("generation_failed_second", extra={"error": str(e)})
        return _GENERIC_FALLBACK, None

    visible2, meta2 = generation.parse_and_strip_meta(raw2)
    ok2, reason2 = validation.validate(visible2, kb_context, user_ingredients)
    if ok2:
        return visible2, meta2

    logger.info("validation_failed_second", extra={"reason": reason2})
    return _GENERIC_FALLBACK, None


def _advance_after_approval(state: AgentState) -> None:
    state.current_dish = None
    if state.mode == "menu" and state.dish_queue:
        next_entity = state.dish_queue.pop(0)
        state.current_dish = CurrentDish(
            entity=next_entity, variant=None, user_ingredients=[]
        )
    else:
        state.mode = None


def _current_task_finished(intent: str, state: AgentState) -> bool:
    if intent != "traducir":
        return False
    if state.completed and state.completed[-1].approved is True:
        return state.current_dish is None and not state.dish_queue
    return False
