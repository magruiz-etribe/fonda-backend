from __future__ import annotations

import dataclasses
import logging
from typing import Any, Final

import bedrock_client
import classifier
import config
import entity_mapper
import generation
import image_analyzer
import retrieval
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


def _coerce_traducir_approve_short_ack(
    decision: TurnDecision,
    message: str,
    had_conforme_pending: bool,
) -> TurnDecision:
    """Si el micro no marca approve pero hay traducción pendiente y el texto es sólo conforme corto."""
    if decision.intent != "traducir":
        return decision
    if decision.user_signal in ("reject", "farewell", "want_translation"):
        return decision
    if decision.user_signal == "approve":
        return decision
    if decision.dish_change and decision.new_entity:
        return decision
    if not had_conforme_pending:
        return decision
    if not generation.message_looks_like_pure_translation_ack(message):
        return decision
    prior = decision.user_signal
    logger.info("coerced_traducir_user_signal_approve", extra={"prior_signal": prior})
    return dataclasses.replace(decision, user_signal="approve")


def _restore_traducir_if_paused(state: AgentState, decision: TurnDecision) -> None:
    """Si el clasificador dejó intent traducir sin platillo pero teníamos una traducción
    pausada con current_dish, recupera ese trabajo (p. ej. tras un falso fallback)."""
    if state.current_dish is not None:
        return
    if decision.dish_change or decision.new_entity:
        return
    pt = state.paused_task
    if pt is None or pt.intent != "traducir" or pt.current_dish is None:
        return
    state.current_dish = pt.current_dish
    if pt.mode is not None:
        state.mode = pt.mode
    state.dish_queue = list(pt.dish_queue)
    state.paused_task = None
    logger.info(
        "traducir_restored_from_pause",
        extra={"entity": state.current_dish.entity if state.current_dish else None},
    )


def handle(
    message: str,
    state: AgentState,
    image_b64: str | None,
    history: list[dict[str, str]],
) -> tuple[str, AgentState]:
    if image_b64:
        _process_image(image_b64, state)

    image_summary = state.image_analysis.raw if state.image_analysis else None
    had_conforme_pending = bool(
        state.completed
        and state.completed[-1].approved is None
        and (state.completed[-1].translation_en or "").strip()
    )
    decision = classifier.classify(state, message, history, image_summary)
    if decision.intent == "traducir":
        decision = _coerce_traducir_approve_short_ack(
            decision, message, had_conforme_pending
        )

    logger.info(
        "classifier_decision",
        extra={
            "intent": decision.intent,
            "intent_changed": decision.intent_changed,
            "dish_change": decision.dish_change,
            "new_entity": decision.new_entity,
            "variant": decision.variant,
            "ingredients_added": decision.ingredients_added,
            "user_signal": decision.user_signal,
            "reasoning": (decision.reasoning or "")[:2000],
        },
    )

    if decision.intent_changed and state.intent and state.intent != decision.intent:
        logger.info(
            "intent_change",
            extra={"from": state.intent, "to": decision.intent},
        )
        pause_current(state)

    state.intent = decision.intent

    if decision.intent == "traducir":
        _restore_traducir_if_paused(state, decision)
        _apply_traducir_updates(decision, state)

    logger.info(
        "dispatch_start",
        extra={
            "intent": decision.intent,
            "has_current_dish": state.current_dish is not None,
            "current_entity": state.current_dish.entity if state.current_dish else None,
            "current_variant": state.current_dish.variant if state.current_dish else None,
            "ing_count": len(state.current_dish.user_ingredients) if state.current_dish else 0,
            "completed_count": len(state.completed),
            "pending_approval": (
                bool(state.completed) and state.completed[-1].approved is None
            ),
        },
    )

    reply = _dispatch(decision, message, state, history, had_conforme_pending)

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


def _apply_traducir_updates(d: TurnDecision, state: AgentState) -> None:
    """Aplica determinísticamente al state lo que el classifier dictaminó para
    el intent traducir: cambio de platillo, promoción de __custom__, variante,
    ingredientes nuevos, aprobación/rechazo de la última traducción."""
    conforme_pending = bool(
        state.completed
        and state.completed[-1].approved is None
        and (state.completed[-1].translation_en or "").strip()
    )
    implicit_approve = (
        conforme_pending
        and bool(d.dish_change and d.new_entity)
        and d.user_signal != "reject"
        and d.user_signal != "farewell"
    )

    # Cerrar traducción pendiente ANTES de asignar el platillo nuevo: si viene
    # approve explícito o cambio claro de platillo, `_advance_after_approval` limpia
    # trabajo anterior y luego el bloque dish_change define el siguiente `current_dish`.
    if conforme_pending:
        last = state.completed[-1]
        if d.user_signal == "reject":
            last.approved = False
        elif d.user_signal == "farewell":
            last.approved = True
            logger.info(
                "farewell_implied_close_pending_translation",
                extra={"dish": last.dish},
            )
            _advance_after_approval(state)
        elif d.user_signal == "approve" or implicit_approve:
            last.approved = True
            if implicit_approve and d.user_signal != "approve":
                logger.info(
                    "implicit_approve_on_dish_change",
                    extra={"from_dish": last.dish, "to_entity": d.new_entity},
                )
            _advance_after_approval(state)

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

    if state.current_dish is not None and state.mode is None:
        state.mode = "menu" if state.dish_queue else "platillo"


def _dispatch(
    decision: TurnDecision,
    message: str,
    state: AgentState,
    history: list[dict[str, str]],
    had_conforme_pending: bool,
) -> str:
    intent = decision.intent
    if intent == "traducir":
        return _handle_traducir(
            message, state, history, decision, had_conforme_pending
        )
    if intent == "higiene":
        return _handle_static(message, state, history, "higiene")
    if intent == "maps":
        return _handle_static(message, state, history, "maps")
    return _handle_fallback(message, state, history)


def _handle_traducir(
    message: str,
    state: AgentState,
    history: list[dict[str, str]],
    decision: TurnDecision,
    had_conforme_pending: bool,
) -> str:
    if decision.user_signal == "farewell":
        logger.info("traducir_light_reply", extra={"mode": "farewell"})
        return generation.generate_traducir_light(message, history, mode="farewell")

    no_new_dish = not (decision.dish_change and decision.new_entity)
    if (
        decision.user_signal == "approve"
        and had_conforme_pending
        and no_new_dish
    ):
        logger.info("traducir_light_reply", extra={"mode": "post_approve"})
        return generation.generate_traducir_light(
            message, history, mode="post_approve"
        )

    cd = state.current_dish
    kb = retrieval.get_dish_context(cd.entity) if cd else ""

    visible, meta = _generate(
        intent="traducir",
        message=message,
        state=state,
        history=history,
        kb_context=kb,
    )

    visible = generation.sanitize_traducir_visible_for_user(visible)
    if meta is None or meta.get("final_translation") is not True:
        inferred = generation.infer_final_translation_meta(visible)
        if inferred:
            meta = inferred
            logger.info("translation_meta_inferred_from_visible_lines")

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
    visible, _ = _generate(
        intent=topic,
        message=message,
        state=state,
        history=history,
        kb_context=kb,
    )
    return visible


def _handle_fallback(
    message: str,
    state: AgentState,
    history: list[dict[str, str]],
) -> str:
    visible, _ = _generate(
        intent="fallback",
        message=message,
        state=state,
        history=history,
        kb_context="",
    )
    return visible


def _return_fallback(
    reason: str,
    intent: str,
    **extra: Any,
) -> tuple[str, dict[str, Any] | None]:
    """Único punto de retorno del GENERIC_FALLBACK en el pipeline de generación.
    Filtra por `fallback_returned` en CloudWatch para ver TODOS los fallbacks
    con su razón y etapa."""
    logger.warning(
        "fallback_returned",
        extra={"fallback_reason": reason, "intent": intent, **extra},
    )
    return _GENERIC_FALLBACK, None


def _generate(
    intent: str,
    message: str,
    state: AgentState,
    history: list[dict[str, str]],
    kb_context: str,
) -> tuple[str, dict[str, Any] | None]:
    """Pipeline de generación. Sin validador post-hoc: el prompt de
    `generation` system text (prompts/generation_system.txt) es la única fuente de verdad sobre formato,
    alérgenos, idioma, etc. Si Bedrock truena se cae a fallback genérico;
    cualquier otro defecto del modelo se trata aguas arriba afinando el
    prompt, no aquí."""
    gi = GenInput(
        state=state,
        intent=intent,
        message=message,
        kb_context=kb_context,
        history=history,
    )

    logger.info(
        "generation_attempt",
        extra={
            "intent": intent,
            "kb_len": len(kb_context),
            "history_len": len(history),
        },
    )
    try:
        raw = generation.generate(gi)
    except bedrock_client.BedrockError as e:
        logger.warning(
            "generation_failed",
            extra={"error_type": type(e).__name__, "error": str(e)},
        )
        return _return_fallback(
            "generation_bedrock_error",
            intent,
            error_type=type(e).__name__,
            error=str(e),
        )

    visible, meta = generation.parse_and_strip_meta(raw)
    visible = generation.strip_leaked_coaching_text(visible)
    logger.info(
        "generation_done",
        extra={
            "raw_len": len(raw),
            "visible_len": len(visible),
            "has_meta": meta is not None,
            "meta_final_translation": bool(meta and meta.get("final_translation")),
        },
    )
    return visible, meta


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
