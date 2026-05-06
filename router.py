from __future__ import annotations

import logging
import re
from typing import Final

import bedrock_client
import classifier
import config
import entity_mapper
import generation
import image_analyzer
import retrieval
import validation
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

_APPROVAL_RE = re.compile(
    r"\b(s[íi]|claro|ok|okay|de acuerdo|conforme|me gusta|qued[oó] bien|"
    r"perfecto|exacto|correcto|aprobado|aprobada|as[íi] est[áa] bien)\b",
    re.IGNORECASE,
)
_REJECTION_RE = re.compile(
    r"\b(no|cambia|c[áa]mbialo|qu[íi]ta|qu[íi]talo|no me gusta|mejor|"
    r"corrige|incorrecto|mal)\b",
    re.IGNORECASE,
)

_TRANSLATION_MARKERS = ("nombre en", "nombre en inglés", "alérgenos", "alergenos")


def handle(
    message: str,
    state: AgentState,
    image_b64: str | None,
    history: list[dict[str, str]],
) -> tuple[str, AgentState]:
    if image_b64:
        _process_image(image_b64, state)

    image_summary = state.image_analysis.raw if state.image_analysis else None
    cls = classifier.classify(state, message, history, image_summary)

    if cls.intent_changed and state.intent and state.intent != cls.intent:
        logger.info(
            "intent_change",
            extra={"from": state.intent, "to": cls.intent},
        )
        pause_current(state)

    state.intent = cls.intent

    if cls.intent == "traducir":
        _resolve_approval(message, state)

    reply = _dispatch(cls.intent, message, state, history)

    if cls.intent == "traducir":
        _post_translation_bookkeeping(message, reply, state)

    if _current_task_finished(cls.intent, state) and state.paused_task is not None:
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
            extras = entities[1:]
            state.current_dish = CurrentDish(entity=primary, variant=None, user_ingredients=[])
            if extras:
                state.dish_queue = extras + state.dish_queue


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
    if state.current_dish is None:
        entities = entity_mapper.map_dishes(message)
        primary = entities[0]
        extras = entities[1:]
        state.current_dish = CurrentDish(
            entity=primary, variant=None, user_ingredients=[]
        )
        if extras:
            state.dish_queue = extras + state.dish_queue
        if state.mode is None:
            state.mode = "menu" if state.dish_queue else "platillo"

    cd = state.current_dish
    _maybe_capture_ingredients(message, cd)

    kb = retrieval.get_dish_context(cd.entity)

    return _generate_with_validation(
        intent="traducir",
        message=message,
        state=state,
        history=history,
        kb_context=kb,
        user_ingredients=cd.user_ingredients,
    )


def _handle_static(
    message: str,
    state: AgentState,
    history: list[dict[str, str]],
    topic: str,
) -> str:
    kb = retrieval.get_static(topic)  # type: ignore[arg-type]
    return _generate_with_validation(
        intent=topic,
        message=message,
        state=state,
        history=history,
        kb_context=kb,
        user_ingredients=[],
    )


def _handle_fallback(
    message: str,
    state: AgentState,
    history: list[dict[str, str]],
) -> str:
    return _generate_with_validation(
        intent="fallback",
        message=message,
        state=state,
        history=history,
        kb_context="",
        user_ingredients=[],
    )


def _generate_with_validation(
    intent: str,
    message: str,
    state: AgentState,
    history: list[dict[str, str]],
    kb_context: str,
    user_ingredients: list[str],
) -> str:
    gi = GenInput(
        state=state,
        intent=intent,
        message=message,
        kb_context=kb_context,
        history=history,
    )
    try:
        reply = generation.generate(gi)
    except bedrock_client.BedrockError as e:
        logger.warning("generation_failed_first", extra={"error": str(e)})
        return _GENERIC_FALLBACK

    ok, reason = validation.validate(reply, kb_context, user_ingredients)
    if ok:
        return reply

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
        reply2 = generation.generate(gi2)
    except bedrock_client.BedrockError as e:
        logger.warning("generation_failed_second", extra={"error": str(e)})
        return _GENERIC_FALLBACK

    ok2, reason2 = validation.validate(reply2, kb_context, user_ingredients)
    if ok2:
        return reply2

    logger.info("validation_failed_second", extra={"reason": reason2})
    return _GENERIC_FALLBACK


def _maybe_capture_ingredients(message: str, cd: CurrentDish) -> None:
    """Heuristic: if the user appears to be listing ingredients (uses commas
    or 'lleva ...' / 'tiene ...' / 'con ...'), capture them. Best-effort, the
    LLM ultimately uses the raw message too."""
    if cd.user_ingredients:
        return
    lowered = message.lower()
    triggers = ("lleva", "tiene", "con ", "ingredientes", ",")
    if not any(t in lowered for t in triggers):
        return
    candidates = re.split(r",|\sy\s|;", message)
    cleaned = [c.strip(" .") for c in candidates if c.strip()]
    cleaned = [c for c in cleaned if 2 <= len(c) <= 60 and " " not in c[:1]]
    if 2 <= len(cleaned) <= 20:
        cd.user_ingredients = cleaned


def _resolve_approval(message: str, state: AgentState) -> None:
    if not state.completed:
        return
    last = state.completed[-1]
    if last.approved is not None:
        return
    if _APPROVAL_RE.search(message):
        last.approved = True
        _advance_after_approval(state)
    elif _REJECTION_RE.search(message):
        last.approved = False


def _advance_after_approval(state: AgentState) -> None:
    state.current_dish = None
    if state.mode == "menu" and state.dish_queue:
        next_entity = state.dish_queue.pop(0)
        state.current_dish = CurrentDish(
            entity=next_entity, variant=None, user_ingredients=[]
        )
    else:
        state.mode = None


def _post_translation_bookkeeping(
    message: str,
    reply: str,
    state: AgentState,
) -> None:
    """If the assistant produced what looks like a final translation, persist
    it into state.completed with approved=None until the next user turn."""
    cd = state.current_dish
    if cd is None:
        return
    if not _looks_like_final_translation(reply):
        return
    if state.completed and state.completed[-1].approved is None and \
            state.completed[-1].dish == _user_dish_label(message, cd):
        return

    allergens = _extract_allergens(reply)
    state.completed.append(CompletedDish(
        dish=_user_dish_label(message, cd),
        ingredients=list(cd.user_ingredients),
        translation_en=_extract_field(reply, "nombre en") or "",
        description_en=_extract_field(reply, "descripción en")
        or _extract_field(reply, "descripcion en")
        or "",
        allergens=allergens,
        approved=None,
    ))


def _looks_like_final_translation(reply: str) -> bool:
    lowered = reply.lower()
    return sum(1 for m in _TRANSLATION_MARKERS if m in lowered) >= 2


def _extract_field(reply: str, key_prefix: str) -> str | None:
    pattern = rf"{re.escape(key_prefix)}[^:]*:\s*(.+?)(?:\n|$)"
    m = re.search(pattern, reply, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip(" *_-")


def _extract_allergens(reply: str) -> list[str]:
    m = re.search(r"al[ée]rgenos?\s*[:\-]\s*(.+?)(?:\n|$)", reply, re.IGNORECASE)
    if not m:
        return []
    parts = re.split(r"[,;/]", m.group(1))
    return [p.strip(" .*_-") for p in parts if p.strip()]


def _user_dish_label(message: str, cd: CurrentDish) -> str:
    text = (message or "").strip()
    if 0 < len(text) <= 120:
        return text
    if cd.entity == CUSTOM_ENTITY:
        return "platillo personalizado"
    return cd.entity


def _current_task_finished(intent: str, state: AgentState) -> bool:
    if intent != "traducir":
        return False
    if state.completed and state.completed[-1].approved is True:
        return state.current_dish is None and not state.dish_queue
    return False
