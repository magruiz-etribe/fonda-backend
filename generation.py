from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Final

import bedrock_client
import config
from classifier import ClassifierResult
from prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_PROMPT: Final[str] = "generation_system.txt"

_FALLBACK: Final[dict[str, Any]] = {
    "response": ["Disculpa, tuve un problema. ¿Puedes repetir tu mensaje? 😊"],
    "current_dishes": [],
    "buttons": [],
    "link": None,
}


@dataclass
class GenResult:
    response: list[str] = field(default_factory=list)
    current_dishes: list[str] = field(default_factory=list)
    buttons: list[str] = field(default_factory=list)
    flags: dict = field(default_factory=dict)
    menu_entry: dict | None = None
    link: dict | None = None
    links: list[dict] = field(default_factory=list)
    completeness_confirmed: bool | None = None
    allergens_confirmed: bool | None = None
    gluten_confirmed: bool | None = None
    spicy_confirmed: bool | None = None


def generate(
    cr: ClassifierResult,
    message: str,
    kb_context: str,
    history: list[dict[str, str]],
    confirmation_state: dict | None = None,
    trigger_info: dict | None = None,
) -> GenResult:
    system = load_prompt(_PROMPT)
    user_text = _build_user_text(cr, message, kb_context, history, confirmation_state, trigger_info)
    messages = [{"role": "user", "content": [{"text": user_text}]}]

    try:
        raw = bedrock_client.converse(
            config.NOVA_PRO_MODEL_ID,
            system,
            messages,
            inference_config={
                "maxTokens": config.GEN_MAX_TOKENS,
                "temperature": 0.5,
            },
        )
    except bedrock_client.BedrockError as e:
        logger.warning("generation_bedrock_error", extra={"error": str(e)})
        return GenResult(**_FALLBACK)

    return _parse(raw)


def _build_user_text(
    cr: ClassifierResult,
    message: str,
    kb_context: str,
    history: list[dict[str, str]],
    confirmation_state: dict | None = None,
    trigger_info: dict | None = None,
) -> str:
    hist_lines: list[str] = []
    for h in history:
        role = "usuario" if h.get("role") == "user" else "agente"
        text = str(h.get("text", "")).strip().replace("\n", " ")
        if len(text) > 400:
            text = text[:397] + "…"
        hist_lines.append(f"  {role}: {text}")
    hist_block = "\n".join(hist_lines) if hist_lines else "(sin historial)"

    kb_block = kb_context.strip() or "(sin contexto KB — platillo personalizado o tema sin ficha)"

    slots_data = [
        {"entity": s.entity, "slot_name": s.slot_name, "options": s.options}
        for s in cr.pending_slots
    ]
    pending_block = json.dumps(slots_data, ensure_ascii=False)
    resolved_block = json.dumps(cr.resolved_variants, ensure_ascii=False)

    # Confirmation state block — only injected for traduccion flows with dishes
    conf_block = ""
    stage_directive = ""
    if confirmation_state is not None:
        cs = confirmation_state
        ti = trigger_info or {}
        conf_block = (
            f"completeness_confirmed: {_fmt_bool(cs.get('completeness_confirmed'))}\n"
            f"allergens_confirmed: {_fmt_bool(cs.get('allergens_confirmed'))}\n"
            f"gluten_confirmed: {_fmt_bool(cs.get('gluten_confirmed'))}\n"
            f"spicy_confirmed: {_fmt_bool(cs.get('spicy_confirmed'))}\n"
            f"allergen_triggers: {json.dumps(ti.get('allergen_triggers', []), ensure_ascii=False)}\n"
            f"gluten_triggers: {json.dumps(ti.get('gluten_triggers', []), ensure_ascii=False)}\n"
            f"spicy_triggers: {json.dumps(ti.get('spicy_triggers', []), ensure_ascii=False)}\n"
        )
        # Only inject stage directive when there are no pending variant/slot questions.
        # If pending_slots is not empty, the CHECK PREVIO step 1 handles it.
        if not cr.pending_slots and not cr.translate_now:
            stage_directive = _build_stage_directive(cs, ti, message)

    return (
        f"{stage_directive}"
        f"Intención: {cr.intent}\n"
        f"Platillos en contexto: {cr.current_dishes}\n"
        f"translate_now: {str(cr.translate_now).lower()}\n"
        f"pending_slots: {pending_block}\n"
        f"resolved_variants: {resolved_block}\n"
        f"{conf_block}\n"
        f"Contexto KB:\n{kb_block}\n\n"
        f"Historial:\n{hist_block}\n\n"
        f"Mensaje del usuario: \"{message}\"\n\n"
        "Devuelve únicamente el JSON."
    )


_CONFIRM_RE: re.Pattern[str] = re.compile(
    r"^(✅|❌|sí|si\b|no\b|listo|claro|ninguno|eso es todo|correcto|exacto)",
    re.IGNORECASE,
)


def _is_confirmation(message: str) -> bool:
    """True when the user's message looks like a yes/no confirmation response."""
    return bool(_CONFIRM_RE.match(message.strip()))


def _next_stage(ti: dict, after: str) -> tuple[str, str]:
    """Return (stage_code, label) for the stage that follows `after`."""
    order = ["A2", "A3", "A4", "B"]
    checks: dict[str, bool] = {
        "A2": bool(ti.get("allergen_triggers")),
        "A3": bool(ti.get("gluten_triggers")),
        "A4": bool(ti.get("spicy_triggers")),
    }
    descs: dict[str, str] = {
        "A2": "la pregunta de alergenos (ETAPA A2)",
        "A3": "la pregunta de ingredientes con posible gluten (ETAPA A3)",
        "A4": "la pregunta de ingredientes picantes (ETAPA A4)",
        "B":  "la descripcion en espanol (ETAPA B)",
    }
    start = order.index(after) + 1 if after in order else 0
    for stage in order[start:]:
        if stage == "B" or checks.get(stage):
            return stage, descs[stage]
    return "B", descs["B"]


def _build_stage_directive(cs: dict, ti: dict, message: str) -> str:
    """Inject explicit stage directive so the model never has to guess what to do."""
    responding = _is_confirmation(message)

    if cs.get("completeness_confirmed") is None:
        if responding:
            nxt, ndesc = _next_stage(ti, "A1")
            return (
                f"ETAPA A1 PROCESA RESPUESTA: si confirma → completeness_confirmed: true, "
                f"siguiente paso obligatorio ETAPA {nxt}, en este turno 1 globo = SOLO {ndesc}. "
                f"Si agrega ingredientes → completeness_confirmed: null y pide mas.\n"
            )
        return "ETAPA A1 HAZ PREGUNTA: pregunta completitud. 1 globo. PROHIBIDO generar descripcion.\n"

    if ti.get("allergen_triggers") and cs.get("allergens_confirmed") is None:
        if responding:
            nxt, ndesc = _next_stage(ti, "A2")
            return (
                f"ETAPA A2 PROCESA RESPUESTA: allergens_confirmed true si confirma, false si niega. "
                f"Siguiente paso obligatorio ETAPA {nxt}. En este turno 1 globo = SOLO {ndesc}.\n"
            )
        return "ETAPA A2 HAZ PREGUNTA: usa formato exacto de ETAPA A2. 1 globo.\n"

    if ti.get("gluten_triggers") and cs.get("gluten_confirmed") is None:
        if responding:
            nxt, ndesc = _next_stage(ti, "A3")
            return (
                f"ETAPA A3 PROCESA RESPUESTA: gluten_confirmed true si confirma, false si niega. "
                f"Siguiente paso obligatorio ETAPA {nxt}. En este turno 1 globo = SOLO {ndesc}.\n"
            )
        return "ETAPA A3 HAZ PREGUNTA: usa formato exacto de ETAPA A3. 1 globo.\n"

    if ti.get("spicy_triggers") and cs.get("spicy_confirmed") is None:
        if responding:
            return (
                "ETAPA A4 PROCESA RESPUESTA: spicy_confirmed true si confirma, false si niega. "
                "Siguiente paso obligatorio ETAPA B. En este turno 1 globo = SOLO la descripcion en espanol (ETAPA B).\n"
            )
        return "ETAPA A4 HAZ PREGUNTA: usa formato exacto de ETAPA A4. 1 globo.\n"

    return ""


def _fmt_bool(val: bool | None) -> str:
    if val is True:
        return "true"
    if val is False:
        return "false"
    return "null"


def _parse(raw: str) -> GenResult:
    try:
        data = bedrock_client.parse_json_strict(raw)
    except Exception as e:
        logger.warning(
            "generation_parse_error",
            extra={"error": str(e), "raw": raw[:300]},
        )
        return GenResult(**_FALLBACK)

    if not isinstance(data, dict):
        return GenResult(**_FALLBACK)

    raw_response = data.get("response") or []
    response: list[str] = []
    if isinstance(raw_response, list):
        for r in raw_response:
            if isinstance(r, str) and r.strip():
                response.append(r.strip())
    if not response:
        logger.warning("generation_empty_response", extra={"raw": str(data)[:200]})
        return GenResult(**_FALLBACK)

    raw_dishes = data.get("current_dishes") or []
    dishes: list[str] = []
    if isinstance(raw_dishes, list):
        for d in raw_dishes:
            if isinstance(d, str) and d.strip():
                dishes.append(d.strip().lower())

    raw_buttons = data.get("buttons") or []
    buttons: list[str] = []
    if isinstance(raw_buttons, list):
        for b in raw_buttons:
            if isinstance(b, str) and b.strip():
                buttons.append(b.strip())

    # Strip any button label that leaked into a response bubble
    if buttons:
        button_labels = {b.strip() for b in buttons}
        cleaned: list[str] = []
        for bubble in response:
            lines = bubble.split("\n")
            while lines and lines[-1].strip() in button_labels:
                lines.pop()
            stripped = "\n".join(lines).strip()
            if stripped:
                cleaned.append(stripped)
        if cleaned:
            response = cleaned

    logger.info(
        "generation_ok",
        extra={
            "bubbles": len(response),
            "buttons": len(buttons),
            "current_dishes_out": dishes,
        },
    )

    return GenResult(
        response=response,
        current_dishes=dishes,
        buttons=buttons,
        completeness_confirmed=_parse_bool_or_none(data.get("completeness_confirmed")),
        allergens_confirmed=_parse_bool_or_none(data.get("allergens_confirmed")),
        gluten_confirmed=_parse_bool_or_none(data.get("gluten_confirmed")),
        spicy_confirmed=_parse_bool_or_none(data.get("spicy_confirmed")),
    )


def _parse_bool_or_none(val: Any) -> bool | None:
    if val is True:
        return True
    if val is False:
        return False
    return None
