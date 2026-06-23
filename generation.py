from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Final

import bedrock_client
import config
import llm_schemas
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
    intent: str = ""
    # State machine fields
    dish_status: str | None = None
    collected_ingredients: list[str] = field(default_factory=list)
    detected_flags: list[str] = field(default_factory=list)
    variables_complete: bool | None = None
    save_to_menu: bool = False
    # Legacy confirmation fields (kept for backward compat with non-traduccion path)
    completeness_confirmed: bool | None = None
    allergens_confirmed: bool | None = None
    gluten_confirmed: bool | None = None
    spicy_confirmed: bool | None = None
    resolved_variants: dict = field(default_factory=dict)
    extra_user_ingredients: list[str] = field(default_factory=list)


def generate(
    cr: ClassifierResult,
    message: str,
    kb_context: str,
    history: list[dict[str, str]],
    confirmation_state: dict | None = None,
    trigger_info: dict | None = None,
    platform: str = "",
    dish_context: dict | None = None,
) -> GenResult:
    system = load_prompt(_PROMPT)
    user_text = _build_user_text(cr, message, kb_context, history, confirmation_state, trigger_info, platform, dish_context)
    messages = [{"role": "user", "content": [{"text": user_text}]}]

    try:
        data = bedrock_client.converse_json(
            config.NOVA_2_LITE_MODEL_ID,
            system,
            messages,
            schema=llm_schemas.GENERATION,
            tool_name="generate_response",
            tool_description="Generate the assistant response bubbles and buttons",
            inference_config={
                "maxTokens": config.GEN_MAX_TOKENS,
                "temperature": 0.5,
            },
            stage="generation",
        )
    except bedrock_client.BedrockError as e:
        logger.warning("generation_bedrock_error", extra={"error": str(e)})
        return GenResult(**_FALLBACK)

    return _parse_data(data)


def _build_user_text(
    cr: ClassifierResult,
    message: str,
    kb_context: str,
    history: list[dict[str, str]],
    confirmation_state: dict | None = None,
    trigger_info: dict | None = None,
    platform: str = "",
    dish_context: dict | None = None,
) -> str:
    hist_lines: list[str] = []
    for h in history[-6:]:
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

    dish_ctx_block = ""
    if dish_context:
        dc = dish_context
        rv = json.dumps(dc.get("resolved_variants") or {}, ensure_ascii=False)
        extras = ", ".join(dc.get("extra_ingredients") or []) or "(ninguno)"
        last_es = dc.get("last_description_es") or ""
        dish_ctx_block = (
            "[CONTEXTO DEL PLATILLO EN CURSO — fuente de verdad]\n"
            f"Platillo principal: {dc.get('main_dish', '')}\n"
            f"Variantes confirmadas: {rv}\n"
            f"Ingredientes extras confirmados: {extras}\n"
        )
        if last_es:
            dish_ctx_block += f"Última descripción en español presentada:\n{last_es}\n"
        dish_ctx_block += "\n"

    platform_line = f"platform: {platform}\n" if platform else ""
    return (
        f"{dish_ctx_block}"
        f"{stage_directive}"
        f"Intención: {cr.intent}\n"
        f"{platform_line}"
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


_YES_RE: re.Pattern[str] = re.compile(
    r"^(✅|sí|si\b|listo|claro|correcto|exacto|contiene)",
    re.IGNORECASE,
)


def _is_yes(message: str) -> bool:
    """True when the confirmation response is affirmative."""
    return bool(_YES_RE.match(message.strip()))


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
            if nxt == "B":
                next_instruction = (
                    "siguiente paso obligatorio ETAPA B: EXACTAMENTE 2 globos — "
                    "globo 1 = card **Nombre**\\nDescripcion, globo 2 = '¿Te parece bien? 😊...'"
                )
            else:
                next_instruction = f"siguiente paso obligatorio ETAPA {nxt}: 1 globo = SOLO {ndesc}"
            return (
                f"ETAPA A1 PROCESA RESPUESTA — DOS RAMAS EXCLUYENTES, elige UNA:\n"
                f"  RAMA CONFIRMA: si el fondero confirma que ya no agrega nada → "
                f"completeness_confirmed: true, {next_instruction}. "
                f"PROHIBIDO incluir '¡Anotado!' ni '¿Algo más?' en esta rama.\n"
                f"  RAMA AGREGA: si el fondero agrega ingredientes nuevos → "
                f"completeness_confirmed: true, {next_instruction}. "
                f"PROHIBIDO '¿Algo más?', '✅ Listo, eso es todo!'. Avanza igual que RAMA CONFIRMA.\n"
            )
        return (
            "ETAPA A1 HAZ PREGUNTA — EXACTAMENTE 1 globo + buttons ['✅ Listo, eso es todo!']. "
            "PROHIBIDO: segundo globo, descripcion del platillo, '¿Te parece bien?', '✅ Adaptar al inglés'. "
            "completeness_confirmed: null.\n"
        )

    if ti.get("allergen_triggers") and cs.get("allergens_confirmed") is None:
        if responding:
            nxt, ndesc = _next_stage(ti, "A2")
            return (
                f"ETAPA A2 PROCESA RESPUESTA: allergens_confirmed true si confirma, false si niega. "
                f"Siguiente paso obligatorio ETAPA {nxt}. En este turno 1 globo = SOLO {ndesc}.\n"
            )
        return (
            "ETAPA A2 HAZ PREGUNTA — EXACTAMENTE 1 globo con formato de ETAPA A2 + "
            "buttons ['✅ Sí, contiene alguno', '❌ No, ninguno de esos']. "
            "PROHIBIDO: segundo globo, descripcion del platillo, '¿Te parece bien?', '✅ Adaptar al inglés'.\n"
        )

    if ti.get("gluten_triggers") and cs.get("gluten_confirmed") is None:
        if responding:
            nxt, ndesc = _next_stage(ti, "A3")
            return (
                f"ETAPA A3 PROCESA RESPUESTA: gluten_confirmed true si confirma, false si niega. "
                f"Siguiente paso obligatorio ETAPA {nxt}. En este turno 1 globo = SOLO {ndesc}.\n"
            )
        return (
            "ETAPA A3 HAZ PREGUNTA — EXACTAMENTE 1 globo con formato de ETAPA A3 + "
            "buttons ['✅ Sí, contiene alguno', '❌ No, ninguno de esos']. "
            "PROHIBIDO: segundo globo, descripcion del platillo, '¿Te parece bien?', '✅ Adaptar al inglés'.\n"
        )

    if ti.get("spicy_triggers") and cs.get("spicy_confirmed") is None:
        if responding:
            return (
                "ETAPA A4 PROCESA RESPUESTA: spicy_confirmed true si confirma, false si niega. "
                "Siguiente paso obligatorio ETAPA B. En este turno 1 globo = SOLO la descripcion en espanol (ETAPA B).\n"
            )
        return (
            "ETAPA A4 HAZ PREGUNTA — EXACTAMENTE 1 globo con formato de ETAPA A4 + "
            "buttons ['✅ Sí, contiene alguno', '❌ No, ninguno de esos']. "
            "PROHIBIDO: segundo globo, descripcion del platillo, '¿Te parece bien?', '✅ Adaptar al inglés'.\n"
        )

    return ""


def _fmt_bool(val: bool | None) -> str:
    if val is True:
        return "true"
    if val is False:
        return "false"
    return "null"


_METADATA_LEAK_RE: re.Pattern[str] = re.compile(
    r"^(current_dishes|current_dish|buttons|intent|links|flags|platform|"
    r"dish_status|collected_ingredients|pending_slots|resolved_variants|"
    r"translate_now|completeness_confirmed|allergens_confirmed|"
    r"gluten_confirmed|spicy_confirmed)\s*[\[=:{(]",
    re.IGNORECASE,
)


def _parse_data(data: dict) -> GenResult:
    if not isinstance(data, dict):
        return GenResult(**_FALLBACK)

    raw_response = data.get("response") or []
    response: list[str] = []
    if isinstance(raw_response, list):
        for r in raw_response:
            if isinstance(r, str) and r.strip():
                bubble = r.strip()
                if _METADATA_LEAK_RE.match(bubble):
                    logger.warning("generation_metadata_leak_dropped", extra={"bubble": bubble[:80]})
                    continue
                response.append(bubble)
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


_SPANISH_ACCENTS: Final[re.Pattern[str]] = re.compile(r"[áéíóúñü]", re.IGNORECASE)
_SPANISH_MARKERS: Final[re.Pattern[str]] = re.compile(
    r"\b("
    r"el|la|los|las|del|con|de|en|y|o|un|una|unos|unas|que|por|para|"
    r"acomp[aá]a|rellen[ao]s?|servid[ao]s?|bañad[ao]s?|preparad[ao]s?|"
    r"tortillas?|queso|crema|lechuga|masa|chile|salsa|frijol|pollo|carne"
    r")\b",
    re.IGNORECASE,
)
_ENGLISH_MARKERS: Final[re.Pattern[str]] = re.compile(
    r"\b(with|served|filled|stuffed|topped|grilled|fried|fresh|cheese|cream|sauce|tortilla|beans|rice|chicken|beef|pork)\b",
    re.IGNORECASE,
)

_TRANSLATE_CARD_SYSTEM: Final[str] = """\
Eres un traductor de menús de fondas mexicanas al inglés (estilo menú de restaurante en EE.UU.).
Devuelve ÚNICAMENTE JSON válido, sin markdown ni texto extra.

Reglas:
- Traduce título y descripción al inglés.
- 2-3 oraciones, solo ingredientes visibles y preparación factual.
- Sin adjetivos de marketing (delicious, creamy, savory, juicy, etc.).
- Sin emojis.
- No agregues calificadores regionales que no estén en el español.
"""


def looks_spanish(text: str) -> bool:
    """Heuristic: True when text is likely Spanish menu prose."""
    if not text.strip():
        return False
    if _SPANISH_ACCENTS.search(text):
        return True
    spanish_hits = len(_SPANISH_MARKERS.findall(text))
    english_hits = len(_ENGLISH_MARKERS.findall(text))
    return spanish_hits >= 2 and english_hits == 0


# ── Extracting deterministic fallback ─────────────────────────────────────────

_VARIABLE_QUESTION_TEMPLATES: Final[dict[str, str]] = {
    "relleno": "¿Con qué relleno preparas {dish}?",
    "tipo_de_salsa": "¿Con qué tipo de salsa preparas {dish}?",
    "tipo_de_carne": "¿Qué proteína llevas en {dish}?",
    "tipo_de_caldo": "¿Qué tipo de caldo usas para {dish}?",
    "acompañamiento": "¿Con qué acompañas los {dish}?",
    "acompanamiento": "¿Con qué acompañas los {dish}?",
}


def _normalize_ingredient(s: str) -> str:
    s = s.strip().lower().replace("_", " ")
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    return " ".join(s.split())


def _ingredient_matches_option(ingredient: str, option: str) -> bool:
    ing = _normalize_ingredient(ingredient)
    opt = _normalize_ingredient(option)
    if not ing or not opt:
        return False
    return ing == opt or opt in ing or ing in opt


def _variable_covered(
    variable: str,
    collected: list[str],
    options: list[str],
    base_defaults: list[str] | None = None,
) -> bool:
    if options:
        for ing in collected:
            for opt in options:
                if _ingredient_matches_option(ing, opt):
                    return True
        var_label = _normalize_ingredient(variable)
        for ing in collected:
            if _ingredient_matches_option(ing, var_label):
                return True
        return False

    defaults = {_normalize_ingredient(d) for d in (base_defaults or [])}
    for ing in collected:
        normalized = _normalize_ingredient(ing)
        if normalized and normalized not in defaults:
            return True
    return False


def _find_first_missing_variable(
    collected: list[str],
    variables_requeridas: list[str],
    variable_opciones: dict,
    base_defaults: list[str] | None = None,
) -> str | None:
    for var in variables_requeridas:
        raw_options = variable_opciones.get(var) or []
        options = (
            [str(o).strip() for o in raw_options if str(o).strip()]
            if isinstance(raw_options, list)
            else []
        )
        if not _variable_covered(var, collected, options, base_defaults):
            return var
    return None


def _format_dish_display_name(current_dish: str, kb_data: dict) -> str:
    names = kb_data.get("common_names") or []
    if names and isinstance(names[0], str) and names[0].strip():
        return names[0].strip()
    return current_dish.replace("_", " ")


def _build_variable_question(variable: str, dish_display: str) -> str:
    template = _VARIABLE_QUESTION_TEMPLATES.get(variable)
    if template:
        return template.format(dish=dish_display)
    label = variable.replace("_", " ")
    return f"¿Qué {label} llevas en {dish_display}?"


def _options_to_buttons(options: list[str]) -> list[str]:
    buttons: list[str] = []
    for opt in options:
        s = str(opt).strip()
        if not s:
            continue
        buttons.append(s[0].upper() + s[1:] if len(s) > 1 else s.upper())
    return buttons


def _build_extracting_deterministic_fallback(
    *,
    current_dish: str,
    collected_ingredients: list[str],
    kb_data: dict,
) -> GenResult | None:
    variables_requeridas = list(kb_data.get("variables_requeridas") or [])
    if not variables_requeridas:
        return None

    variable_opciones = kb_data.get("variable_opciones") or {}
    base_defaults = list(kb_data.get("ingredientes_base_default") or [])
    collected = list(collected_ingredients)
    missing = _find_first_missing_variable(
        collected, variables_requeridas, variable_opciones, base_defaults
    )

    if missing is None:
        return GenResult(
            response=[],
            variables_complete=True,
            collected_ingredients=collected,
            buttons=[],
        )

    dish_display = _format_dish_display_name(current_dish, kb_data)
    question = _build_variable_question(missing, dish_display)
    raw_options = variable_opciones.get(missing) or []
    options = (
        [str(o).strip() for o in raw_options if str(o).strip()]
        if isinstance(raw_options, list)
        else []
    )
    return GenResult(
        response=[question],
        variables_complete=False,
        collected_ingredients=collected,
        buttons=_options_to_buttons(options),
    )


def _merge_collected_ingredients(*sources: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for source in sources:
        for item in source:
            normalized = _normalize_ingredient(item)
            if normalized and normalized not in seen:
                seen.add(normalized)
                merged.append(normalized)
    return merged


def _prefill_collected_from_message(
    message: str,
    collected: list[str],
    kb_data: dict,
) -> list[str]:
    """Capture KB option values explicitly mentioned in the user's message."""
    merged = _merge_collected_ingredients(collected)
    seen = set(merged)
    message_norm = _normalize_ingredient(message)
    if not message_norm:
        return merged

    for options in (kb_data.get("variable_opciones") or {}).values():
        if not isinstance(options, list):
            continue
        for opt in options:
            opt_norm = _normalize_ingredient(str(opt))
            if opt_norm and opt_norm in message_norm and opt_norm not in seen:
                seen.add(opt_norm)
                merged.append(opt_norm)

    return merged


def _fill_extracting_buttons(result: GenResult, kb_data: dict) -> GenResult:
    """Inject KB option buttons when the LLM asks a question but omits buttons."""
    if result.buttons or result.variables_complete or not result.response:
        return result

    variables_requeridas = list(kb_data.get("variables_requeridas") or [])
    variable_opciones = kb_data.get("variable_opciones") or {}
    base_defaults = list(kb_data.get("ingredientes_base_default") or [])
    collected = list(result.collected_ingredients or [])

    missing = _find_first_missing_variable(
        collected, variables_requeridas, variable_opciones, base_defaults
    )
    if not missing:
        return result

    raw_options = variable_opciones.get(missing) or []
    if not isinstance(raw_options, list):
        return result
    options = [str(o).strip() for o in raw_options if str(o).strip()]
    if not options:
        return result

    result.buttons = _options_to_buttons(options)
    logger.info(
        "gen_extracting_buttons_injected",
        extra={"variable": missing, "buttons": result.buttons},
    )
    return result


def variables_satisfied(collected: list[str], kb_data: dict) -> bool:
    """True when every variables_requeridas entry is covered by collected ingredients."""
    variables_requeridas = list(kb_data.get("variables_requeridas") or [])
    if not variables_requeridas:
        return True
    variable_opciones = kb_data.get("variable_opciones") or {}
    base_defaults = list(kb_data.get("ingredientes_base_default") or [])
    return (
        _find_first_missing_variable(
            collected, variables_requeridas, variable_opciones, base_defaults
        )
        is None
    )


def prefill_collected(message: str, collected: list[str], kb_data: dict) -> list[str]:
    """Merge session collected with KB option values mentioned in the user message."""
    return _prefill_collected_from_message(message, collected, kb_data)


def _finalize_extracting_result(
    result: GenResult,
    *,
    current_dish: str,
    kb_data: dict,
) -> GenResult:
    """Reconcile LLM output with KB rules: one question, correct completion, buttons."""
    collected = list(result.collected_ingredients or [])
    variables_requeridas = list(kb_data.get("variables_requeridas") or [])
    variable_opciones = kb_data.get("variable_opciones") or {}
    base_defaults = list(kb_data.get("ingredientes_base_default") or [])

    missing = _find_first_missing_variable(
        collected, variables_requeridas, variable_opciones, base_defaults
    )

    if missing is None:
        if result.response or not result.variables_complete:
            logger.info(
                "gen_extracting_force_complete_from_collected",
                extra={"collected": collected, "dish": current_dish},
            )
        return GenResult(
            response=[],
            variables_complete=True,
            collected_ingredients=collected,
            buttons=[],
        )

    det = _build_extracting_deterministic_fallback(
        current_dish=current_dish,
        collected_ingredients=collected,
        kb_data=kb_data,
    )
    if det is not None:
        if det.response or not det.variables_complete:
            logger.info(
                "gen_extracting_question_normalized",
                extra={
                    "dish": current_dish,
                    "missing": missing,
                    "llm_bubbles": len(result.response),
                },
            )
        return _fill_extracting_buttons(det, kb_data) if not det.variables_complete else det

    return _fill_extracting_buttons(result, kb_data)


# ── New state-machine generation functions ────────────────────────────────────

def generate_extracting(
    *,
    current_dish: str,
    companions: list[str],
    collected_ingredients: list[str],
    message: str,
    history: list[dict[str, str]],
    kb_data: dict,
) -> GenResult:
    effective_collected = _prefill_collected_from_message(
        message, collected_ingredients, kb_data
    )
    system = load_prompt("extracting_system.txt")
    user_text = _build_extracting_text(
        current_dish, companions, effective_collected, message, history, kb_data
    )
    messages = [{"role": "user", "content": [{"text": user_text}]}]

    variables_requeridas = list(kb_data.get("variables_requeridas") or [])

    # Two attempts: first natural (0.3), retry deterministic (0.0) if LLM returns
    # variables_complete=false with empty response.
    for attempt, temperature in enumerate([0.3, 0.0]):
        try:
            data = bedrock_client.converse_json(
                config.NOVA_2_LITE_MODEL_ID,
                system,
                messages,
                schema=llm_schemas.EXTRACTING,
                tool_name="extract_variables",
                tool_description="Extract dish variables and ask for missing ones",
                inference_config={"maxTokens": 512, "temperature": temperature},
                stage="gen_extracting",
            )
        except bedrock_client.BedrockError as e:
            logger.warning("gen_extracting_bedrock_error", extra={"error": str(e)})
            return GenResult(**_FALLBACK)

        parsed = _parse_extracting_data(data, effective_collected)
        parsed.collected_ingredients = _merge_collected_ingredients(
            effective_collected,
            parsed.collected_ingredients,
        )
        result = _finalize_extracting_result(
            parsed,
            current_dish=current_dish,
            kb_data=kb_data,
        )
        if result.response or result.variables_complete:
            if attempt > 0:
                logger.info("gen_extracting_retry_ok", extra={"attempt": attempt})
            return result

        # LLM returned variables_complete=false with empty response — contradictory output.
        # If there are genuinely no required variables, force-complete so the flow continues.
        if not variables_requeridas:
            logger.warning(
                "gen_extracting_force_complete",
                extra={"attempt": attempt, "reason": "no_variables_required"},
            )
            return GenResult(
                response=[],
                variables_complete=True,
                collected_ingredients=result.collected_ingredients or list(collected_ingredients),
                buttons=[],
            )

        logger.warning(
            "gen_extracting_empty_question",
            extra={"attempt": attempt, "data": str(data)[:200]},
        )

        det = _build_extracting_deterministic_fallback(
            current_dish=current_dish,
            collected_ingredients=result.collected_ingredients or list(collected_ingredients),
            kb_data=kb_data,
        )
        if det is not None:
            logger.info(
                "gen_extracting_deterministic_fallback",
                extra={
                    "attempt": attempt,
                    "variables_complete": det.variables_complete,
                    "missing_var": None if det.variables_complete else "inferred",
                },
            )
            return det

    return GenResult(**_FALLBACK)


def _build_extracting_text(
    current_dish: str,
    companions: list[str],
    collected_ingredients: list[str],
    message: str,
    history: list[dict[str, str]],
    kb_data: dict,
) -> str:
    hist_lines: list[str] = []
    for h in history[-6:]:
        role = "usuario" if h.get("role") == "user" else "agente"
        text = str(h.get("text", "")).strip().replace("\n", " ")
        if len(text) > 300:
            text = text[:297] + "…"
        hist_lines.append(f"  {role}: {text}")
    hist_block = "\n".join(hist_lines) if hist_lines else "(sin historial)"

    comp_str = ", ".join(companions) if companions else "(ninguno)"
    vars_req = json.dumps(kb_data.get("variables_requeridas") or [], ensure_ascii=False)
    base_desc = kb_data.get("base_description") or ""
    var_opciones = kb_data.get("variable_opciones") or {}
    opciones_str = json.dumps(var_opciones, ensure_ascii=False) if var_opciones else "{}"

    return (
        f"current_dish: {current_dish}\n"
        f"companions: {comp_str}\n"
        f"variables_requeridas: {vars_req}\n"
        f"variable_opciones: {opciones_str}\n"
        f"collected_ingredients: {json.dumps(collected_ingredients, ensure_ascii=False)}\n"
        f"base_description: {base_desc}\n\n"
        f"Historial:\n{hist_block}\n\n"
        f"Mensaje del usuario: \"{message}\"\n\n"
        "Devuelve únicamente el JSON."
    )


def _parse_extracting_data(raw: dict | str, fallback_collected: list[str]) -> GenResult:
    if isinstance(raw, str):
        try:
            data = bedrock_client.parse_json_lenient(raw)
        except Exception as e:
            logger.warning("gen_extracting_parse_error", extra={"error": str(e), "raw": raw[:200]})
            return GenResult(**_FALLBACK)
    else:
        data = raw

    if not isinstance(data, dict):
        return GenResult(**_FALLBACK)

    raw_response = data.get("response") or []
    response = [r.strip() for r in raw_response if isinstance(r, str) and r.strip()]

    variables_complete = bool(data.get("variables_complete", False))

    raw_collected = data.get("collected_ingredients")
    collected: list[str] = []
    if isinstance(raw_collected, list):
        for c in raw_collected:
            if isinstance(c, str) and c.strip():
                collected.append(c.strip().lower())
    if not collected:
        collected = list(fallback_collected)

    raw_buttons = data.get("buttons") or []
    buttons = [b.strip() for b in raw_buttons if isinstance(b, str) and b.strip()]

    logger.info("gen_extracting_ok", extra={"variables_complete": variables_complete, "collected": collected})

    return GenResult(
        response=response,
        variables_complete=variables_complete,
        collected_ingredients=collected,
        buttons=buttons,
    )


def generate_confirming_flags(
    *,
    current_dish: str,
    companions: list[str],
    collected_ingredients: list[str],
    detected_flags: list[str],
    message: str,
    history: list[dict[str, str]],
) -> GenResult:
    system = load_prompt("confirming_flags_system.txt")
    user_text = _build_confirming_flags_text(
        current_dish, companions, collected_ingredients, detected_flags, message, history
    )
    messages = [{"role": "user", "content": [{"text": user_text}]}]

    try:
        data = bedrock_client.converse_json(
            config.NOVA_2_LITE_MODEL_ID,
            system,
            messages,
            schema=llm_schemas.CONFIRMING_FLAGS,
            tool_name="confirm_flags",
            tool_description="Ask the user to confirm detected allergens",
            inference_config={"maxTokens": 512, "temperature": 0.3},
            stage="gen_confirming_flags",
        )
    except bedrock_client.BedrockError as e:
        logger.warning("gen_confirming_flags_bedrock_error", extra={"error": str(e)})
        return GenResult(**_FALLBACK)

    return _parse_confirming_flags_data(data)


def _build_confirming_flags_text(
    current_dish: str,
    companions: list[str],
    collected_ingredients: list[str],
    detected_flags: list[str],
    message: str,
    history: list[dict[str, str]],
) -> str:
    hist_lines: list[str] = []
    for h in history[-6:]:
        role = "usuario" if h.get("role") == "user" else "agente"
        text = str(h.get("text", "")).strip().replace("\n", " ")
        if len(text) > 300:
            text = text[:297] + "…"
        hist_lines.append(f"  {role}: {text}")
    hist_block = "\n".join(hist_lines) if hist_lines else "(sin historial)"

    comp_str = ", ".join(companions) if companions else "(ninguno)"
    flags_str = ", ".join(detected_flags) if detected_flags else "(ninguno)"

    return (
        f"current_dish: {current_dish}\n"
        f"companions: {comp_str}\n"
        f"collected_ingredients: {json.dumps(collected_ingredients, ensure_ascii=False)}\n"
        f"detected_flags: {flags_str}\n\n"
        f"Historial:\n{hist_block}\n\n"
        f"Mensaje del usuario: \"{message}\"\n\n"
        "Devuelve únicamente el JSON."
    )


def _parse_confirming_flags_data(data: dict) -> GenResult:
    if not isinstance(data, dict):
        return GenResult(**_FALLBACK)

    raw_response = data.get("response") or []
    response = [r.strip() for r in raw_response if isinstance(r, str) and r.strip()]
    # Empty response is valid: LLM signals nothing to confirm; caller will skip to drafting.

    raw_buttons = data.get("buttons") or []
    buttons = [b.strip() for b in raw_buttons if isinstance(b, str) and b.strip()]

    return GenResult(response=response, buttons=buttons)


_DEFAULT_DRAFTING_BUTTONS: Final[list[str]] = ["🍽️ Ayúdame con otro platillo"]


def _call_drafting_llm(system: str, user_text: str, current_dish: str, companions: list[str]) -> GenResult:
    messages = [{"role": "user", "content": [{"text": user_text}]}]
    for attempt, temperature in enumerate([0.5, 0.0]):
        try:
            data = bedrock_client.converse_json(
                config.NOVA_2_LITE_MODEL_ID,
                system,
                messages,
                schema=llm_schemas.DRAFTING,
                tool_name="draft_menu_card",
                tool_description="Generate the bilingual menu card draft",
                inference_config={"maxTokens": config.GEN_MAX_TOKENS, "temperature": temperature},
                stage="gen_drafting",
            )
        except bedrock_client.BedrockError as e:
            logger.warning("gen_drafting_bedrock_error", extra={"error": str(e)})
            return GenResult(**_FALLBACK)

        result = _try_parse_drafting_data(data, current_dish, companions)
        if result is not None:
            if attempt > 0:
                logger.info("gen_drafting_retry_ok", extra={"attempt": attempt})
            return result

        logger.warning("gen_drafting_parse_error", extra={"attempt": attempt, "raw": str(data)[:200]})

    return GenResult(**_FALLBACK)


def generate_drafting(
    *,
    current_dish: str,
    companions: list[str],
    collected_ingredients: list[str],
    detected_flags: list[str],
    message: str,
    history: list[dict[str, str]],
    kb_data: dict,
) -> GenResult:
    system = load_prompt("drafting_system.txt")
    user_text = _build_drafting_text(
        current_dish, companions, collected_ingredients, detected_flags, message, history, kb_data
    )
    return _call_drafting_llm(system, user_text, current_dish, companions)


def generate_draft_edit(
    *,
    current_dish: str,
    companions: list[str],
    collected_ingredients: list[str],
    detected_flags: list[str],
    edit_instruction: str,
    previous_card: str,
    history: list[dict[str, str]],
    kb_data: dict,
) -> GenResult:
    """Re-draft applying a free-form edit instruction to the previous card."""
    system = load_prompt("drafting_system.txt")
    user_text = _build_drafting_text(
        current_dish, companions, collected_ingredients, detected_flags,
        edit_instruction, history, kb_data,
        edit_instruction=edit_instruction,
        previous_card=previous_card,
    )
    return _call_drafting_llm(system, user_text, current_dish, companions)


def _build_drafting_text(
    current_dish: str,
    companions: list[str],
    collected_ingredients: list[str],
    detected_flags: list[str],
    message: str,
    history: list[dict[str, str]],
    kb_data: dict,
    *,
    edit_instruction: str = "",
    previous_card: str = "",
) -> str:
    hist_lines: list[str] = []
    for h in history[-6:]:
        role = "usuario" if h.get("role") == "user" else "agente"
        text = str(h.get("text", "")).strip().replace("\n", " ")
        if len(text) > 300:
            text = text[:297] + "…"
        hist_lines.append(f"  {role}: {text}")
    hist_block = "\n".join(hist_lines) if hist_lines else "(sin historial)"

    comp_str = ", ".join(companions) if companions else "(ninguno)"
    flags_str = ", ".join(detected_flags) if detected_flags else "(ninguno)"
    base_desc = kb_data.get("base_description") or ""

    edit_block = ""
    if edit_instruction and previous_card:
        edit_block = (
            f"MODO EDICIÓN — aplica solo los cambios indicados:\n"
            f"Tarjeta anterior:\n{previous_card}\n\n"
            f"Instrucción de edición: \"{edit_instruction}\"\n\n"
        )

    return (
        f"{edit_block}"
        f"current_dish: {current_dish}\n"
        f"companions: {comp_str}\n"
        f"base_description: {base_desc}\n"
        f"collected_ingredients: {json.dumps(collected_ingredients, ensure_ascii=False)}\n"
        f"detected_flags: {flags_str}\n\n"
        f"Historial:\n{hist_block}\n\n"
        f"Mensaje del usuario: \"{message}\"\n\n"
        "Devuelve únicamente el JSON."
    )


def _fix_char_split(items: list) -> list:
    """Reassemble bubbles when the LLM emits individual characters as separate list items."""
    if not items:
        return items
    single_char = sum(1 for item in items if isinstance(item, str) and len(item) == 1)
    if single_char > len(items) // 2:
        return ["".join(str(item) for item in items)]
    return items


def _try_parse_drafting_data(
    data: dict | str,
    current_dish: str,
    companions: list[str],
) -> GenResult | None:
    if isinstance(data, str):
        try:
            parsed = bedrock_client.parse_json_lenient(data)
        except Exception:
            return None
        data = parsed

    if not isinstance(data, dict):
        return None

    raw_response = data.get("response") or []
    if not isinstance(raw_response, list):
        raw_response = []
    response: list[str] = []
    for r in _fix_char_split(raw_response):
        if not isinstance(r, str) or not r.strip():
            continue
        bubble = r.strip()
        if _METADATA_LEAK_RE.match(bubble):
            logger.warning("gen_drafting_metadata_leak_dropped", extra={"bubble": bubble[:80]})
            continue
        response.append(bubble)
    if not response:
        return None

    raw_buttons = data.get("buttons") or []
    buttons = [b.strip() for b in raw_buttons if isinstance(b, str) and b.strip()]
    if not buttons:
        buttons = list(_DEFAULT_DRAFTING_BUTTONS)

    dishes_out = [current_dish] + companions

    logger.info("gen_drafting_ok", extra={"bubbles": len(response), "buttons": buttons})

    return GenResult(
        response=response,
        buttons=buttons,
        current_dishes=dishes_out,
    )


def translate_menu_card(
    *,
    name_es: str,
    description_es: str,
    name_en_hint: str = "",
) -> dict[str, str] | None:
    """Focused translation call when ETAPA C returns Spanish by mistake."""
    if not description_es.strip():
        return None

    hint_line = f"Título en inglés sugerido: {name_en_hint}\n" if name_en_hint else ""
    user_text = (
        f"{hint_line}"
        f"Título en español: {name_es or '(sin título)'}\n"
        f"Descripción en español: {description_es}\n\n"
        'Devuelve JSON: {"name_en": "...", "description_en": "..."}'
    )
    messages = [{"role": "user", "content": [{"text": user_text}]}]

    try:
        data = bedrock_client.converse_json(
            config.NOVA_2_LITE_MODEL_ID,
            _TRANSLATE_CARD_SYSTEM,
            messages,
            schema=llm_schemas.TRANSLATION_CARD,
            tool_name="translate_menu_card",
            tool_description="Translate the Spanish menu card to English",
            inference_config={"maxTokens": 512, "temperature": 0.0},
            stage="translation_retry",
        )
    except Exception as e:
        logger.warning("translation_retry_error", extra={"error": str(e)})
        return None

    if not isinstance(data, dict):
        return None

    name_en = str(data.get("name_en", "")).strip()
    description_en = str(data.get("description_en", "")).strip()
    if not description_en or looks_spanish(description_en):
        logger.warning(
            "translation_retry_still_spanish",
            extra={"description_en": description_en[:200]},
        )
        return None

    return {
        "name_en": name_en or name_en_hint,
        "description_en": description_en,
    }
