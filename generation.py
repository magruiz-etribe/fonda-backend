from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Final

import bedrock_client
import config
from state import CUSTOM_ENTITY, AgentState

logger = logging.getLogger(__name__)


_SYSTEM_BASE: Final[str] = (
    "Eres un asistente para fonderos de CDMX. Respondes SIEMPRE en español de México, "
    "tono amable, claro y directo. Frases cortas.\n\n"
    "REGLAS DURAS:\n"
    "1) NO inventes ingredientes ni recetas. Respeta lo que diga el fondero, aunque "
    "no esté en el knowledge base.\n"
    "2) PREGUNTA UN DATO A LA VEZ. Nunca hagas dos o más preguntas en el mismo "
    "mensaje. No saturas al fondero.\n"
    "3) Si traduces un platillo, los alérgenos que declares deben venir del KB o de "
    "los ingredientes que el fondero te dio. No los inventes.\n"
    "4) Cuando produzcas la TRADUCCIÓN FINAL del platillo, formato OBLIGATORIO:\n"
    "   - Nombre EN: <nombre>\n"
    "   - Descripción EN: <descripción 1-2 frases>\n"
    "   - Alérgenos: <lista separada por comas, o 'ninguno'>\n"
    "   Cierra preguntando explícitamente si quedó conforme.\n"
    "   Y al FINAL del mensaje agrega EXACTAMENTE este bloque (en una sola línea, "
    "JSON válido):\n"
    "   <META>{\"final_translation\": true, \"translation_en\": \"...\", "
    "\"description_en\": \"...\", \"allergens\": [\"...\"]}</META>\n"
    "5) Si el mensaje NO es una traducción final (entrevista, follow-up, etc.), NO "
    "incluyas el bloque <META>.\n"
    "6) No repitas información que el usuario ya tiene; sé conciso."
)


@dataclass
class GenInput:
    state: AgentState
    intent: str
    message: str
    kb_context: str
    history: list[dict[str, str]]
    correction_note: str | None = None


_META_RE = re.compile(r"<META>\s*(.+?)\s*</META>", re.DOTALL)


def generate(gi: GenInput) -> str:
    system = _SYSTEM_BASE + "\n\n" + _intent_directives(gi.intent, gi.state)
    user_text = _build_user_text(gi)

    messages = [{"role": "user", "content": [{"text": user_text}]}]
    return bedrock_client.converse(
        config.NOVA_LITE_MODEL_ID,
        system,
        messages,
        inference_config={
            "maxTokens": config.GEN_MAX_TOKENS,
            "temperature": 0.3,
        },
    )


def parse_and_strip_meta(reply: str) -> tuple[str, dict[str, Any] | None]:
    """Extrae bloque <META>{...}</META> del reply y lo separa de la parte
    visible. Devuelve (texto_visible, meta_dict_or_None)."""
    if not reply:
        return reply, None
    m = _META_RE.search(reply)
    if not m:
        return reply, None
    visible = _META_RE.sub("", reply).strip()
    try:
        parsed = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        logger.warning("meta_block_invalid_json", extra={"error": str(e)})
        return visible, None
    if not isinstance(parsed, dict):
        return visible, None
    return visible, parsed


def _build_user_text(gi: GenInput) -> str:
    state = gi.state
    history_block = (
        "\n".join(f"- {h.get('role', '?')}: {h.get('text', '')}" for h in gi.history)
        or "(sin historial)"
    )
    state_block = _summarize_state(state)
    kb_block = (
        gi.kb_context.strip()
        or "(sin contexto del KB; modo personalizado o tema sin ficha)"
    )
    correction = (
        f"\n\n[INSTRUCCIÓN DE CORRECCIÓN]\n{gi.correction_note}\n"
        "Reescribe respetando esta corrección."
        if gi.correction_note
        else ""
    )

    return (
        f"Intención actual: {gi.intent}\n\n"
        f"Estado:\n{state_block}\n\n"
        f"Contexto del KB:\n{kb_block}\n\n"
        f"Últimos turnos:\n{history_block}\n\n"
        f"Mensaje del usuario:\n\"{gi.message}\""
        f"{correction}"
    )


def _summarize_state(state: AgentState) -> str:
    cd = state.current_dish
    if cd is None:
        cd_str = "ninguno"
    else:
        ings = (
            ", ".join(cd.user_ingredients)
            if cd.user_ingredients
            else "no proporcionados aún"
        )
        cd_str = (
            f"entity={cd.entity}, variante={cd.variant or 'no definida'}, "
            f"ingredientes_del_fondero=[{ings}]"
        )

    completed_str = (
        ", ".join(c.dish for c in state.completed if c.dish) or "ninguno"
    )
    paused = state.paused_task.intent if state.paused_task else None
    queue_str = ", ".join(state.dish_queue) if state.dish_queue else "vacío"

    return (
        f"intent={state.intent} | mode={state.mode} | "
        f"current_dish=({cd_str}) | dish_queue=[{queue_str}] | "
        f"completed=[{completed_str}] | paused_task={paused}"
    )


def _intent_directives(intent: str, state: AgentState) -> str:
    if intent == "traducir":
        return _traducir_directives(state)
    if intent == "maps":
        return (
            "Estás dando ayuda sobre Google Maps. Da pasos accionables y concretos "
            "basados en el contexto del KB. Si el usuario hace una pregunta puntual, "
            "respóndela; no des un tutorial completo si no lo pidió."
        )
    if intent == "higiene":
        return (
            "Estás dando tips de higiene para una fonda. Da tips concretos y "
            "aplicables al día a día, alineados al contexto del KB."
        )
    return (
        "El usuario salió del alcance del agente (puede traducir platillos, ayudar "
        "con Google Maps o dar tips de higiene). Responde brevemente, con cortesía, "
        "y reencáuzalo a esos tres temas."
    )


def _traducir_directives(state: AgentState) -> str:
    cd = state.current_dish

    if cd is None:
        return (
            "El usuario apenas inicia. Pregunta UNA cosa: '¿Qué platillo quieres "
            "traducir?'. Sin más texto."
        )

    is_custom = cd.entity == CUSTOM_ENTITY
    has_variant = bool(cd.variant)
    has_ingredients = bool(cd.user_ingredients)
    ing_count = len(cd.user_ingredients)

    if is_custom:
        if not has_ingredients:
            return (
                "Modo PERSONALIZADO (platillo no está en el KB).\n"
                "Pregunta SOLO UNA cosa: el ingrediente principal del platillo. "
                "Cuando lo tengas en este turno (vía analysis), pasa a preguntar el "
                "siguiente en el próximo turno."
            )
        return (
            "Modo PERSONALIZADO con ingredientes capturados.\n"
            "Pregunta UN ingrediente más a la vez (¿algo más? ¿algún chile? ¿especias?). "
            "Cuando el usuario diga 'es todo' / 'ya está' / 'tradúcelo', genera la "
            "TRADUCCIÓN FINAL con el formato y bloque <META> obligatorios."
        )

    if not has_variant:
        return (
            "Tienes el platillo identificado pero no la variante.\n"
            "Si el KB lista variantes claras, pregunta ÚNICAMENTE qué variante prepara "
            "(una sola pregunta corta). Si el KB no tiene variantes, salta a preguntar "
            "el primer ingrediente."
        )

    if not has_ingredients:
        return (
            "Ya tienes platillo y variante. Pregunta SOLO UN ingrediente principal. "
            "No supongas la receta del KB; el fondero puede tener su propia receta."
        )

    closing_note = ""
    if state.mode == "menu" and state.dish_queue:
        closing_note = (
            " Estás en modo MENÚ con platillos pendientes; al cerrar este, el sistema "
            "pasará al siguiente automáticamente."
        )

    if ing_count < 3:
        return (
            "Tienes algunos ingredientes pero pocos. Pregunta UN ingrediente más a la "
            "vez (¿alguna especia? ¿caldo? ¿chile?). Cuando el usuario diga que ya es "
            "todo o tengas suficiente, pasa a la traducción final."
            f"{closing_note}"
        )

    return (
        "Tienes platillo, variante e ingredientes suficientes (>= 3). "
        "Si el usuario en su último mensaje pide traducir o señala que terminó, "
        "PRODUCE LA TRADUCCIÓN FINAL con formato obligatorio + bloque <META>. "
        "Si no es claro, pregunta UNA sola vez '¿Le pones algo más o ya con esto "
        "traduzco?'."
        f"{closing_note}"
    )
