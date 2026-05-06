from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import bedrock_client
import config
from state import AgentState, CUSTOM_ENTITY

logger = logging.getLogger(__name__)


_SYSTEM_BASE: Final[str] = (
    "Eres un asistente para fonderos de CDMX. Respondes SIEMPRE en español de México, "
    "con tono amable, claro y directo. Frases cortas. "
    "REGLAS DURAS:\n"
    "1) NO inventes ingredientes ni recetas. Respeta lo que diga el fondero, aunque "
    "no esté en el knowledge base.\n"
    "2) Si traduces un platillo, incluye SIEMPRE: nombre en inglés, descripción "
    "breve en inglés y la lista de alérgenos detectados (incluyendo gluten cuando aplique).\n"
    "3) Los alérgenos que declares deben venir del contexto del KB o de los "
    "ingredientes que el usuario te dijo. No los inventes.\n"
    "4) Al cerrar una traducción, pregunta explícitamente al usuario si quedó "
    "conforme con el resultado.\n"
    "5) No repitas información si el usuario ya la tiene; sé conciso."
)


@dataclass
class GenInput:
    state: AgentState
    intent: str
    message: str
    kb_context: str
    history: list[dict[str, str]]
    correction_note: str | None = None


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


def _build_user_text(gi: GenInput) -> str:
    state = gi.state

    history_block = (
        "\n".join(f"- {h.get('role', '?')}: {h.get('text', '')}" for h in gi.history)
        or "(sin historial)"
    )
    state_block = _summarize_state(state)
    kb_block = gi.kb_context.strip() or "(sin contexto del KB; modo personalizado o tema sin ficha)"
    correction = (
        f"\n\n[INSTRUCCIÓN DE CORRECCIÓN]\n{gi.correction_note}\nReescribe respetando esta corrección."
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
        ings = ", ".join(cd.user_ingredients) if cd.user_ingredients else "no proporcionados aún"
        cd_str = f"entity={cd.entity}, ingredientes_del_fondero=[{ings}]"

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
    mode = state.mode

    if cd is None:
        return (
            "Estás iniciando una traducción. Identifica con el usuario qué platillo "
            "quiere traducir. Si la información es ambigua, haz UNA pregunta clara."
        )

    is_custom = cd.entity == CUSTOM_ENTITY
    no_ingredients = not cd.user_ingredients

    if is_custom and no_ingredients:
        return (
            "El platillo no está en el KB (modo personalizado). Entrevista brevemente "
            "al usuario: pídele el nombre exacto, los ingredientes principales y una "
            "descripción corta. Hazlo en UN solo turno con preguntas concretas."
        )

    if no_ingredients:
        return (
            "Tienes la entidad del KB pero faltan los ingredientes específicos del "
            "fondero. Pregúntale por sus ingredientes principales (no asumas la "
            "receta del KB; el fondero puede tener su propia receta)."
        )

    closing_note = ""
    if mode == "menu" and state.dish_queue:
        closing_note = (
            " Estás en modo MENÚ con platillos pendientes; al cerrar este, el sistema "
            "pasará automáticamente al siguiente. Avisa al usuario que continuamos con "
            "el siguiente platillo después de su confirmación."
        )

    return (
        "Tienes la entidad y los ingredientes del fondero. Genera la traducción "
        "FINAL del platillo: nombre en inglés, descripción breve en inglés (1-2 frases), "
        "y lista clara de alérgenos. Formato sugerido:\n"
        "- Nombre EN: ...\n- Descripción EN: ...\n- Alérgenos: a, b, c\n"
        "Cierra preguntando explícitamente si el resultado queda conforme."
        f"{closing_note}"
    )
