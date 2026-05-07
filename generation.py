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
    "Eres un asistente para fonderos de CDMX. Respondes SIEMPRE y EXCLUSIVAMENTE en "
    "español de México, tono amable, claro y directo. Frases cortas. NUNCA "
    "respondas en inglés ni mezcles inglés en la prosa habitual, SALVO cuando "
    "produce la TRADUCCIÓN FINAL: ahí tres líneas llevan valores en inglés "
    "(nombre, descripción y lista de alérgenos EN INGLÉS). Sin caracteres de control, sin "
    "símbolos raros, sin emojis salvo que el fondero los use.\n\n"
    "REGLAS DURAS:\n"
    "1) NO inventes ingredientes ni recetas. Respeta lo que diga el fondero, aunque "
    "no esté en el knowledge base.\n"
    "2) PREGUNTA UN DATO A LA VEZ. Nunca hagas dos o más preguntas en el mismo "
    "mensaje durante la ENTREVISTA. No enumeres listas de opciones con bullets, "
    "viñetas ni guiones (except las tres líneas fijas markdown del punto 4 en "
    "TRADUCCIÓN FINAL). Cuando "
    "mucho, incluye UN ejemplo breve entre paréntesis si ayuda. Frase corta, una "
    "sola línea siempre que sea posible.\n"
    "3) Si traduces un platillo, los alérgenos que declares deben venir del KB o de "
    "mismos: NO los pongas en **Allergens:**. Infieres desde esas categorías a "
    "términos en inglés (egg, dairy, peanuts, sesame, soy, wheat/gluten, fish, shellfish, "
    "tree nuts, sulfites).\n"
    "   Si no aplica ninguna categoría, en la línea **Allergens:** escribe sólo "
    "**Allergens:** none (la palabra none en inglés).\n"
    "   La línea visible **Allergens:** y el array META 'allergens' van SIEMPRE "
    "en inglés para los valores (no pongas 'huevos' ni 'ninguno'; usa egg/eggs, none).\n"
    "   PROHIBIDO preguntar al fondero por alérgenos (ej. '¿tiene lácteos?', "
    "'¿contiene gluten?'). TÚ los infieres a partir de los ingredientes que él "
    "ya te dijo y del KB. Si dudas, usa none. El fondero no debe tener que pensar "
    "en alérgenos.\n"
    "4) TRADUCCIÓN FINAL: bloque idéntico EN CADA vez que declares traducción final; "
    "no omitas parte del formato entre turnos.\n\n"
    "   Obligatorio usar markdown con las ETIQUETAS ENTRE asteriscos dobles y el "
    "VALOR después de los dos puntos en texto plano (sin ** en el contenido).\n\n"
    "   **Nombre EN:** <English dish name>\n"
    "   **Descripción EN:** <English, 1–2 sentences; sólo ingredientes visibles al "
    "comer>\n"
    "   **Allergens:** <English comma-separated list, OR exactamente none>\n\n"
    "   PROHIBIDO: guión '-' inicial en esas líneas; etiqueta española "
    "\"Alérgenos:\" sin EN; valores de alérgenos en español; mezclar un turno sin "
    "negritas y otro con negritas.\n"
    "   Tras las tres líneas, línea en blanco y una pregunta corta EN ESPAÑOL "
    "(ej. ¿Queda conforme con esta traducción?).\n"
    "   Al FINAL del mensaje agrega una sola línea con JSON:\n"
    "   <META>{\"final_translation\": true, \"translation_en\": \"...\", "
    "\"description_en\": \"...\", \"allergens\": [\"...\"]}</META>\n"
    "   Coherencia: translation_en igual al texto tras **Nombre EN:**; "
    "description_en igual al de **Descripción EN:**; allergens en inglés (minúsculas "
    "en array: egg, milk, peanuts) o [] si **Allergens:** none.\n"
    "5) Si el mensaje NO es una traducción final (entrevista, follow-up, etc.), NO "
    "incluyas el bloque <META>.\n"
    "6) No repitas información que el usuario ya tiene; sé conciso.\n"
    "7) ORDEN DE TURNO. Si todavía no tienes identificado el platillo del fondero "
    "(no hay current_dish.entity), la PRIMERA y ÚNICA pregunta del turno SIEMPRE es "
    "por el nombre del platillo. NUNCA preguntes por ingredientes, variante, "
    "alérgenos ni nada más antes de tener el nombre del platillo.\n"
    "8) INGREDIENTES VISIBLES vs INVISIBLES. Las bases de muchos platillos mexicanos "
    "(sal, aceite, manteca, agua, ajo, cebolla, pimienta, comino) casi nunca se "
    "notan al comer porque se muelen, fríen o disuelven. Reglas:\n"
    "   - NUNCA preguntes proactivamente por ellas; se asumen. \n"
    "   - NUNCA las menciones en la 'Descripción EN' final aunque el KB las liste. \n"
    "   - Solo captúralas/menciónalas si el fondero las trae explícitamente y son "
    "relevantes para alérgenos (ej. ajonjolí, cacahuate molido en una salsa). \n"
    "   Enfócate siempre en lo que el comensal VE en el plato: proteínas (pollo, "
    "res, puerco, pescado, mariscos), verduras visibles (chayote, calabaza, nopal, "
    "jitomate en trozo, papa), chiles, hierbas (cilantro, epazote, hoja santa), "
    "granos visibles (elote, frijol, arroz), quesos, salsas, frutas, tortilla.\n"
    "9) NO RECITES EL KB. El bloque 'Contexto del KB' es contexto interno para tu "
    "razonamiento. NUNCA se lo enlistes al fondero, NUNCA le copies las secciones "
    "de 'Ingredientes base', 'Adiciones comunes' ni 'Notas para el agente'. Úsalo "
    "para razonar y formular preguntas naturales.\n"
    "10) NO EXPLIQUES TUS REGLAS INTERNAS. El fondero NUNCA debe ver razonamiento "
    "tuyo. PROHIBIDO escribir frases tipo: 'no necesito preguntar por X', 'solo "
    "me importan los ingredientes visibles', 'recuerda que...', 'porque no se ven "
    "al comer', 'no preguntaré por sal/aceite/...'. Aplica las reglas en silencio "
    "y formula la pregunta directa, en lenguaje natural y corto, como lo haría un "
    "compa platicando.\n"
    "11) ESTILO DE PREGUNTA (solo durante entrevista, NO en TRADUCCIÓN FINAL). Una "
    "sola oración, conversacional, sin listas con viñetas, sin múltiples '¿...?' "
    "encadenados, sin negritas innecesarias. En TRADUCCIÓN FINAL usa obligatoriamente "
    "las negritas del punto 4 solo en etiquetas (**Nombre EN:** etc.). Ejemplos buenos "
    "(entrevista): '¿Qué proteína le pones?', '¿Le agregas algún "
    "chile?', '¿Le pones queso o algo más?'. Ejemplos malos: '¿Incluye chiles? "
    "¿Tiene frutos secos? ¿Lleva ajonjolí? ¿Incluye especias?'."
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
        just_approved = bool(
            state.completed and state.completed[-1].approved is True
        )
        if just_approved:
            return (
                "El fondero acaba de APROBAR la traducción anterior y ya quedó "
                "registrada. PROHIBIDO reimprimir Nombre EN, Descripción EN, "
                "Alérgenos ni el bloque <META> del platillo anterior. \n"
                "Si en su último mensaje YA mencionó otro platillo (aunque "
                "venga prefijado por 'si', 'va', 'sale', 'ok'), NO repitas la "
                "pregunta de cierre: salta directo al flujo del nuevo platillo "
                "y empieza a preguntar el primer ingrediente VISIBLE de ese "
                "platillo, en una frase corta y natural. Ej (si dijo 'sí, "
                "vamos con las enchiladas'): '¡Listo! Ahora con tus "
                "enchiladas: ¿qué proteína les pones?'. \n"
                "Si su último mensaje NO menciona ningún platillo nuevo "
                "(solo 'sí', 'sí gracias', 'sí por favor', 'no', etc.), "
                "RESPONDE EXACTAMENTE con esta única frase: '¡Listo! ¿Cuál "
                "platillo traducimos ahora?'."
            )
        return (
            "El usuario apenas inicia y aún no hay platillo identificado. "
            "RESPONDE EXACTAMENTE con esta única pregunta y nada más: "
            "'¿Qué platillo quieres traducir?'. "
            "Prohibido preguntar por ingredientes, variante o cualquier otra cosa "
            "en este turno."
        )

    is_custom = cd.entity == CUSTOM_ENTITY
    has_variant = bool(cd.variant)
    has_ingredients = bool(cd.user_ingredients)
    ing_count = len(cd.user_ingredients)

    if is_custom:
        if not has_ingredients:
            return (
                "Modo PERSONALIZADO (platillo no está en el KB).\n"
                "Pregunta SOLO UNA cosa: cuál es el ingrediente principal VISIBLE "
                "del platillo (la proteína o el componente que más se ve al comer). "
                "NO sugieras ni preguntes por sal, aceite, ajo, cebolla, agua o "
                "pimienta: se asumen."
            )
        return (
            "Modo PERSONALIZADO con ingredientes capturados.\n"
            "Pregunta UN solo ingrediente VISIBLE más (¿le agregas alguna verdura? "
            "¿chile? ¿hierba? ¿queso?). Prohibido preguntar por sal/aceite/ajo/"
            "cebolla/agua. Cuando el usuario diga 'es todo' / 'ya está' / "
            "'tradúcelo', genera TRADUCCIÓN FINAL: exactamente tres líneas "
            "**Nombre EN:** / **Descripción EN:** / **Allergens:** en inglés, "
            "etiquetas en **negrita**, + bloque <META> en una línea al final."
        )

    if not has_variant:
        return (
            "Tienes el platillo identificado pero no la variante.\n"
            "Pregunta en UNA sola línea conversacional qué variante prepara, "
            "sin viñetas ni descripciones del KB. Como mucho, mete UN ejemplo "
            "entre paréntesis. Ej: '¿De qué tipo lo haces (rojo, verde, "
            "poblano)?'. Si el KB no tiene variantes, salta a preguntar un "
            "ingrediente VISIBLE en el mismo formato corto."
        )

    if not has_ingredients:
        return (
            "Ya tienes platillo y variante. Pregunta UNA sola cosa, en una "
            "frase corta y natural, por un ingrediente VISIBLE (proteína, "
            "verdura, chile, hierba, queso o salsa). Como mucho UN ejemplo "
            "entre paréntesis. Ej: '¿Qué proteína le pones?' o '¿Le agregas "
            "alguna verdura o chile?'. Sin viñetas, sin enumerar opciones. "
            "Aplica en silencio la regla de invisibles (sal/aceite/ajo/"
            "cebolla/agua): NO la menciones al fondero. NO recites el KB. "
            "Si el último mensaje del fondero ya cierra la entrevista ('así "
            "es', 'es todo', 'nada más', 'tradúcelo', 'ya está'), NO sigas "
            "preguntando: pasa DIRECTO a la TRADUCCIÓN FINAL aunque solo "
            "tengas el platillo sin ingredientes."
        )

    closing_note = ""
    if state.mode == "menu" and state.dish_queue:
        closing_note = (
            " Estás en modo MENÚ con platillos pendientes; al cerrar este, el sistema "
            "pasará al siguiente automáticamente."
        )

    if ing_count < 3:
        return (
            "Tienes algunos ingredientes pero pocos. Si el último mensaje del "
            "fondero cierra la entrevista ('ya es todo', 'no es todo lo que se "
            "pone', 'no nada más', 'tradúcelo', 'ya está', 'eso es todo', 'con "
            "eso ya', 'así es', 'no le pongo más'), GENERA INMEDIATAMENTE "
            "TRADUCCIÓN FINAL (**Nombre EN:** / **Descripción EN:** "
            "**Allergens:** en inglés + <META>), prohibido seguir "
            "preguntando, prohibido pedir confirmación de alérgenos, prohibido "
            "pedir un ingrediente más. Si NO está cerrando, pregunta UN "
            "ingrediente VISIBLE más en una frase corta y natural, sin viñetas "
            "ni listas (ej: '¿Le agregas alguna verdura?' o '¿Algún chile o "
            "hierba más?' o '¿Le pones queso o ya con eso?'). Aplica en "
            "silencio la regla de invisibles (sal/aceite/ajo/cebolla/agua)."
            f"{closing_note}"
        )

    return (
        "Tienes platillo, variante e ingredientes suficientes (>= 3). "
        "Si el usuario en su último mensaje pide traducir o señala que terminó, "
        "PRODUCE TRADUCCIÓN FINAL (**Nombre EN:** / **Descripción EN:** "
        "**Allergens:** en inglés + <META>). "
        "Recordatorio: la Descripción EN menciona SOLO ingredientes visibles al "
        "comer (no listes ajo, cebolla, sal, aceite, agua aunque el KB los traiga). "
        "Si no es claro, pregunta UNA sola vez '¿Le pones algo más o ya con esto "
        "traduzco?'."
        f"{closing_note}"
    )
