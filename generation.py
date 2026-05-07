from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Final

import bedrock_client
import config
from prompt_loader import load_prompt
from state import CUSTOM_ENTITY, AgentState

logger = logging.getLogger(__name__)

_GENERATION_PROMPT_PATH: Final[str] = "generation_system.txt"


def _system_base() -> str:
    return load_prompt(_GENERATION_PROMPT_PATH)


@dataclass
class GenInput:
    state: AgentState
    intent: str
    message: str
    kb_context: str
    history: list[dict[str, str]]
    correction_note: str | None = None


_META_RE = re.compile(
    r"<META\b[^>]*>\s*(.+?)\s*</META\s*>",
    re.DOTALL | re.IGNORECASE,
)


def generate(gi: GenInput) -> str:
    system = _system_base() + "\n\n" + _intent_directives(gi.intent, gi.state)
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
    """Quita todos los bloques <META>...</META> del texto y devuelve el último
    JSON válido con final_translation:true (si existe). Cualquier otro META
    (schemas inventados) se descarta tras log."""
    if not reply:
        return reply, None
    chosen: dict[str, Any] | None = None
    last_invalid_idx: int | None = None
    for m in _META_RE.finditer(reply):
        blob = m.group(1).strip()
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError as e:
            logger.warning(
                "meta_block_invalid_json",
                extra={"error": str(e), "snippet": blob[:120]},
            )
            last_invalid_idx = m.start()
            continue
        if not isinstance(parsed, dict):
            last_invalid_idx = m.start()
            continue
        if parsed.get("final_translation") is True:
            chosen = parsed
        else:
            logger.info(
                "meta_block_ignored_non_final",
                extra={"sample_keys": list(parsed.keys())[:8]},
            )
    visible = _META_RE.sub("", reply).strip()
    meta = chosen
    if meta is None and last_invalid_idx is not None:
        logger.warning(
            "meta_only_invalid_or_non_final_reply",
            extra={"reply_tail": reply[max(0, last_invalid_idx - 40) :][:200]},
        )
    return visible, meta


def infer_final_translation_meta(visible: str) -> dict[str, Any] | None:
    """Si Nova mostró las tres líneas de TRADUCCIÓN FINAL pero olvidó <META>,
    reconstruye el dict para registrar estado y cerrar ciclo conforme."""
    if not visible or "**Nombre EN:**" not in visible:
        return None
    if "**Descripción EN:**" not in visible or "**Allergens:**" not in visible:
        return None
    tn = _extract_between_labels(
        visible,
        r"\*\*Nombre\s*EN:\*\*",
        r"\*\*Descripción\s*EN:\*\*",
    )
    td = _extract_between_labels(
        visible,
        r"\*\*Descripción\s*EN:\*\*",
        r"\*\*Allergens:\*\*",
    )
    ta = _extract_between_labels(
        visible,
        r"\*\*Allergens:\*\*",
        None,
    )
    if not tn or not td or ta is None:
        return None
    tn_f = _one_line(tn)
    td_f = " ".join(td.split())
    ta_f = _one_line(ta)
    allergens = _parse_allergen_tokens(ta_f)
    return {
        "final_translation": True,
        "translation_en": tn_f,
        "description_en": td_f,
        "allergens": allergens,
    }


def _extract_between_labels(
    text: str,
    start_pat: str,
    end_pat: str | None,
) -> str | None:
    ms = re.search(start_pat, text, flags=re.IGNORECASE | re.DOTALL)
    if not ms:
        return None
    tail = text[ms.end() :]
    if end_pat is None:
        chunk = re.split(r"\n\s*(?:¿|\n)", tail, maxsplit=1)[0]
        return chunk.strip() if chunk.strip() else None
    me = re.search(end_pat, tail, flags=re.IGNORECASE | re.DOTALL)
    if not me:
        return None
    return tail[: me.start()].strip()


def _one_line(s: str) -> str:
    return " ".join(s.replace("\r", " ").split()).strip()


def _parse_allergen_tokens(line: str) -> list[str]:
    low = line.strip().lower()
    if not low or low in ("none", "none.", "ninguno", "ninguna"):
        return []
    parts = re.split(r"[,;]", line)
    out: list[str] = []
    for p in parts:
        t = p.strip().lower()
        if t and t not in out:
            out.append(t)
    return out


def sanitize_traducir_visible_for_user(visible: str) -> str:
    """Quita restos de META y negritas de estilo (**palabra**) durante entrevista
    (cuando NO hay bloque TRADUCCIÓN FINAL con etiquetas estándar)."""
    if not visible:
        return visible
    v = _META_RE.sub("", visible).strip()
    v = re.sub(r"<META\b[^>]*>[\s\S]*?</META\s*>", "", v, flags=re.IGNORECASE).strip()
    if "**Nombre EN:**" not in v:
        v = re.sub(r"\*\*([^*]+)\*\*", r"\1", v)
    return v.replace("`", "").strip()


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

    lc = state.completed[-1] if state.completed else None
    pending_approval = bool(
        lc
        and lc.approved is None
        and (lc.translation_en or "").strip()
    )
    if pending_approval:
        return (
            "ESTADO: el cliente ya recibió las tres líneas de TRADUCCIÓN FINAL (inglés "
            "+ **Allergens:** + META).\n"
            "EN ESTE TURNO NO vuelvas a pegar ese bloque completo NI otro META final salvo "
            "que pida un CAMBIO concreto al texto en inglés (entonces sí: bloque corregido + "
            "META de final_translation).\n"
            "Si solo asiente, agradece o cierra (sí, va, listo, perfecto, gracias, quedó, etc.): "
            "UNA frase corta en español, tono humano, sin trámite; ej. '¡Listo! ¿Cuál otro "
            "platillo le armamos en inglés?' — no hagas nueva pregunta rígida de conformación.\n"
            "Si en el mismo mensaje ya viene OTRO platillo nuevo, celebra breve y entra "
            "derecho al flujo de ese platillo (una pregunta útil máximo), sin exigir que "
            "\"aprueben\" antes el anterior por escrito.\n"
            "Si rechaza algo, pregunta qué cambiar antes de repetir todo el inglés.\n"
            "Sin entrevistar de nuevo para alérgenos."
        )

    if cd is None:
        just_approved = bool(
            state.completed and state.completed[-1].approved is True
        )
        if just_approved:
            return (
                "Acaba de quedar atrás una traducción (aprobación o siguiente paso). "
                "PROHIBIDO reimprimir el bloque inglés anterior.\n"
                "Si ya nombró otro platillo, entra derecho sin repreguntas vacías "
                "(máximo UNA aclaración concreta). Ej.: 'Órale, ¿tus enchiladas qué llevan "
                "de protagonista?'.\n"
                "Si no nombró otro platillo, saluda breve al siguiente pedido "
                "(una sola línea tipo '¿Cuál platillo necesitas en inglés hoy?', "
                "no hace falta copiar verbatim una plantilla)."
            )
        return (
            "Sin platillo cargado en sistema aún. Pregunta con naturalidad qué platillo "
            "quiere en el menú en inglés (una sola línea corta y amable — no hagas checklist "
            "de ingredientes en este mismo turno)."
        )

    is_custom = cd.entity == CUSTOM_ENTITY
    has_variant = bool(cd.variant)
    has_ingredients = bool(cd.user_ingredients)
    ing_count = len(cd.user_ingredients)

    _coherent = (
        "\n\nCOHERENCIA: lee «Últimos turnos». No hagas segunda vuelta de lo mismo: si algo "
        "ya quedó claro inclúyelo en tu cabeza y avanza—si ya alcanza para traducción lista, "
        "no inventes preguntitas extra sólo porque el flujo típico pide tres pasos. "
        "Alérgenos: no menciones al cliente; sólo aparecen discretamente en formato final "
        "(si incierto→none)."
    )

    if is_custom:
        if not has_ingredients:
            return (
                "Modo PERSONALIZADO (platillo no está en el KB).\n"
                "Pregunta SOLO UNA cosa: cuál es el ingrediente principal VISIBLE "
                "del platillo (la proteína o el componente que más se ve al comer). "
                "NO sugieras ni preguntes por sal, aceite, ajo, cebolla, agua o "
                "pimienta: se asumen."
            ) + _coherent
        return (
            "Modo PERSONALIZADO con ingredientes capturados.\n"
            "Pregunta UN solo ingrediente VISIBLE más (¿le agregas alguna verdura? "
            "¿chile? ¿hierba? ¿queso?). Prohibido preguntar por sal/aceite/ajo/"
            "cebolla/agua. Cuando el usuario diga 'es todo' / 'ya está' / "
            "'tradúcelo', genera TRADUCCIÓN FINAL: exactamente tres líneas "
            "**Nombre EN:** / **Descripción EN:** / **Allergens:** en inglés, "
            "etiquetas en **negrita**, + bloque <META> en una línea al final."
        ) + _coherent

    if not has_variant:
        return (
            "Tienes el platillo identificado pero no la variante.\n"
            "Pregunta en UNA sola línea conversacional qué variante prepara, "
            "sin viñetas ni descripciones del KB. Como mucho, mete UN ejemplo "
            "entre paréntesis. Ej: '¿De qué tipo lo haces (rojo, verde, "
            "poblano)?'. Si el KB no tiene variantes, salta a preguntar un "
            "ingrediente VISIBLE en el mismo formato corto."
        ) + _coherent

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
        ) + _coherent

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
        ) + _coherent

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
    ) + _coherent
