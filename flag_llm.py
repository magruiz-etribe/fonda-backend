from __future__ import annotations

import logging
from typing import Final

import bedrock_client
import config
from flags import _allergens, _vegetarian_markers, _spicy_markers

logger = logging.getLogger(__name__)

_FLAGS_MAX_TOKENS: Final[int] = 600

_FALLBACK: Final[dict] = {
    "allergens": False,
    "allergen_triggers": [],
    "gluten_free": True,
    "gluten_triggers": [],
    "vegetarian": True,
    "vegan": True,
    "spicy_level": "none",
    "spicy_triggers": [],
}

_SYSTEM_PROMPT: str | None = None


def _get_system_prompt() -> str:
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = _build_system_prompt()
    return _SYSTEM_PROMPT


def _build_system_prompt() -> str:
    allergens = _allergens()
    vegetarian = _vegetarian_markers()
    spicy = _spicy_markers()

    allergen_lines: list[str] = []
    gluten_triggers: list[str] = []
    for group_name, group in allergens.get("groups", {}).items():
        triggers = [t.replace("_", " ") for t in group.get("triggers", [])]
        if group_name == "gluten":
            gluten_triggers = triggers
        else:
            label = group.get("label_es", group_name)
            allergen_lines.append(f"  {label}: {', '.join(triggers)}")

    meat   = [t.replace("_", " ") for t in vegetarian.get("meat_proteins", [])]
    seafood = [t.replace("_", " ") for t in vegetarian.get("seafood", [])]
    animal  = [t.replace("_", " ") for t in vegetarian.get("animal_products", [])]

    spicy_levels = spicy.get("levels", {})
    mild   = [t.replace("_", " ") for t in spicy_levels.get("mild", [])]
    medium = [t.replace("_", " ") for t in spicy_levels.get("medium", [])]
    hot    = [t.replace("_", " ") for t in spicy_levels.get("hot", [])]

    allergen_section = "\n".join(allergen_lines) or "(vacío)"

    return (
        "Eres un analizador de ingredientes para fondas mexicanas de CDMX. "
        "Determina las banderas dietéticas de un platillo dados su nombre, variante, "
        "ingredientes del KB y contexto de la conversación.\n"
        "Devuelves ÚNICAMENTE un objeto JSON válido, sin markdown, sin texto extra.\n\n"
        "## Listas de referencia\n\n"
        f"### Alérgenos — activan allergens=true\n{allergen_section}\n\n"
        f"### Con gluten — activan gluten_free=false\n{', '.join(gluten_triggers)}\n\n"
        f"### No vegetariano (carnes/aves) — activan vegetarian=false y vegan=false\n{', '.join(meat)}\n\n"
        f"### No vegetariano (mariscos/pescado) — activan vegetarian=false y vegan=false\n{', '.join(seafood)}\n\n"
        f"### No vegano (prod. animales sin carne) — activan solo vegan=false\n{', '.join(animal)}\n\n"
        f"### Picante\n- mild: {', '.join(mild)}\n- medium: {', '.join(medium)}\n- hot: {', '.join(hot)}\n\n"
        "## Reglas de análisis\n"
        "1. Analiza TODOS los componentes: nombre del platillo, variante, ingredientes del KB, "
        "extras del usuario y la conversación (incluyendo la descripción generada si ya existe).\n"
        "2. Si un ingrediente de las listas aparece en cualquier fuente → activa su bandera.\n"
        "2a. El nombre del platillo principal y de la variante en `Platillos en contexto` siempre "
        "cuentan como ingredientes confirmados. Ejemplos: platillo='huevo' → 'huevo' está presente; "
        "platillo='milanesa' → 'carne' y 'pan molido' están presentes; variante='con_jamon' → "
        "'jamón' está presente. Evalúalos contra las listas igual que cualquier ingrediente.\n"
        "3. Si el usuario negó explícitamente un ingrediente (\"no lleva X\", \"sin X\") → "
        "tómalo como ausente (prioridad del usuario sobre el KB).\n"
        "4. spicy_level: usa el nivel más alto detectado entre todos los ingredientes picantes.\n"
        "5. Los triggers deben ser los ingredientes concretos encontrados, no categorías.\n"
        "6. Si no hay información suficiente para confirmar un ingrediente → asume ausente. "
        "Esta regla NO aplica cuando el ingrediente está implícito en el nombre del platillo "
        "(regla 2a).\n\n"
        '## Formato de salida\n'
        '{"reasoning": "<qué ingredientes encontraste y en qué fuente, qué banderas activan>", '
        '"allergens": true/false, "allergen_triggers": ["ingrediente"], '
        '"gluten_free": true/false, "gluten_triggers": ["ingrediente"], '
        '"vegetarian": true/false, "vegan": true/false, '
        '"spicy_level": "none|mild|medium|hot", "spicy_triggers": ["ingrediente"]}'
    )


def compute_flags_for_dish(
    current_dish: str,
    companions: list[str],
    collected_ingredients: list[str],
    kb_ingredients_per_dish: dict[str, list[str]],
) -> dict:
    """Convenience wrapper for the new state-machine context shape."""
    dishes = [d for d in [current_dish] + companions if d]
    kb_lines = [
        f"{dish}: {', '.join(ingr)}"
        for dish, ingr in kb_ingredients_per_dish.items()
        if ingr
    ]
    return compute_flags_llm(
        current_dishes=dishes,
        resolved_variants={},
        extra_user_ingredients=list(collected_ingredients),
        conversation="",
        kb_context="\n".join(kb_lines) or "(sin KB)",
    )


def compute_flags_llm(
    current_dishes: list[str],
    resolved_variants: dict[str, str],
    extra_user_ingredients: list[str],
    conversation: str,
    kb_context: str,
) -> dict:
    """Compute dietary flags using LLM analysis of all available dish context."""
    system = _get_system_prompt()
    user_text = _build_user_text(
        current_dishes, resolved_variants, extra_user_ingredients, conversation, kb_context
    )
    messages = [{"role": "user", "content": [{"text": user_text}]}]

    try:
        raw = bedrock_client.converse(
            config.NOVA_2_LITE_MODEL_ID,
            system,
            messages,
            inference_config={"maxTokens": _FLAGS_MAX_TOKENS, "temperature": 0.0},
            stage="flags",
        )
    except bedrock_client.BedrockError as e:
        logger.warning("flag_llm_bedrock_error", extra={"error": str(e)})
        return dict(_FALLBACK)

    return _parse(raw)


def _build_user_text(
    current_dishes: list[str],
    resolved_variants: dict[str, str],
    extra_user_ingredients: list[str],
    conversation: str,
    kb_context: str,
) -> str:
    dish_parts: list[str] = []
    for dish in current_dishes:
        variant = resolved_variants.get(dish)
        dish_parts.append(f"{dish} — variante: {variant}" if variant else dish)
    dishes_str = "\n".join(f"  - {d}" for d in dish_parts) or "  (sin platillo)"
    extras_str = ", ".join(extra_user_ingredients) or "(ninguno)"

    # Use the most recent 800 chars of conversation (captures generated description if any)
    conv_trimmed = conversation[-800:] if len(conversation) > 800 else conversation
    kb_trimmed = kb_context[:1000] if len(kb_context) > 1000 else kb_context

    return (
        f"Platillos en contexto:\n{dishes_str}\n\n"
        f"Ingredientes adicionales mencionados por el usuario: {extras_str}\n\n"
        f"Ingredientes del KB para estos platillos:\n{kb_trimmed}\n\n"
        f"Conversación reciente (incluye descripción generada si ya existe):\n{conv_trimmed}\n\n"
        "Devuelve únicamente el JSON."
    )


def _parse(raw: str) -> dict:
    try:
        data = bedrock_client.parse_json_strict(raw)
    except Exception as e:
        logger.warning("flag_llm_parse_error", extra={"error": str(e), "raw": raw[:200]})
        return dict(_FALLBACK)

    if not isinstance(data, dict):
        return dict(_FALLBACK)

    def _bool(key: str, default: bool) -> bool:
        val = data.get(key)
        return bool(val) if isinstance(val, bool) else default

    def _strlist(key: str) -> list[str]:
        val = data.get(key)
        if not isinstance(val, list):
            return []
        return [str(i).strip().lower() for i in val if str(i).strip()]

    spicy = str(data.get("spicy_level", "none")).lower()
    if spicy not in ("none", "mild", "medium", "hot"):
        spicy = "none"

    result = {
        "allergens": _bool("allergens", False),
        "allergen_triggers": _strlist("allergen_triggers"),
        "gluten_free": _bool("gluten_free", True),
        "gluten_triggers": _strlist("gluten_triggers"),
        "vegetarian": _bool("vegetarian", True),
        "vegan": _bool("vegan", True),
        "spicy_level": spicy,
        "spicy_triggers": _strlist("spicy_triggers"),
    }

    logger.info(
        "flag_llm_ok",
        extra={
            "vegetarian": result["vegetarian"],
            "vegan": result["vegan"],
            "allergens": result["allergens"],
            "gluten_free": result["gluten_free"],
            "spicy_level": result["spicy_level"],
            "reasoning": str(data.get("reasoning", ""))[:300],
        },
    )
    return result
