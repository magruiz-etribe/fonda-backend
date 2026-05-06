from __future__ import annotations

import logging
from typing import Final

import bedrock_client
import config
from retrieval import get_entities_index
from state import CUSTOM_ENTITY

logger = logging.getLogger(__name__)


_SYSTEM: Final[str] = (
    "Eres un mapeador de platillos mexicanos a entidades canónicas de un knowledge base. "
    "Recibes un mensaje del usuario y un índice {alias_o_nombre -> entidad_canonica}. "
    "Tu tarea: identificar UNO o MÁS platillos mencionados y devolverlos como entidades "
    "canónicas. Para platillos compuestos como 'mole con arroz' devuelve ambas entidades "
    "por separado. "
    "Devuelves SOLO un JSON con esta forma exacta: "
    '{"entities": ["<canonica1>", "<canonica2>", ...]}. '
    "Sin markdown, sin texto extra. "
    f'Si ninguna coincide claramente, devuelve {{"entities": ["{CUSTOM_ENTITY}"]}}.'
)


def map_dishes(message: str, image_dishes: list[str] | None = None) -> list[str]:
    index = get_entities_index()
    if not index:
        return [CUSTOM_ENTITY]

    canonicals = set(index.values())
    aliases_block = "\n".join(f"- {alias} -> {canonical}" for alias, canonical in index.items())

    image_block = ""
    if image_dishes:
        joined = ", ".join(image_dishes)
        image_block = f"Platillos detectados en imagen: {joined}\n\n"

    user_text = (
        f"Índice de entidades canónicas:\n{aliases_block}\n\n"
        f"{image_block}"
        f"Mensaje del usuario:\n\"{message}\"\n\n"
        "Responde solo con JSON."
    )
    messages = [{"role": "user", "content": [{"text": user_text}]}]

    try:
        raw = bedrock_client.converse(
            config.NOVA_MICRO_MODEL_ID,
            _SYSTEM,
            messages,
            inference_config={
                "maxTokens": config.ENTITY_MAPPER_MAX_TOKENS,
                "temperature": 0.0,
            },
        )
    except bedrock_client.BedrockError as e:
        logger.warning("entity_mapper_bedrock_error", extra={"error": str(e)})
        return [CUSTOM_ENTITY]

    return _parse(raw, canonicals)


def _parse(raw: str, canonicals: set[str]) -> list[str]:
    try:
        data = bedrock_client.parse_json_strict(raw)
    except Exception as e:
        logger.warning("entity_mapper_parse_error", extra={"error": str(e), "raw": raw[:200]})
        return [CUSTOM_ENTITY]

    if not isinstance(data, dict):
        return [CUSTOM_ENTITY]

    entities = data.get("entities")
    if not isinstance(entities, list):
        return [CUSTOM_ENTITY]

    cleaned: list[str] = []
    for e in entities:
        if not isinstance(e, str) or not e:
            continue
        if e == CUSTOM_ENTITY or e in canonicals:
            cleaned.append(e)

    return cleaned or [CUSTOM_ENTITY]
