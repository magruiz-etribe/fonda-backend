from __future__ import annotations

import logging
from typing import Final

import bedrock_client
import config
from state import ImageAnalysis

logger = logging.getLogger(__name__)


_SYSTEM: Final[str] = (
    "Eres un analizador de imágenes para un agente que ayuda a fonderos de CDMX. "
    "Recibes UNA imagen. Determina si muestra: "
    "(a) un PLATILLO individual (un solo plato servido), o "
    "(b) un MENÚ impreso con varios platillos listados. "
    "Devuelves SOLO un JSON con esta forma exacta: "
    '{"type": "platillo"|"menu", "raw": "<descripción concisa en español>", '
    '"detected_dishes": ["<nombre1>", ...]}. '
    "Reglas: "
    "- Si type='platillo', detected_dishes tiene exactamente UN elemento (mejor estimación del nombre). "
    "- Si type='menu', detected_dishes lista los nombres de platillos extraídos del texto del menú. "
    "- raw debe ser una descripción breve (máx 2-3 líneas) de lo que ves. "
    "Sin markdown, sin texto extra fuera del JSON."
)

_PROMPT: Final[str] = "Analiza esta imagen y responde solo con el JSON especificado."

_RAW_MAX_LEN: Final[int] = 2000


class ImageAnalysisError(Exception):
    pass


def analyze(image_b64: str) -> ImageAnalysis:
    raw = bedrock_client.converse_with_image(
        config.NOVA_PRO_MODEL_ID,
        _SYSTEM,
        _PROMPT,
        image_b64,
        inference_config={
            "maxTokens": config.IMAGE_MAX_TOKENS,
            "temperature": 0.0,
        },
    )
    return _parse(raw)


def _parse(raw: str) -> ImageAnalysis:
    try:
        data = bedrock_client.parse_json_strict(raw)
    except Exception as e:
        raise ImageAnalysisError(f"failed to parse image analysis JSON: {e}") from e

    if not isinstance(data, dict):
        raise ImageAnalysisError("image analysis is not an object")

    img_type = data.get("type")
    if img_type not in ("platillo", "menu"):
        img_type = "platillo"

    dishes_raw = data.get("detected_dishes")
    dishes: list[str] = []
    if isinstance(dishes_raw, list):
        dishes = [str(d).strip() for d in dishes_raw if isinstance(d, (str, int, float)) and str(d).strip()]

    return ImageAnalysis(
        type=img_type,
        raw=str(data.get("raw", ""))[:_RAW_MAX_LEN],
        detected_dishes=dishes,
    )
