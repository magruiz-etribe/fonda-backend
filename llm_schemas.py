"""JSON schemas for Bedrock structured outputs via Nova tool use."""
from __future__ import annotations

from typing import Any, Final

_STRING_ARRAY: Final[dict[str, Any]] = {
    "type": "array",
    "items": {"type": "string"},
}

CLASSIFIER_INTENT: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "Brief chain of thought in Spanish"},
        "intent": {"type": "string", "description": "Classified intent"},
        "platform": {
            "type": "string",
            "description": "google_maps, yelp, tripadvisor, or empty string",
        },
    },
    "required": ["reasoning", "intent", "platform"],
    "additionalProperties": False,
}

EXTRACTOR_TRADUCCION: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "Brief extraction reasoning"},
        "current_dish": {"type": "string", "description": "Canonical dish name or empty string"},
        "companions": {
            **_STRING_ARRAY,
            "description": "Side dishes served separately",
        },
        "custom_dish_known": {
            "type": "boolean",
            "description": "Whether a custom dish name is a real dish",
        },
    },
    "required": ["reasoning", "current_dish", "companions", "custom_dish_known"],
    "additionalProperties": False,
}

EXTRACTING: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "response": {
            **_STRING_ARRAY,
            "description": "One question bubble when variables are incomplete",
        },
        "variables_complete": {"type": "boolean"},
        "collected_ingredients": _STRING_ARRAY,
        "buttons": _STRING_ARRAY,
    },
    "required": ["response", "variables_complete", "collected_ingredients", "buttons"],
    "additionalProperties": False,
}

CONFIRMING_FLAGS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "response": _STRING_ARRAY,
        "buttons": _STRING_ARRAY,
    },
    "required": ["response", "buttons"],
    "additionalProperties": False,
}

DRAFTING: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "response": {
            **_STRING_ARRAY,
            "description": "Bilingual menu card bubble and confirmation bubble",
        },
        "buttons": _STRING_ARRAY,
        "current_dishes": _STRING_ARRAY,
    },
    "required": ["response", "buttons", "current_dishes"],
    "additionalProperties": False,
}

GENERATION: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "response": _STRING_ARRAY,
        "current_dishes": _STRING_ARRAY,
        "buttons": _STRING_ARRAY,
    },
    "required": ["response", "current_dishes", "buttons"],
    "additionalProperties": False,
}

FLAGS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "allergens": {"type": "boolean"},
        "allergen_triggers": _STRING_ARRAY,
        "gluten_free": {"type": "boolean"},
        "gluten_triggers": _STRING_ARRAY,
        "vegetarian": {"type": "boolean"},
        "vegan": {"type": "boolean"},
        "spicy_level": {
            "type": "string",
            "enum": ["none", "mild", "medium", "hot"],
        },
        "spicy_triggers": _STRING_ARRAY,
    },
    "required": [
        "reasoning",
        "allergens",
        "allergen_triggers",
        "gluten_free",
        "gluten_triggers",
        "vegetarian",
        "vegan",
        "spicy_level",
        "spicy_triggers",
    ],
    "additionalProperties": False,
}

TRANSLATION_CARD: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "name_en": {"type": "string"},
        "description_en": {"type": "string"},
    },
    "required": ["name_en", "description_en"],
    "additionalProperties": False,
}
