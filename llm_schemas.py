"""JSON schemas for Bedrock structured outputs via Nova tool use."""
from __future__ import annotations

import json
from typing import Any, Final

# Bedrock structured-output subset (JSON Schema Draft 2020-12).
_UNSUPPORTED_SCHEMA_KEYS: Final[frozenset[str]] = frozenset({
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "multipleOf",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "maxItems",
    "pattern",
    "minProperties",
    "maxProperties",
})


def _string_array(description: str = "") -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "array",
        "items": {"type": "string"},
    }
    if description:
        schema["description"] = description
    return schema


def _bool_optional(description: str = "") -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "boolean"}
    if description:
        schema["description"] = description
    return schema


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
        "current_dish": {"type": "string", "description": "Canonical dish name, 'custom' if not in KB, or empty string"},
        "companions": _string_array("Side dishes served separately"),
    },
    "required": ["reasoning", "current_dish", "companions"],
    "additionalProperties": False,
}

EXTRACTING: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "response": _string_array(
            "Exactly one question bubble when variables_complete is false; empty when complete"
        ),
        "variables_complete": {"type": "boolean"},
        "collected_ingredients": _string_array("Ingredients collected so far"),
        "buttons": _string_array("Quick-reply button labels for the active question"),
    },
    "required": ["response", "variables_complete", "collected_ingredients", "buttons"],
    "additionalProperties": False,
}

CONFIRMING_FLAGS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "response": _string_array("One confirmation question bubble"),
        "buttons": _string_array("Yes/no confirmation buttons"),
    },
    "required": ["response", "buttons"],
    "additionalProperties": False,
}

DRAFTING: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "response": _string_array(
            "Bilingual menu card bubble and confirmation bubble; use \\n for line breaks"
        ),
        "buttons": _string_array("Draft action buttons"),
        "current_dishes": _string_array("Canonical dish ids in context"),
    },
    "required": ["response", "buttons", "current_dishes"],
    "additionalProperties": False,
}

GENERATION: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "response": _string_array("Chat bubbles shown to the user"),
        "current_dishes": _string_array("Active dish ids"),
        "buttons": _string_array("Quick-reply buttons for the active question"),
        "completeness_confirmed": _bool_optional(
            "True/false only when the user just answered completeness; omit otherwise"
        ),
        "allergens_confirmed": _bool_optional(
            "True/false only when the user just answered allergen confirmation; omit otherwise"
        ),
        "gluten_confirmed": _bool_optional(
            "True/false only when the user just answered gluten confirmation; omit otherwise"
        ),
        "spicy_confirmed": _bool_optional(
            "True/false only when the user just answered spicy confirmation; omit otherwise"
        ),
    },
    "required": ["response", "current_dishes", "buttons"],
    "additionalProperties": False,
}

FLAGS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "allergens": {"type": "boolean"},
        "allergen_triggers": _string_array(),
        "gluten_free": {"type": "boolean"},
        "gluten_triggers": _string_array(),
        "vegetarian": {"type": "boolean"},
        "vegan": {"type": "boolean"},
        "spicy_level": {
            "type": "string",
            "enum": ["none", "mild", "medium", "hot"],
        },
        "spicy_triggers": _string_array(),
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

ALL_SCHEMAS: Final[dict[str, dict[str, Any]]] = {
    "CLASSIFIER_INTENT": CLASSIFIER_INTENT,
    "EXTRACTOR_TRADUCCION": EXTRACTOR_TRADUCCION,
    "EXTRACTING": EXTRACTING,
    "CONFIRMING_FLAGS": CONFIRMING_FLAGS,
    "DRAFTING": DRAFTING,
    "GENERATION": GENERATION,
    "FLAGS": FLAGS,
    "TRANSLATION_CARD": TRANSLATION_CARD,
}


def validate_bedrock_schema(schema: dict[str, Any], *, path: str = "$") -> list[str]:
    """Return human-readable errors if a schema violates Bedrock structured-output rules."""
    errors: list[str] = []

    if path == "$":
        if schema.get("type") != "object":
            errors.append(f"{path}: root type must be 'object'")
        try:
            json.dumps(schema)
        except (TypeError, ValueError) as e:
            errors.append(f"{path}: schema is not JSON-serializable: {e}")

    for key, value in schema.items():
        if key in _UNSUPPORTED_SCHEMA_KEYS:
            errors.append(f"{path}: unsupported key '{key}'")
        if isinstance(value, dict):
            if value.get("type") == "object" and value.get("additionalProperties") is not False:
                errors.append(f"{path}.{key}: nested object must set additionalProperties to false")
            errors.extend(validate_bedrock_schema(value, path=f"{path}.{key}"))

    if schema.get("type") == "object":
        if schema.get("additionalProperties") is not False:
            errors.append(f"{path}: additionalProperties must be false")
        props = schema.get("properties")
        if not isinstance(props, dict) or not props:
            errors.append(f"{path}: object schemas must declare non-empty properties")
        else:
            for prop_name, prop_schema in props.items():
                if isinstance(prop_schema, dict):
                    errors.extend(validate_bedrock_schema(prop_schema, path=f"{path}.{prop_name}"))

    items = schema.get("items")
    if isinstance(items, dict):
        errors.extend(validate_bedrock_schema(items, path=f"{path}.items"))

    return errors


def assert_all_schemas_valid() -> None:
    """Raise AssertionError if any registered schema is invalid for Bedrock."""
    for name, schema in ALL_SCHEMAS.items():
        errors = validate_bedrock_schema(schema)
        if errors:
            joined = "; ".join(errors)
            raise AssertionError(f"schema {name} is invalid for Bedrock structured output: {joined}")
