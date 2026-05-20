from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Final, Literal

import config

logger = logging.getLogger(__name__)

StaticTopic = Literal["higiene", "maps"]

_STATIC_TOPICS: Final[frozenset[str]] = frozenset({"higiene", "maps"})
_CUSTOM_ENTITY: Final[str] = "custom"

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _yaml = None  # type: ignore[assignment]
    _YAML_AVAILABLE = False


@lru_cache(maxsize=128)
def get_dish_data(entity: str) -> dict | None:
    """Returns parsed YAML data for a dish, or None if YAML is unavailable or missing."""
    if entity == _CUSTOM_ENTITY or not _YAML_AVAILABLE:
        return None
    path = os.path.join(config.KB_PATH, "platillos", f"{entity}.yaml")
    try:
        with open(path, encoding="utf-8") as f:
            data = _yaml.safe_load(f)
            return data if isinstance(data, dict) else None
    except FileNotFoundError:
        return None
    except _yaml.YAMLError as e:
        logger.error("dish_yaml_parse_error", extra={"entity": entity, "error": str(e)})
        return None


@lru_cache(maxsize=128)
def get_dish_context(entity: str) -> str:
    """Returns KB context string for a dish. Prefers YAML, falls back to .txt."""
    if entity == _CUSTOM_ENTITY:
        return ""
    data = get_dish_data(entity)
    if data is not None:
        return _yaml_to_context_str(data)
    return _read_text(os.path.join("platillos", f"{entity}.txt"))


@lru_cache(maxsize=8)
def get_static(topic: StaticTopic) -> str:
    if topic not in _STATIC_TOPICS:
        return ""
    return _read_text(f"{topic}.txt")


@lru_cache(maxsize=1)
def get_entities_index() -> dict[str, str]:
    full = os.path.join(config.KB_PATH, "entities_index.json")
    try:
        with open(full, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.warning("entities_index_missing", extra={"path": full})
        return {}
    except json.JSONDecodeError as e:
        logger.error("entities_index_invalid_json", extra={"path": full, "error": str(e)})
        return {}

    if not isinstance(data, dict):
        return {}
    return {
        str(k): str(v)
        for k, v in data.items()
        if isinstance(k, str) and isinstance(v, str)
    }


@lru_cache(maxsize=1)
def get_entities_with_variants() -> list[str]:
    """Returns canonical entity names with variants (YAML first, .txt fallback)."""
    platillos_dir = os.path.join(config.KB_PATH, "platillos")
    result: list[str] = []
    try:
        for fname in sorted(os.listdir(platillos_dir)):
            entity: str | None = None
            has_variants = False

            if fname.endswith(".yaml") and _YAML_AVAILABLE:
                entity = fname[:-5]
                data = get_dish_data(entity)
                has_variants = bool(data and data.get("variants"))

            elif fname.endswith(".txt"):
                entity = fname[:-4]
                # Skip .txt when a YAML already covers this entity
                if get_dish_data(entity) is not None:
                    continue
                full = os.path.join(platillos_dir, fname)
                try:
                    with open(full, encoding="utf-8") as f:
                        has_variants = "## Variantes" in f.read()
                except OSError:
                    pass

            if entity and has_variants and entity not in result:
                result.append(entity)

    except OSError:
        logger.warning("platillos_dir_missing", extra={"path": platillos_dir})
    return sorted(result)


def get_context_for_dishes(dishes: list[str]) -> str:
    """Concatenates KB context for all dishes in the list."""
    parts: list[str] = []
    for dish in dishes:
        if dish == _CUSTOM_ENTITY:
            continue
        ctx = get_dish_context(dish)
        if ctx:
            parts.append(f"## {dish.capitalize()}\n{ctx}")
    return "\n\n".join(parts)


def _yaml_to_context_str(data: dict) -> str:
    """Converts YAML dish data to a context string for the LLM."""
    lines: list[str] = []

    if cn := data.get("canonical_name"):
        lines.append(f"Platillo: {cn}")
    if cat := data.get("category"):
        lines.append(f"Categoría: {cat}")
    if base := data.get("base_ingredients"):
        lines.append(f"Ingredientes base: {', '.join(base)}")

    variants = data.get("variants") or {}
    if variants:
        lines.append("\n## Variantes")
        for key, v in variants.items():
            if not isinstance(v, dict):
                continue
            name_es = v.get("name_es", key)
            lines.append(f"\n### {name_es} ({key})")
            if name_en := v.get("name_en"):
                lines.append(f"- Nombre EN: {name_en}")
            if extras := v.get("extra_ingredients"):
                lines.append(f"- Ingredientes extra: {', '.join(extras)}")
            if tech := v.get("technique"):
                lines.append(f"- Técnica: {tech}")
            if desc_es := v.get("description_es"):
                lines.append(f"- Descripción ES: {desc_es.strip()}")
            if desc_en := v.get("description_en"):
                lines.append(f"- Descripción EN: {desc_en.strip()}")
            if sides := v.get("typical_sides"):
                lines.append(f"- Acompañamientos típicos: {', '.join(sides)}")

    return "\n".join(lines)


def _read_text(rel_path: str) -> str:
    full = os.path.join(config.KB_PATH, rel_path)
    try:
        with open(full, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning("kb_file_missing", extra={"path": full})
        return ""
    except OSError as e:
        logger.error("kb_file_read_error", extra={"path": full, "error": str(e)})
        return ""
