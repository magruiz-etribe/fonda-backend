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


@lru_cache(maxsize=128)
def get_dish_context(entity: str) -> str:
    if entity == _CUSTOM_ENTITY:
        return ""
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
    """Returns canonical entity names whose KB file contains a '## Variantes' section."""
    platillos_dir = os.path.join(config.KB_PATH, "platillos")
    result: list[str] = []
    try:
        for fname in os.listdir(platillos_dir):
            if not fname.endswith(".txt"):
                continue
            entity = fname[:-4]
            full = os.path.join(platillos_dir, fname)
            try:
                with open(full, encoding="utf-8") as f:
                    if "## Variantes" in f.read():
                        result.append(entity)
            except OSError:
                pass
    except OSError:
        logger.warning("platillos_dir_missing", extra={"path": platillos_dir})
    return sorted(result)


def get_context_for_dishes(dishes: list[str]) -> str:
    """Concatena el contexto KB de todos los platillos del array."""
    parts: list[str] = []
    for dish in dishes:
        if dish == _CUSTOM_ENTITY:
            continue
        ctx = get_dish_context(dish)
        if ctx:
            parts.append(f"## {dish.capitalize()}\n{ctx}")
    return "\n\n".join(parts)


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
