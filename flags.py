from __future__ import annotations

import logging
import os
import unicodedata
from functools import lru_cache
from typing import Literal

import yaml

import config

logger = logging.getLogger(__name__)

SpicyLevel = Literal["none", "mild", "medium", "hot"]


def compute_flags(
    ingredients: list[str],
    extras: list[str] | None = None,
) -> dict:
    """Compute dietary flags from a unified list of ingredients.

    Args:
        ingredients: Base, variant, and user-provided ingredients for the dish(es).
        extras: Deprecated alias kept for backward compatibility in tests.

    Returns dict with: allergens, gluten_free, vegetarian, vegan, spicy_level.
    """
    merged = list(ingredients)
    if extras:
        merged.extend(extras)
    all_ingr = {_normalize(i) for i in merged}

    allergens: list[str] = []
    allergen_triggers: set[str] = set()
    gluten_triggers: set[str] = set()
    for group_name, group in _allergens().get("groups", {}).items():
        triggers = {_normalize(t) for t in group.get("triggers", [])}
        matched = all_ingr & triggers
        if matched:
            allergens.append(group_name)
            if group_name == "gluten":
                gluten_triggers |= matched
            else:
                allergen_triggers |= matched

    veg = _vegetarian_markers()
    breakers_meat = {_normalize(i) for i in veg.get("meat_proteins", [])}
    breakers_sea  = {_normalize(i) for i in veg.get("seafood", [])}
    breakers_ani  = {_normalize(i) for i in veg.get("animal_products", [])}
    is_vegetarian = not bool(all_ingr & (breakers_meat | breakers_sea))
    is_vegan      = not bool(all_ingr & (breakers_meat | breakers_sea | breakers_ani))

    spicy_level: SpicyLevel = "none"
    spicy_triggers: set[str] = set()
    for level in ("hot", "medium", "mild"):
        markers = {_normalize(i) for i in _spicy_markers().get("levels", {}).get(level, [])}
        matched_spicy = all_ingr & markers
        if matched_spicy:
            spicy_triggers |= matched_spicy
            if spicy_level == "none":
                spicy_level = level  # type: ignore[assignment]

    return {
        "allergens": bool(allergen_triggers),
        "allergen_triggers": sorted(allergen_triggers),
        "gluten_free": "gluten" not in allergens,
        "gluten_triggers": sorted(gluten_triggers),
        "vegetarian": is_vegetarian,
        "vegan": is_vegan,
        "spicy_level": spicy_level,
        "spicy_triggers": sorted(spicy_triggers),
    }


@lru_cache(maxsize=1)
def _allergens() -> dict:
    return _load_ref("allergens.yaml")


@lru_cache(maxsize=1)
def _spicy_markers() -> dict:
    return _load_ref("spicy_markers.yaml")


@lru_cache(maxsize=1)
def _vegetarian_markers() -> dict:
    return _load_ref("vegetarian_markers.yaml")


def _normalize(s: str) -> str:
    """Lowercase + strip accents + unify underscores/spaces for robust ingredient matching."""
    s = s.lower().strip().replace('_', ' ')
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def _load_ref(filename: str) -> dict:
    path = os.path.join(config.KB_PATH, filename)
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("flags_ref_missing", extra={"file": filename})
        return {}
    except yaml.YAMLError as e:
        logger.error("flags_ref_parse_error", extra={"file": filename, "error": str(e)})
        return {}
