from __future__ import annotations

import logging
import os
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
    """Compute dietary flags from a list of ingredients.

    Args:
        ingredients: Base + variant ingredients for the dish.
        extras: Additional ingredients provided by the fondero.

    Returns dict with: allergens, gluten_free, vegetarian, vegan, spicy_level.
    """
    all_ingr = {i.lower().strip() for i in (ingredients + (extras or []))}

    allergens: list[str] = []
    for group_name, group in _allergens().get("groups", {}).items():
        triggers = {t.lower() for t in group.get("triggers", [])}
        if all_ingr & triggers:
            allergens.append(group_name)

    veg = _vegetarian_markers()
    breakers_meat = {i.lower() for i in veg.get("meat_proteins", [])}
    breakers_sea  = {i.lower() for i in veg.get("seafood", [])}
    breakers_ani  = {i.lower() for i in veg.get("animal_products", [])}
    is_vegetarian = not bool(all_ingr & (breakers_meat | breakers_sea))
    is_vegan      = not bool(all_ingr & (breakers_meat | breakers_sea | breakers_ani))

    spicy_level: SpicyLevel = "none"
    for level in ("hot", "medium", "mild"):
        markers = {i.lower() for i in _spicy_markers().get("levels", {}).get(level, [])}
        if all_ingr & markers:
            spicy_level = level  # type: ignore[assignment]
            break

    return {
        "allergens": sorted(allergens),
        "gluten_free": "gluten" not in allergens,
        "vegetarian": is_vegetarian,
        "vegan": is_vegan,
        "spicy_level": spicy_level,
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
