from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from functools import lru_cache
from typing import Final

import config

logger = logging.getLogger(__name__)

_CUSTOM_ENTITY: Final[str] = "custom"
_STOP_TOKENS: Final[frozenset[str]] = frozenset(
    {"de", "con", "en", "la", "el", "y", "a", "los", "las", "del", "al"}
)
# Include full variant detail (descriptions, ingredients) up to this count.
_MAX_FULL_VARIANTS: Final[int] = 8
# Cap variant keys sent as pending-slot options when a dish has many variants.
_MAX_SLOT_OPTIONS: Final[int] = 24
_MAX_COMPACT_VARIANTS: Final[int] = 24

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _yaml = None  # type: ignore[assignment]
    _YAML_AVAILABLE = False


@lru_cache(maxsize=512)
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
    """Returns full KB context string for a dish (uncached filter — tests/admin)."""
    if entity == _CUSTOM_ENTITY:
        return ""
    data = get_dish_data(entity)
    if data is not None:
        return _yaml_to_context_str(data)
    return _read_text(os.path.join("platillos", f"{entity}.txt"))


def get_variant_keys_for_slot(entity: str, limit: int = _MAX_SLOT_OPTIONS) -> list[str]:
    """Variant keys for pending-slot buttons; capped when the dish has many variants."""
    data = get_dish_data(entity)
    if not data:
        return []
    keys = sorted((data.get("variants") or {}).keys())
    if len(keys) <= limit:
        return keys
    return keys[:limit]


@lru_cache(maxsize=1)
def _load_topics_raw() -> str:
    return _read_text("topics.md")


def get_topic(intent: str, platform: str | None = None) -> tuple[str, list[dict]]:
    """Returns (context_text, links) for the given intent from topics.md.

    Links are filtered by platform when provided; intents with no platform
    sub-sections return their universal link list regardless of platform value.
    """
    raw = _load_topics_raw()
    if not raw:
        return "", []

    section = _extract_topic_section(raw, intent)
    if not section:
        return "", []

    parts = re.split(r"^### links\s*$", section, maxsplit=1, flags=re.MULTILINE)
    text = parts[0].strip()
    links: list[dict] = _parse_topic_links(parts[1], platform) if len(parts) > 1 else []
    return text, links


def _extract_topic_section(raw: str, intent: str) -> str:
    m = re.search(r"^## " + re.escape(intent) + r"\s*$", raw, re.MULTILINE)
    if not m:
        return ""
    start = m.end()
    next_m = re.search(r"^## ", raw[start:], re.MULTILINE)
    end = start + next_m.start() if next_m else len(raw)
    return raw[start:end]


def _parse_topic_links(links_raw: str, platform: str | None) -> list[dict]:
    # Split by #### platform sub-headers
    blocks = re.split(r"^#### (\S+)\s*$", links_raw, flags=re.MULTILINE)
    if len(blocks) == 1:
        # No platform sub-sections — universal links
        return _parse_json_array(links_raw.strip())

    result: list[dict] = []
    # blocks = [preamble, plat1, content1, plat2, content2, ...]
    it = iter(blocks[1:])
    for plat_name, content in zip(it, it):
        if platform is None or plat_name.strip() == platform:
            result.extend(_parse_json_array(content.strip()))
    return result


def _parse_json_array(text: str) -> list[dict]:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("["):
            try:
                parsed = json.loads(line)
                if isinstance(parsed, list):
                    return [d for d in parsed if isinstance(d, dict)]
            except json.JSONDecodeError:
                pass
    return []


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
    """Returns canonical entity names that have variants.

    YAML platillos are listed by filename only (no per-file parse) — every
    platillo YAML in the KB defines a variants block. Legacy .txt entries are
    scanned for a ## Variantes section when no YAML exists for that entity.
    """
    platillos_dir = os.path.join(config.KB_PATH, "platillos")
    result: list[str] = []
    yaml_entities: set[str] = set()
    try:
        for fname in sorted(os.listdir(platillos_dir)):
            if fname.endswith(".yaml") and _YAML_AVAILABLE:
                entity = fname[:-5]
                yaml_entities.add(entity)
                result.append(entity)
                continue

            if not fname.endswith(".txt"):
                continue

            entity = fname[:-4]
            if entity in yaml_entities:
                continue

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


def get_context_for_dishes(
    dishes: list[str],
    *,
    resolved_variants: dict[str, str] | None = None,
    pending_variant_entities: set[str] | frozenset[str] | None = None,
    conversation: str = "",
) -> str:
    """Concatenates KB context for dishes, filtering variants when possible."""
    resolved = resolved_variants or {}
    pending = pending_variant_entities or frozenset()
    parts: list[str] = []
    for dish in dishes:
        if dish == _CUSTOM_ENTITY:
            continue
        ctx = _build_filtered_dish_context(dish, resolved, pending, conversation)
        if ctx:
            parts.append(f"## {dish.capitalize()}\n{ctx}")
    return "\n\n".join(parts)


def _build_filtered_dish_context(
    entity: str,
    resolved_variants: dict[str, str],
    pending_variant_entities: set[str] | frozenset[str],
    conversation: str,
) -> str:
    data = get_dish_data(entity)
    if data is not None:
        variants = data.get("variants") or {}
        selected, compact = _select_variants_for_context(
            entity, variants, resolved_variants, pending_variant_entities, conversation
        )
        return _yaml_to_context_str(data, variants_subset=selected, compact_variants=compact)
    return _read_text(os.path.join("platillos", f"{entity}.txt"))


def _normalize_variant_key(raw: str | None, variants: dict) -> str | None:
    if not raw or not variants:
        return None
    if raw in variants:
        return raw
    normalized = raw.replace(" ", "_")
    if normalized in variants:
        return normalized
    accent_norm = _normalize_text(raw).replace(" ", "_")
    if accent_norm in variants:
        return accent_norm
    return None


def _select_variants_for_context(
    entity: str,
    variants: dict,
    resolved_variants: dict[str, str],
    pending_variant_entities: set[str] | frozenset[str],
    conversation: str,
) -> tuple[dict, bool]:
    """Return (variants_to_include, use_compact_format)."""
    if not variants:
        return {}, False

    resolved_key = _normalize_variant_key(resolved_variants.get(entity), variants)
    if resolved_key:
        return {resolved_key: variants[resolved_key]}, False

    if conversation:
        matched = match_variant_in_text(variants, conversation)
        if matched:
            return {matched: variants[matched]}, False

    n = len(variants)
    if entity in pending_variant_entities or n > _MAX_FULL_VARIANTS:
        keys = sorted(variants.keys())
        if n > _MAX_COMPACT_VARIANTS:
            subset = {k: variants[k] for k in keys[:_MAX_COMPACT_VARIANTS]}
            return subset, True
        return variants, True

    return variants, False


def conversation_text(message: str, history: list[dict[str, str]]) -> str:
    parts = [message]
    for turn in history:
        parts.append(str(turn.get("text", "")))
    return " ".join(parts)


def collect_ingredients_for_flags(
    dish: str,
    resolved_variants: dict[str, str],
    conversation: str,
) -> list[str]:
    """Collect all KB ingredients relevant to flag computation for one dish."""
    data = get_dish_data(dish)
    if not data:
        return []

    selected: list[str] = []
    seen: set[str] = set()

    def add(ingredient: str) -> None:
        normalized = _normalize_text(ingredient)
        if normalized and normalized not in seen:
            seen.add(normalized)
            selected.append(ingredient.strip().lower())

    for ingredient in data.get("base_ingredients") or []:
        add(str(ingredient))

    variants = data.get("variants") or {}
    resolved_key = resolved_variants.get(dish)
    if resolved_key and resolved_key in variants:
        for ingredient in variants[resolved_key].get("extra_ingredients") or []:
            add(str(ingredient))
    elif variants:
        matched_variant = match_variant_in_text(variants, conversation)
        if matched_variant:
            for ingredient in variants[matched_variant].get("extra_ingredients") or []:
                add(str(ingredient))

    kb_inventory = _dish_kb_inventory(data)
    for ingredient in kb_inventory:
        if _ingredient_mentioned(ingredient, conversation):
            add(ingredient)

    return selected


def _normalize_text(value: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", value.lower().strip())
        if unicodedata.category(c) != "Mn"
    )


def _singular(token: str) -> str:
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _phrase_tokens(phrase: str) -> set[str]:
    return {
        _singular(token)
        for token in _normalize_text(phrase).replace(",", " ").split()
        if token and token not in _STOP_TOKENS
    }


def _phrase_matches_text(phrase: str, text: str) -> bool:
    phrase_norm = _normalize_text(phrase)
    text_norm = _normalize_text(text)
    if phrase_norm and phrase_norm in text_norm:
        return True
    tokens = _phrase_tokens(phrase)
    if not tokens:
        return False
    text_tokens = {_singular(token) for token in text_norm.replace(",", " ").split()}
    return tokens.issubset(text_tokens)


def resolve_variants_from_conversation(
    dishes: list[str],
    resolved_variants: dict[str, str],
    conversation: str,
) -> dict[str, str]:
    """Merge LLM-resolved variants with matches found in conversation text."""
    merged = dict(resolved_variants)
    for dish in dishes:
        data = get_dish_data(dish)
        if not data:
            continue
        variants = data.get("variants") or {}
        if not variants:
            continue

        existing = merged.get(dish)
        if existing:
            if existing in variants:
                continue  # Already a valid YAML key
            # Try space→underscore (e.g. "con jamon" → "con_jamon")
            normalized = existing.replace(" ", "_")
            if normalized in variants:
                merged[dish] = normalized
                continue
            # Try accent-stripped + space→underscore (e.g. "con jamón" → "con_jamon")
            accent_norm = _normalize_text(existing).replace(" ", "_")
            if accent_norm in variants:
                merged[dish] = accent_norm
                continue
            # User stated a variant not in KB — keep as-is (user priority).
            continue

        # No variant from user — try to find one in conversation text.
        matched = match_variant_in_text(variants, conversation)
        if matched:
            merged[dish] = matched
    return merged


def match_variant_in_text(variants: dict, conversation: str) -> str | None:
    best_key: str | None = None
    best_score = 0
    for key, variant in variants.items():
        if not isinstance(variant, dict):
            continue
        candidates = [key.replace("_", " ")]
        if name_es := variant.get("name_es"):
            candidates.append(str(name_es))
        for phrase in candidates:
            if _phrase_matches_text(phrase, conversation):
                score = len(_normalize_text(phrase))
                if score > best_score:
                    best_score = score
                    best_key = key
    return best_key


def _dish_kb_inventory(data: dict) -> set[str]:
    inventory: set[str] = set()
    for ingredient in data.get("base_ingredients") or []:
        inventory.add(str(ingredient).strip().lower())
    for variant in (data.get("variants") or {}).values():
        if not isinstance(variant, dict):
            continue
        for ingredient in variant.get("extra_ingredients") or []:
            inventory.add(str(ingredient).strip().lower())
    return inventory


def _ingredient_mentioned(ingredient: str, conversation: str) -> bool:
    ingredient_norm = _normalize_text(ingredient)
    conversation_norm = _normalize_text(conversation)
    if not ingredient_norm or not conversation_norm:
        return False
    if ingredient_norm in conversation_norm:
        return True
    spaced = ingredient_norm.replace("_", " ")
    if spaced in conversation_norm:
        return True
    tokens = _phrase_tokens(spaced)
    if not tokens:
        return False
    conversation_tokens = {
        _singular(token)
        for token in conversation_norm.replace(",", " ").split()
    }
    return tokens.issubset(conversation_tokens)


def _yaml_to_context_str(
    data: dict,
    *,
    variants_subset: dict | None = None,
    compact_variants: bool = False,
) -> str:
    """Converts YAML dish data to a context string for the LLM."""
    lines: list[str] = []

    if cn := data.get("canonical_name"):
        lines.append(f"Platillo: {cn}")
    if cat := data.get("category"):
        lines.append(f"Categoría: {cat}")
    if base := data.get("base_ingredients"):
        lines.append(f"Ingredientes base: {', '.join(base)}")

    all_variants = data.get("variants") or {}
    variants = variants_subset if variants_subset is not None else all_variants
    if not variants:
        return "\n".join(lines)

    total = len(all_variants)
    shown = len(variants)

    if compact_variants:
        lines.append("\n## Variantes (resumen)")
        if total > shown:
            lines.append(
                f"({shown} de {total} variantes mostradas — pregunta al fondero cuál prepara)"
            )
        for key, v in variants.items():
            if not isinstance(v, dict):
                continue
            name_es = v.get("name_es", key)
            lines.append(f"- {name_es} ({key})")
        return "\n".join(lines)

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
