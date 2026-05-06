from __future__ import annotations

import logging
import re
import unicodedata
from typing import Final

import config

logger = logging.getLogger(__name__)


_ES_STOPWORDS: Final[frozenset[str]] = frozenset({
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "y", "o", "u", "de", "del", "al", "a", "en", "que", "con",
    "por", "para", "es", "son", "no", "si", "sí", "su", "sus",
    "se", "te", "me", "lo", "le", "como", "más", "pero", "muy",
    "esto", "eso", "esta", "este", "estos", "estas", "ya", "también",
    "está", "están", "puede", "puedes", "tiene", "tu", "tus",
})

_EN_STOPWORDS: Final[frozenset[str]] = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "is", "are", "was",
    "were", "for", "with", "in", "on", "at", "by", "this", "that",
    "these", "those", "it", "its", "as", "be", "you", "your",
    "have", "has", "from", "but",
})

_COMMON_ALLERGENS: Final[frozenset[str]] = frozenset({
    "gluten", "trigo", "lactosa", "lacteos", "lacteo", "leche", "queso",
    "huevo", "huevos", "frutos secos", "nueces", "cacahuate", "cacahuates",
    "almendra", "almendras", "mariscos", "crustaceos", "moluscos",
    "soya", "soja", "ajonjoli", "sesamo", "pescado", "mostaza", "apio",
    "sulfitos", "altramuces",
})

_WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_ALLERGEN_LINE_RE = re.compile(
    r"al[ée]rgenos?\s*[:\-]\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)

_MIN_ES_STOPWORD_HITS: Final[int] = 2


def validate(
    reply: str,
    kb_context: str,
    user_ingredients: list[str],
) -> tuple[bool, str | None]:
    if not reply or not reply.strip():
        return False, "respuesta vacía"

    if len(reply) > config.MAX_REPLY_LEN:
        return False, "respuesta excede longitud máxima"

    if _CONTROL_CHAR_RE.search(reply):
        return False, "contiene caracteres de control"

    if not _looks_spanish(reply):
        return False, "respuesta no parece estar en español"

    invented = _check_invented_allergens(reply, kb_context, user_ingredients)
    if invented:
        return False, f"alérgenos no respaldados: {', '.join(invented)}"

    return True, None


def _looks_spanish(text: str) -> bool:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if not words:
        return False
    es = sum(1 for w in words if w in _ES_STOPWORDS)
    en = sum(1 for w in words if w in _EN_STOPWORDS)
    return es >= _MIN_ES_STOPWORD_HITS and es >= en


def _strip_diacritics(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _check_invented_allergens(
    reply: str,
    kb_context: str,
    user_ingredients: list[str],
) -> list[str]:
    m = _ALLERGEN_LINE_RE.search(reply)
    if not m:
        return []
    declared = [d.strip() for d in re.split(r"[,;/]", m.group(1)) if d.strip()]
    if not declared:
        return []

    allowed_haystack = _strip_diacritics(
        kb_context + " " + " ".join(user_ingredients)
    )
    common = {_strip_diacritics(c) for c in _COMMON_ALLERGENS}

    invented: list[str] = []
    for d in declared:
        d_norm = _strip_diacritics(d)
        if not d_norm:
            continue
        if any(_token_match(c, d_norm) for c in common):
            continue
        if d_norm in allowed_haystack:
            continue
        invented.append(d)
    return invented


def _token_match(canonical: str, declared: str) -> bool:
    return canonical == declared or canonical in declared or declared in canonical
