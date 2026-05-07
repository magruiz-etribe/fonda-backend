"""Carga prompts de texto desde el directorio prompts/ (junto al código)."""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(name: str) -> str:
    """Lee prompts/<name>.strip() en UTF-8. name incluye extensión, ej. 'classifier_system.txt'."""
    path = _PROMPTS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def prompts_dir() -> Path:
    return _PROMPTS_DIR
