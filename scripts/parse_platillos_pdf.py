"""Parse platillos from extracted PDF text."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF_TEXT = ROOT / ".tmp_platillos_pdf.txt"
DEFAULT_PDF = Path(
    r"c:\Users\migue\AppData\Roaming\Cursor\User\workspaceStorage"
    r"\df34ac80976748aee8371a6f4929b177\pdfs"
    r"\bcef4de9-f84d-4839-9325-fdfd4b6fca37"
    r"\Copia de Base de conocimiento platillos _Huevito_.pdf"
)


def join_lines(block: str) -> str:
    lines = [
        ln.strip()
        for ln in block.splitlines()
        if ln.strip() and not ln.strip().startswith("--- PAGE")
    ]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def slug(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def parse_entries(text: str) -> list[dict]:
    start = text.find("1.  Platillo:")
    if start < 0:
        raise ValueError("platillos section not found")
    section = text[start:]
    chunks = re.split(r"(?=\n\d+\.\s+Platillo:)", "\n" + section)
    entries: list[dict] = []
    pattern = re.compile(
        r"\n?(\d+)\.\s+Platillo:\s+(.+?)\s+Nombre\s+en\s+ingl[eé]s:\s+"
        r"(.+?)\s+Descripci[oó]n\s+del\s+platillo:\s+(.+?)\s+Ingredientes\s+principales:\s+"
        r"(.+?)\s+Variantes:\s*(.*?)(?=\n\d+\.\s+Platillo:|\Z)",
        re.I | re.S,
    )
    for chunk in chunks:
        m = pattern.match(chunk)
        if not m:
            continue
        num, name_es, name_en, desc, ingredients, variants = m.groups()
        entries.append(
            {
                "num": int(num),
                "name_es": join_lines(name_es),
                "name_en": join_lines(name_en),
                "description_es": join_lines(desc),
                "ingredients_raw": join_lines(ingredients),
                "variants_raw": join_lines(variants),
                "slug": slug(join_lines(name_es)),
            }
        )
    return entries


def load_pdf_text() -> str:
    if PDF_TEXT.exists():
        return PDF_TEXT.read_text(encoding="utf-8")
    if not DEFAULT_PDF.exists():
        raise FileNotFoundError(f"PDF not found: {DEFAULT_PDF}")
    from pypdf import PdfReader

    reader = PdfReader(str(DEFAULT_PDF))
    parts = []
    for i, page in enumerate(reader.pages):
        parts.append(f"--- PAGE {i + 1} ---\n{page.extract_text() or ''}")
    text = "\n".join(parts)
    PDF_TEXT.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    text = load_pdf_text()
    entries = parse_entries(text)
    print(f"total entries: {len(entries)}")
    if entries:
        print(f"range: {entries[0]['num']} - {entries[-1]['num']}")

    out = ROOT / ".tmp_platillos_parsed.json"
    out.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")

    existing = {p.stem for p in (ROOT / "kb" / "platillos").glob("*.yaml")}
    new_slugs = [e for e in entries if e["slug"] not in existing]
    print(f"not matching existing yaml slug: {len(new_slugs)}")
    for e in new_slugs[:30]:
        print(f"  {e['num']}. {e['name_es']} -> {e['slug']}")
    if len(new_slugs) > 30:
        print(f"  ... and {len(new_slugs) - 30} more")


if __name__ == "__main__":
    main()
