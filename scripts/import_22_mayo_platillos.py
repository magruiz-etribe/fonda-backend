"""Import platillo YAML from Menú del día / 22 Mayo source documents."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from import_mayo_platillos import (  # noqa: E402
    INDEX,
    build_index_entries,
    import_folder,
)

MAYO_22 = Path(
    r"c:\Users\migue\Downloads\Menú del día-20260526T091545Z-3-001\Menú del día\22 Mayo"
)

DISHES_22: dict[str, tuple[str, str, dict[str, str] | None]] = {
    "totopos": ("totopos", "botana", {"totopos": "totopos"}),
    "tostadas": ("tostadas", "antojito", {"tostadas": "tostadas"}),
    "tortitas de papa": (
        "tortitas_de_papa",
        "antojito",
        {"tortitas de papa": "tortitas_de_papa"},
    ),
    "tinga de pollo": ("tinga", "guisado", {"tinga de pollo": "tinga_de_pollo"}),
    "tamales": ("tamales", "antojito", {"tamales": "tamales"}),
    "sopes fritos": ("sopes", "antojito", {"sopes": "sopes"}),
    "sopas": (
        "sopas",
        "caldo",
        {
            "sopa de fideo": "de_fideo",
            "sopa de lentejas": "de_lentejas",
            "sopa de tortilla": "de_tortilla",
        },
    ),
    "salpicón de res": (
        "salpicon",
        "platillo_principal",
        {"salpicón": "salpicon", "salpicon": "salpicon"},
    ),
    "romeritos": ("romeritos", "guisado", {"romeritos": "romeritos"}),
    "res en pasilla": (
        "res_en_pasilla",
        "guisado",
        {"res en pasilla": "res_en_pasilla"},
    ),
}


def main() -> None:
    created, index_updates = import_folder(MAYO_22, DISHES_22)

    with INDEX.open(encoding="utf-8") as f:
        index = json.load(f)
    index.update(index_updates)
    with INDEX.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Created/updated {len(created)} dishes: {', '.join(created)}")


if __name__ == "__main__":
    main()
