"""One-off helper: compare platillo txt sources with yaml descriptions."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

DOWNLOADS = Path(
    r"c:\Users\migue\Downloads\Menú del día-20260526T091545Z-3-001\Menú del día"
)
KB = Path(r"c:\Users\migue\OneDrive\Documentos\GitHub\fonda-backend\kb\platillos")

TXT_TO_YAML = {
    "arroz": "arroz",
    "churros": "churros",
    "chuleta frita": "chuleta_frita",
    "chimichanga": "chimichanga",
    "chiles": "chiles_rellenos",
    "chicharron prensado": "chicharron_prensado",
    "chicharron en salsa": "chicharron_en_salsa",
    "chapulines": "chapulines",
    "ceviche": "ceviche",
    "carnitas": "carnitas",
    "carne en su jugo": "carne_en_su_jugo",
    "carne de cerdo con verdolagas": "carne_con_verdolagas",
    "carne adobada": "carne_adobada",
    "capirotada": "capirotada",
    "caldo": "caldo",
    "calabacitas": "calabacitas",
    "bistec": "bistec",
    "birria": "birria",
    "ate con queso": "ate_con_queso",
    "arroz con leche": "arroz_con_leche",
    "asado de puerco": "asado_de_puerco",
    "albondigas": "albondigas",
    "aguas frescas": "aguas_frescas",
    "chilaquiles": "chilaquiles",
    "huevo": "huevo",
    "tacos dorados": "tacos_dorados",
    "carne asada": "carne_asada",
    "pechuga": "pechuga",
    "tortas de huauzontle": "tortas_de_huauzontle",
    "pozole": "pozole",
    "enchiladas": "enchiladas",
    "mole": "mole",
}


def parse_txt(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    data: dict = {"common_names": [], "base_ingredients": [], "variants": []}

    m = re.search(r"## Nombres comunes\s*\n(.*?)(?=\n## |\Z)", text, re.S)
    if m:
        block = m.group(1).strip()
        if block.startswith("-"):
            data["common_names"] = [
                re.sub(r"^-\s*", "", ln.strip())
                for ln in block.splitlines()
                if ln.strip().startswith("-")
            ]
        else:
            data["common_names"] = [block.split("\n")[0].strip()]

    m = re.search(r"## Ingredientes base\s*\n.*?\n\n(.*?)(?=\n## |\Z)", text, re.S)
    if m:
        for ln in m.group(1).splitlines():
            ln = ln.strip()
            if ln.startswith("-"):
                data["base_ingredients"].append(re.sub(r"^-\s*", "", ln).strip())

    m = re.search(r"## Variantes\s*\n(.*)", text, re.S)
    if not m:
        return data

    block = re.split(r"\n## Notas", m.group(1))[0]
    parts = re.split(r"\n### ", block)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.splitlines()
        variant = {
            "name": lines[0].strip(),
            "name_en": "",
            "description": "",
            "extra_ingredients": [],
        }
        for ln in lines[1:]:
            if ln.startswith("- **Nombre EN:**"):
                variant["name_en"] = ln.split(":", 1)[1].strip()
            elif ln.startswith("- **Descripción:**"):
                variant["description"] = ln.split(":", 1)[1].strip()
            elif re.match(r"\s+-\s+", ln):
                variant["extra_ingredients"].append(re.sub(r"^\s+-\s+", "", ln).strip())
        data["variants"].append(variant)
    return data


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def main() -> None:
    mismatches: list[str] = []
    for txt_key, yaml_key in TXT_TO_YAML.items():
        txt_path = next((f for f in DOWNLOADS.glob("*.txt") if f.stem.lower() == txt_key), None)
        if not txt_path:
            mismatches.append(f"MISSING TXT: {txt_key}")
            continue
        parsed = parse_txt(txt_path)
        ypath = KB / f"{yaml_key}.yaml"
        ydata = yaml.safe_load(ypath.read_text(encoding="utf-8"))
        yvars = ydata.get("variants") or {}
        txt_descs = {norm(v["name"]): v["description"] for v in parsed["variants"]}
        yaml_descs = {norm(v.get("name_es", k)): (v.get("description_es") or "").strip().replace("\n", " ") for k, v in yvars.items()}
        print(f"\n{yaml_key}: txt={len(parsed['variants'])} yaml={len(yvars)}")
        for tname, tdesc in txt_descs.items():
            matched = False
            for yname, ydesc in yaml_descs.items():
                if tname in yname or yname in tname or norm(tname.split()[0]) in yname:
                    if norm(tdesc) != norm(ydesc):
                        mismatches.append(f"{yaml_key} | {tname[:40]} | TXT: {tdesc[:50]}... | YAML: {ydesc[:50]}...")
                    matched = True
                    break
            if not matched:
                mismatches.append(f"{yaml_key} | MISSING VARIANT IN YAML: {tname}")

    print("\n--- SUMMARY ---")
    for line in mismatches:
        print(line)


if __name__ == "__main__":
    main()
