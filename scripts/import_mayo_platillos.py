"""Import platillo YAML from Menú del día / 21 Mayo source documents."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import yaml

KB = Path(__file__).resolve().parents[1] / "kb" / "platillos"
INDEX = Path(__file__).resolve().parents[1] / "kb" / "entities_index.json"
MAYO = Path(
    r"c:\Users\migue\Downloads\Menú del día-20260526T091545Z-3-001\Menú del día\21 Mayo"
)

# txt stem (lower) -> (yaml_key, category, optional variant_key overrides)
DISHES: dict[str, tuple[str, str, dict[str, str] | None]] = {
    "cochinita pibil": ("cochinita_pibil", "platillo_principal", {"cochinita pibil": "cochinita_pibil"}),
    "rajas con crema": ("rajas_con_crema", "guisado", {"rajas con crema": "rajas_con_crema"}),
    "quesadillas": ("quesadillas", "antojito", {"quesadilla frita": "frita", "quesadilla": "clasica"}),
    "pipian": ("pipian", "guisado", {"pipian": "pipian", "pipian en chilacayote": "chilacayote"}),
    "picadillo": ("picadillo", "guisado", {"picadillo": "picadillo"}),
    "pescado": (
        "pescado",
        "platillo_principal",
        {"pescado a la veracruzana": "veracruzana", "pescado empanizado": "empanizado"},
    ),
    "papas": ("papas", "guarnicion", {"papas con chorizo": "con_chorizo", "papas a la francesa": "a_la_francesa"}),
    "pancita": ("pancita", "caldo", {"pancita": "pancita"}),
    "pambazos": ("pambazos", "antojito", {"pambazos": "pambazos"}),
    "nopales": ("nopales", "guarnicion", {"nopales en salsa": "en_salsa", "nopales navegantes": "navegantes"}),
    "moronga": ("moronga", "antojito", {"moronga": "moronga"}),
    "molletes": ("molletes", "antojito", None),  # wrong heading in source; fixed below
    "mole de olla": ("mole_de_olla", "caldo", {"mole de olla": "mole_de_olla"}),
    "mixiote": ("mixiote", "platillo_principal", {"mixiote": "mixiote"}),
    "milanesa": ("milanesa", "platillo_principal", {"milanesa": "milanesa"}),
    "lentejas con tocino": ("lentejas_con_tocino", "guisado", {"lentejas con tocino": "lentejas_con_tocino"}),
    "jericalla": ("jericalla", "postre", {"jericalla": "jericalla"}),
    "huazontle": (
        "huauzontle",
        "platillo_principal",
        {"huazontle en jitomate": "en_jitomate", "huontle en pasilla": "en_pasilla"},
    ),
    "huaraches": ("huaraches", "antojito", {"huarache": "huarache"}),
    "gelatina": ("gelatina", "postre", {"gelatina": "gelatina"}),
    "hígado encebollado": ("higado_encebollado", "platillo_principal", {"hígado encebollado": "higado_encebollado"}),
    "gordita frita": ("gordita_frita", "antojito", {"gordita frita": "gordita_frita"}),
    "frijoles": ("frijoles", "guarnicion", {"frijoles charros": "charros", "frijoles de olla": "de_olla"}),
    "flan": ("flan", "postre", {"flan": "flan"}),
    "espinazo con verdolagas": (
        "espinazo_con_verdolagas",
        "guisado",
        {"espinazo con verdolagas": "espinazo_con_verdolagas"},
    ),
    "ensaladas": ("ensaladas", "guarnicion", {"ensalada de atún": "de_atun", "ensalada de pepino": "de_pepino"}),
    "empanada frita": ("empanada_frita", "antojito", {"empanada": "empanada"}),
    "crema": (
        "crema",
        "caldo",
        {
            "crema de elote": "de_elote",
            "crema de espinaca": "de_espinaca",
            "crema de verduras": "de_verduras",
            "crema de champiñones": "de_champinones",
        },
    ),
    "costillas": (
        "costillas",
        "guisado",
        {"costillas en salsa verde": "en_salsa_verde", "costillas en salsa": "en_salsa_roja"},
    ),
    "consomé": (
        "consome",
        "caldo",
        {"consomé de borrego": "de_borrego", "consomé de verduras": "de_verduras"},
    ),
}

HUAUZONTLE_FIX = {"huontle en pasilla": "en_pasilla", "huazontle en pasilla": "en_pasilla"}

ING_SLUG: dict[str, str] = {
    "cerdo": "carne_de_cerdo",
    "carne de res": "carne_de_res",
    "carne de cerdo": "carne_de_cerdo",
    "carne de borrego, pollo, cerdo o conejo": "carne",
    "carne de res o cerdo": "carne",
    "panza de res": "panza_de_res",
    "hígado": "higado",
    "hígado de res o cerdo": "higado",
    "pescado clanco": "pescado_blanco",
    "pescado blanco": "pescado_blanco",
    "masa de maíz": "masa_de_maiz",
    "masa": "masa",
    "pan": "pan",
    "bolillo": "bolillo",
    "leche": "leche",
    "huevo": "huevo",
    "huauzontle": "huauzontle",
    "huazontle": "huauzontle",
    "chile poblano": "chile_poblano",
    "nopales": "nopales",
    "frijoles": "frijoles",
    "lentejas": "lentejas",
    "tocino": "tocino",
    "caldo": "caldo",
    "verduras": "verduras",
    "carne de cerdo pegada al hueso de la columna": "espinazo_de_cerdo",
    "tortillas": "tortilla_de_maiz",
    "tortillas fritas": "tortilla_de_maiz",
    "tortillas horneadas": "tortilla_de_maiz",
    "pollo deshebrado": "pollo_deshebrado",
    "carne de res deshebrada": "carne_de_res_deshebrada",
    "papa": "papa",
    "romeritos": "romeritos",
    "chile pasilla": "chile_pasilla",
    "carne de res": "carne_de_res",
}


MOJIBAKE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("preparaciรณn", "preparación"),
    ("Descripciรณn", "Descripción"),
    ("limรณn", "limón"),
    ("jamรณn", "jamón"),
    ("plรกtano", "plátano"),
    ("aรฑade", "añade"),
    ("chicharrรณn", "chicharrón"),
)


def fix_mojibake(text: str) -> str:
    for old, new in MOJIBAKE_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"^#+\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def slug_ingredient(raw: str) -> str:
    key = norm(raw)
    if key in ING_SLUG:
        return ING_SLUG[key]
    decomposed = unicodedata.normalize("NFKD", key)
    ascii_key = "".join(c for c in decomposed if not unicodedata.combining(c))
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_key)
    return slug.strip("_") or key


def parse_txt(path: Path) -> dict:
    text = fix_mojibake(path.read_text(encoding="utf-8"))
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
    parts = re.split(r"\n(?:###|##) ", block)
    for part in parts:
        part = part.strip()
        if not part or part.startswith("- "):
            continue
        lines = part.splitlines()
        variant = {"name": lines[0].strip(), "name_en": "", "description": "", "extra_ingredients": []}
        variant["name"] = re.sub(r"^#+\s*", "", variant["name"]).strip()
        in_extras = False
        for ln in lines[1:]:
            if ln.startswith("- **Ingredientes extra:**"):
                in_extras = True
                continue
            m_en = re.match(r"- \*\*Nombre EN:\*\*\s*(.*)", ln, re.I)
            if m_en:
                variant["name_en"] = m_en.group(1).strip()
                in_extras = False
                continue
            m_desc = re.match(r"- \*\*Descripción:\*\*\s*(.*)", ln, re.I)
            if not m_desc:
                m_desc = re.match(r"- \*\*Descripci.+?:\*\*\s*(.*)", ln, re.I)
            if m_desc:
                variant["description"] = m_desc.group(1).strip()
                in_extras = False
                continue
            if in_extras and re.match(r"\s+-\s+", ln):
                variant["extra_ingredients"].append(re.sub(r"^\s+-\s+", "", ln).strip())
        if variant["description"]:
            data["variants"].append(variant)
    return data


def resolve_key(yaml_key: str, variant_name: str, overrides: dict[str, str] | None) -> str:
    n = norm(variant_name)
    if overrides:
        if n in overrides:
            return overrides[n]
        for k, v in overrides.items():
            if k in n or n in k:
                return v
    if yaml_key == "huauzontle":
        for k, v in HUAUZONTLE_FIX.items():
            if k in n:
                return v
    slug = re.sub(r"[^a-z0-9]+", "_", n).strip("_")
    return slug or yaml_key


def _folded(key: str, value: str, indent: int = 4) -> list[str]:
    pad = " " * indent
    lines = [f"{pad}{key}: >"]
    for part in value.split("\n"):
        part = part.strip()
        if part:
            lines.append(f"{pad}  {part}")
    return lines


def dump_yaml(data: dict) -> str:
    lines = [
        f"canonical_name: {data['canonical_name']}",
        "common_names:",
    ]
    for name in data.get("common_names", []):
        lines.append(f"  - {name}")
    lines.append(f"category: {data['category']}")
    lines.append("base_ingredients:")
    for ing in data.get("base_ingredients", []):
        lines.append(f"  - {ing}")
    lines.append("variants:")
    for key, variant in (data.get("variants") or {}).items():
        pad = "  "
        lines.append(f"{pad}{key}:")
        for field in ("name_es", "name_en", "extra_ingredients", "technique"):
            if field not in variant:
                continue
            val = variant[field]
            if field == "extra_ingredients":
                lines.append(f"{pad}  {field}:")
                for item in val:
                    lines.append(f"{pad}    - {item}")
            else:
                lines.append(f"{pad}  {field}: {val}")
        for desc_key in ("description_es", "description_en"):
            if desc_key in variant:
                lines.extend(_folded(desc_key, str(variant[desc_key]).strip(), 4))
    return "\n".join(lines) + "\n"


def build_yaml(txt_path: Path, yaml_key: str, category: str, overrides: dict[str, str] | None) -> dict:
    parsed = parse_txt(txt_path)
    common_names = [n.strip() for n in parsed["common_names"] if n.strip()]
    if not common_names:
        common_names = [yaml_key.replace("_", " ")]

    base = [slug_ingredient(i) for i in parsed["base_ingredients"]]
    variants: dict = {}

    for tv in parsed["variants"]:
        vkey = resolve_key(yaml_key, tv["name"], overrides)
        name_es = tv["name"].strip()
        if yaml_key == "molletes" and "mole de olla" in norm(tv["name"]):
            vkey = "molletes"
            name_es = "Molletes"

        extras = [slug_ingredient(x) for x in tv["extra_ingredients"] if x.lower() not in {"no", "ninguno"}]
        variants[vkey] = {
            "name_es": name_es.title() if name_es.islower() else name_es,
            "name_en": tv["name_en"] or name_es,
            "extra_ingredients": extras,
            "technique": "",
            "description_es": tv["description"],
            "description_en": tv["description"],
        }

    return {
        "canonical_name": yaml_key,
        "common_names": common_names,
        "category": category,
        "base_ingredients": base,
        "variants": variants,
    }


def sync_existing_pechuga() -> None:
    path = MAYO / "pechuga.txt"
    parsed = parse_txt(path)
    yaml_path = KB / "pechuga.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    variants = data.setdefault("variants", {})
    mapping = {
        "pechuga asada": "asada",
        "pechuga empanizada": "empanizada",
        "milanesa de pollo": "milanesa",
        "pechuga en salsa verde": "en_salsa_verde",
        "pechuga en salsa roja": "en_salsa_roja",
        "pechuga a la mexicana": "a_la_mexicana",
        "pechuga rellena": "rellena",
        "pollo en chipotle": "en_chipotle",
    }
    for tv in parsed["variants"]:
        vkey = mapping.get(norm(tv["name"]))
        if not vkey:
            continue
        entry = variants.setdefault(vkey, {})
        entry.setdefault("name_es", tv["name"].title())
        entry.setdefault("name_en", tv["name_en"])
        entry["description_es"] = tv["description"]
        if vkey == "en_chipotle":
            entry["extra_ingredients"] = [slug_ingredient(x) for x in tv["extra_ingredients"]]
            entry["technique"] = "pollo cocinado en salsa cremosa de chipotle"
            entry["description_en"] = "Chicken cooked in a creamy chipotle sauce."
    names = data.get("common_names", [])
    if "pollo en chipotle" not in names:
        names.append("pollo en chipotle")
    data["common_names"] = names
    yaml_path.write_text(dump_yaml(data), encoding="utf-8")


def sync_existing_huevo() -> None:
    path = MAYO / "huevo.txt"
    parsed = parse_txt(path)
    yaml_path = KB / "huevo.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    variants = data.setdefault("variants", {})
    mapping = {
        "huevos rancheros": "rancheros",
        "huevos a la mexicana": "a_la_mexicana",
        "huevos con chorizo": "con_chorizo",
        "huevos divorciados": "divorciados",
        "huevos motuleños": "motulenos",
        "huevos tirados": "tirados",
        "huevos estrellados": "estrellados",
        "huevos revueltos": "revueltos",
        "omelette": "omelette",
        "huevos con jamón": "con_jamon",
        "huevos con salchicha": "con_salchicha",
        "chilaquiles con huevo": "chilaquiles_con_huevo",
        "molletes con huevo": "molletes_con_huevo",
        "torta de huevo": "torta_de_huevo",
        "huevos en salsa": "en_salsa",
        "huevos en ahogados": "en_ahogados",
        "huevos con nopales": "con_nopales",
        "huevos con tocino": "con_tocino",
    }
    new_variants = {
        "en_ahogados": {
            "name_es": "Huevos en Ahogados",
            "name_en": "Smothered Eggs",
            "extra_ingredients": ["salsa", "frijoles"],
            "technique": "huevos cocidos directamente en salsa caliente",
            "description_en": "Eggs cooked directly in hot salsa.",
        },
        "con_nopales": {
            "name_es": "Huevos con Nopales",
            "name_en": "Eggs with Cactus",
            "extra_ingredients": ["nopales", "salsa"],
            "technique": "huevos cocinados con nopales tiernos",
            "description_en": "Eggs cooked with tender cactus paddles.",
        },
        "con_tocino": {
            "name_es": "Huevos con Tocino",
            "name_en": "Eggs with Bacon",
            "extra_ingredients": ["tocino", "salsa", "frijoles"],
            "technique": "huevos con tocino crujiente",
            "description_en": "Eggs served with or mixed with crispy bacon.",
        },
    }
    for tv in parsed["variants"]:
        vkey = mapping.get(norm(tv["name"]))
        if not vkey:
            continue
        entry = variants.setdefault(vkey, {})
        if not entry.get("name_es"):
            entry["name_es"] = tv["name"].title()
        if tv["name_en"] and not entry.get("name_en"):
            entry["name_en"] = tv["name_en"]
        entry["description_es"] = tv["description"]
    for vkey, template in new_variants.items():
        if vkey not in variants:
            variants[vkey] = {**template}
        else:
            variants[vkey].setdefault("name_es", template["name_es"])
            variants[vkey].setdefault("name_en", template["name_en"])
            variants[vkey].setdefault("extra_ingredients", template["extra_ingredients"])
            variants[vkey].setdefault("technique", template["technique"])
            variants[vkey].setdefault("description_en", template["description_en"])
    yaml_path.write_text(dump_yaml(data), encoding="utf-8")


def build_index_entries(yaml_key: str, data: dict) -> dict[str, str]:
    entries: dict[str, str] = {yaml_key: yaml_key}
    for name in data.get("common_names", []):
        entries[norm(name)] = yaml_key
    for variant in (data.get("variants") or {}).values():
        if name_es := variant.get("name_es"):
            entries[norm(name_es)] = yaml_key
    return entries


def import_folder(
    folder: Path,
    dishes: dict[str, tuple[str, str, dict[str, str] | None]],
    *,
    skip_stems: set[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    created: list[str] = []
    index_updates: dict[str, str] = {}
    skip = skip_stems or set()

    for txt_file in sorted(folder.glob("*.txt")):
        stem = txt_file.stem.lower()
        if stem in skip:
            continue
        if stem not in dishes:
            print(f"skip unmapped: {txt_file.name}")
            continue
        yaml_key, category, overrides = dishes[stem]
        data = build_yaml(txt_file, yaml_key, category, overrides)
        out = KB / f"{yaml_key}.yaml"
        out.write_text(dump_yaml(data), encoding="utf-8")
        created.append(yaml_key)
        index_updates.update(build_index_entries(yaml_key, data))
    return created, index_updates


def main() -> None:
    created, index_updates = import_folder(MAYO, DISHES, skip_stems={"pechuga", "huevo"})
    sync_existing_pechuga()
    sync_existing_huevo()
    index_updates.update(build_index_entries("pechuga", yaml.safe_load((KB / "pechuga.yaml").read_text(encoding="utf-8"))))
    index_updates.update(build_index_entries("huevo", yaml.safe_load((KB / "huevo.yaml").read_text(encoding="utf-8"))))

    with INDEX.open(encoding="utf-8") as f:
        index = json.load(f)
    index.update(index_updates)
    with INDEX.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Created/updated {len(created)} dishes: {', '.join(created)}")
    print("Updated pechuga, huevo")


if __name__ == "__main__":
    main()
