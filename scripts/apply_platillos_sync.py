"""Sync platillo YAML files with Menú del día source documents."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

KB = Path(__file__).resolve().parents[1] / "kb" / "platillos"
DOWNLOADS = Path(
    r"c:\Users\migue\Downloads\Menú del día-20260526T091545Z-3-001\Menú del día"
)

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

# Maps normalized txt variant heading -> yaml variant key
VARIANT_KEY_MAP: dict[str, dict[str, str]] = {
    "arroz": {
        "blanco": "blanco",
        "rojo": "rojo",
        "verde": "verde",
    },
    "churros": {
        "churros tradicionales": "tradicionales",
        "churros rellenos": "rellenos",
    },
    "chuleta_frita": {"chuleta frita": "chuleta_frita"},
    "chimichanga": {"chimichanga": "chimichanga"},
    "chiles_rellenos": {
        "chiles en nogada": "en_nogada",
        "chiles rellenos de queso": "relleno_de_queso",
        "chiles rellenos de carne o picadillo": "relleno_de_picadillo",
    },
    "chicharron_prensado": {
        "chicharrón prensado": "chicharron_prensado",
        "chicharron prensado": "chicharron_prensado",
    },
    "chicharron_en_salsa": {
        "chicharrón en salsa roja": "roja",
        "chicharron en salsa roja": "roja",
        "chicharrón en salsa verde": "verde",
        "chicharron en salsa verde": "verde",
    },
    "chapulines": {"chapulines": "chapulines"},
    "ceviche": {"ceviche": "ceviche"},
    "carnitas": {"carnitas": "carnitas"},
    "carne_en_su_jugo": {"carne en su jugo": "carne_en_su_jugo"},
    "carne_con_verdolagas": {"carne de cerdo con verdolagas": "carne_con_verdolagas"},
    "carne_adobada": {"carne adobada": "carne_adobada"},
    "capirotada": {"capirotada": "capirotada"},
    "caldo": {
        "caldo de habas": "habas",
        "caldo de pollo": "pollo",
        "caldo de res": "res",
        "caldo tlalpeño": "tlalpeno",
        "caldo tlalpeno": "tlalpeno",
    },
    "calabacitas": {
        "calabacitas a la mexicana": "a_la_mexicana",
        "calabacitas con puerco": "con_puerco",
    },
    "bistec": {
        "bistec encebollado": "encebollado",
        "bistec a la mexicana": "a_la_mexicana",
    },
    "birria": {"birria": "birria"},
    "ate_con_queso": {"ate con queso": "ate_con_queso", "asado de puerco": "ate_con_queso"},
    "arroz_con_leche": {
        "arroz con leche tradicional": "arroz_con_leche",
        "arroz con leche": "arroz_con_leche",
    },
    "asado_de_puerco": {"asado de puerco": "asado_de_puerco"},
    "albondigas": {
        "albóndigas en chipotle": "en_chipotle",
        "albondigas en chipotle": "en_chipotle",
        "sopa de albóndigas": "sopa",
        "sopa de albondigas": "sopa",
        "albóndigas al chipotle con arroz": "con_arroz",
        "albondigas al chipotle con arroz": "con_arroz",
        "albóndigas en salsa roja": "en_salsa_roja",
        "albondigas en salsa roja": "en_salsa_roja",
        "albóndigas en salsa verde": "en_salsa_verde",
        "albondigas en salsa verde": "en_salsa_verde",
        "albóndigas a la jardinera": "a_la_jardinera",
        "albondigas a la jardinera": "a_la_jardinera",
        "albóndigas en caldo": "en_caldo",
        "albondigas en caldo": "en_caldo",
        "albóndigas rellenas de huevo": "rellenas_de_huevo",
        "albondigas rellenas de huevo": "rellenas_de_huevo",
        "albóndigas en mole": "en_mole",
        "albondigas en mole": "en_mole",
        "albóndigas de res": "de_res",
        "albondigas de res": "de_res",
        "albóndigas mixtas (res y cerdo)": "mixtas",
        "albondigas mixtas (res y cerdo)": "mixtas",
        "albóndigas de pollo": "de_pollo",
        "albondigas de pollo": "de_pollo",
        "albóndigas vegetarianas": "vegetarianas",
        "albondigas vegetarianas": "vegetarianas",
    },
    "aguas_frescas": {
        "agua de limón con chía": "limon_con_chia",
        "agua de limon con chia": "limon_con_chia",
        "pepino con limón": "pepino_con_limon",
        "pepino con limon": "pepino_con_limon",
        "jamaica": "jamaica",
        "horchata": "horchata",
    },
    "chilaquiles": {
        "chilaquiles verdes": "verdes",
        "chilaquiles rojos": "rojos",
        "chilaquiles divorciados": "divorciados",
        "chilaquiles con pollo": "con_pollo",
        "chilaquiles con huevo": "con_huevo",
        "chilaquiles con arrachera o bistec": "con_arrachera",
        "chilaquiles con cecina": "con_cecina",
        "chilaquiles con chorizo": "con_chorizo",
        "chilaquiles de mole": "de_mole",
        "torta de chilaquiles": "torta",
    },
    "huevo": {
        "huevos rancheros": "rancheros",
        "huevos a la mexicana": "a_la_mexicana",
        "huevos con chorizo": "con_chorizo",
        "huevos divorciados": "divorciados",
        "huevos motuleños": "motulenos",
        "huevos motulenos": "motulenos",
        "huevos tirados": "tirados",
        "huevos estrellados": "estrellados",
        "huevos revueltos": "revueltos",
        "omelette": "omelette",
        "huevos con jamón": "con_jamon",
        "huevos con jamon": "con_jamon",
        "huevos con salchicha": "con_salchicha",
        "chilaquiles con huevo": "chilaquiles_con_huevo",
        "molletes con huevo": "molletes_con_huevo",
        "torta de huevo": "torta_de_huevo",
        "huevos en salsa": "en_salsa",
    },
    "tacos_dorados": {
        "tacos dorados de papa": "papa",
        "tacos dorados de pollo": "pollo",
        "tacos dorados de res": "res",
        "tacos dorados de barbacoa": "barbacoa",
        "flautas": "flautas",
        "tacos dorados de frijol": "frijol",
        "tacos dorados de requesón": "requeson",
        "tacos dorados de requeson": "requeson",
        "tacos dorados de camarón": "camaron",
        "tacos dorados de camaron": "camaron",
        "tacos dorados de chicharrón prensado": "chicharron_prensado",
        "tacos dorados de chicharron prensado": "chicharron_prensado",
    },
    "carne_asada": {
        "carne asada con guarniciones": "con_guarniciones",
        "tacos de carne asada": "tacos",
        "torta de carne asada": "torta",
        "burritos de carne asada": "burritos",
        "gringas de carne asada": "gringas",
    },
    "pechuga": {
        "pechuga asada": "asada",
        "pechuga empanizada": "empanizada",
        "milanesa de pollo": "milanesa",
        "pechuga en salsa verde": "en_salsa_verde",
        "pechuga en salsa roja": "en_salsa_roja",
        "pechuga a la mexicana": "a_la_mexicana",
        "pechuga rellena": "rellena",
    },
    "tortas_de_huauzontle": {
        "tortas de huauzontle en chile pasilla": "en_chile_pasilla",
        "tortas de huauzontle en jitomate": "en_jitomate",
    },
    "pozole": {
        "pozole blanco": "blanco",
        "pozole rojo": "rojo",
        "pozole verde": "verde",
    },
    "enchiladas": {
        "enchiladas rojas": "rojas",
        "enchiladas verdes": "verdes",
        "enchiladas suizas": "suizas",
        "enchiladas potosinas": "potosinas",
        "enchiladas mineras": "mineras",
        "enchiladas placeras": "placeras",
        "enchiladas de mole": "de_mole",
        "enchiladas queretanas": "queretanas",
        "enfrijoladas": "enfrijoladas",
        "entomatadas": "entomatadas",
    },
    "mole": {
        "poblano": "poblano",
        "negro (oaxaqueño)": "negro",
        "negro (oaxaqueno)": "negro",
        "verde": "verde",
        "rojo (genérico, comida corrida)": "rojo",
        "rojo (generico, comida corrida)": "rojo",
    },
}

NEW_VARIANTS: dict[str, dict] = {
    "albondigas": {
        "a_la_jardinera": {
            "name_es": "Albóndigas a la Jardinera",
            "name_en": "Meatballs with Mixed Vegetables",
            "extra_ingredients": ["zanahoria", "chayote", "papa", "ejotes"],
            "technique": "albóndigas cocidas con verduras mixtas en salsa ligera",
            "description_es": "Albóndigas acompañadas de verduras mixtas en una salsa ligera o caldillo.",
            "description_en": "Meatballs served with mixed vegetables in a light sauce or broth.",
        },
        "de_res": {
            "name_es": "Albóndigas de Res",
            "name_en": "Beef Meatballs",
            "extra_ingredients": ["carne_molida_de_res"],
            "technique": "albóndigas hechas exclusivamente con carne de res",
            "description_es": "Albóndigas tradicionales hechas exclusivamente con carne molida de res y sazonadores básicos.",
            "description_en": "Traditional meatballs made exclusively with ground beef and basic seasonings.",
        },
        "mixtas": {
            "name_es": "Albóndigas Mixtas (Res y Cerdo)",
            "name_en": "Mixed Beef and Pork Meatballs",
            "extra_ingredients": ["carne_molida_de_res", "carne_molida_de_cerdo"],
            "technique": "mezcla de carne de res y cerdo molida",
            "description_es": "Mezcla de carne de res y cerdo",
            "description_en": "Mixed beef and pork meatballs.",
        },
        "vegetarianas": {
            "name_es": "Albóndigas Vegetarianas",
            "name_en": "Vegetarian Meatballs",
            "extra_ingredients": ["soya", "verduras"],
            "technique": "albóndigas elaboradas sin carne",
            "description_es": "Albóndigas elaboradas sin carne",
            "description_en": "Meatless meatballs.",
        },
    },
    "chilaquiles": {
        "con_cecina": {
            "name_es": "Chilaquiles con Cecina",
            "name_en": "Cecina Chilaquiles",
            "extra_ingredients": [
                "cecina",
                "crema",
                "queso_fresco",
                "cebolla",
                "frijoles",
                "aguacate",
            ],
            "technique": "totopos en salsa con cecina",
            "description_es": "verdes o rojos servidos con cecina",
            "description_en": "Green or red chilaquiles served with cecina.",
        },
    },
    "tacos_dorados": {
        "camaron": {
            "name_es": "Tacos Dorados de Camarón",
            "name_en": "Crispy Shrimp Tacos",
            "extra_ingredients": [
                "camaron",
                "col_rallada",
                "mayonesa",
                "salsa_picante",
                "limon",
            ],
            "technique": "tortilla rellena de camarón sazonado, frita hasta dorar",
            "description_es": "Tortillas fritas rellenas de camarón sazonado, comunes en regiones costeras.",
            "description_en": "Fried tortillas stuffed with seasoned shrimp, common in coastal regions.",
        },
    },
    "enchiladas": {
        "placeras": {
            "name_es": "Enchiladas Placeras",
            "name_en": "Market-style enchiladas",
            "extra_ingredients": [
                "chile_guajillo",
                "queso_rallado",
                "crema",
                "lechuga",
                "papa",
                "zanahoria",
            ],
            "technique": "tortilla frita y bañada en salsa guajillo con papa y zanahoria",
            "description_es": "tortillas de maíz fritas ligeramente, bañadas en salsa roja de chile guajillo, y suelen rellenarse de queso fresco o pollo deshebrado. Se acompañan con papas y zanahorias cocidas,",
            "description_en": "Lightly fried corn tortillas in guajillo red sauce, usually stuffed with fresh cheese or shredded chicken, served with cooked potatoes and carrots.",
        },
    },
    "pechuga": {
        "a_la_mexicana": {
            "name_es": "Pechuga a la Mexicana",
            "name_en": "Mexican-style chicken",
            "extra_ingredients": ["jitomate", "cebolla", "chile_verde"],
            "technique": "trozos de pechuga salteados con jitomate, cebolla y chile",
            "description_es": "Se prepara con trozos de pechuga salteados con jitomate, cebolla y chile verde. Es una preparación sencilla de sabor fresco, ligeramente picante y muy tradicional, que recuerda a los colores de la bandera mexicana.",
            "description_en": "Chicken breast pieces sautéed with tomato, onion, and green chile.",
        },
    },
    "huevo": {
        "con_salchicha": {
            "name_es": "Huevos con Salchicha",
            "name_en": "Eggs with sausage",
            "extra_ingredients": ["salchicha", "cebolla"],
            "technique": "huevos revueltos con salchicha",
            "description_es": "Huevos revueltos combinados con salchicha en rodajas o picada.",
            "description_en": "Scrambled eggs combined with sliced or chopped sausage.",
        },
        "chilaquiles_con_huevo": {
            "name_es": "Chilaquiles con Huevo",
            "name_en": "Chilaquiles with eggs",
            "extra_ingredients": ["totopos", "salsa", "crema", "queso", "huevo"],
            "technique": "totopos en salsa con huevo estrellado o revuelto",
            "description_es": "Totopos de maíz bañados en salsa roja o verde, acompañados con huevos estrellados o revueltos. Se sirven con crema, queso, cebolla",
            "description_en": "Corn totopos in red or green salsa with fried or scrambled eggs, served with cream, cheese, and onion.",
        },
        "molletes_con_huevo": {
            "name_es": "Molletes con Huevo",
            "name_en": "Open-faced beans and cheese bread with eggs",
            "extra_ingredients": ["pan", "frijoles", "queso", "huevo"],
            "technique": "bolillo con frijoles, queso gratinado y huevo",
            "description_es": "Bolillo abierto con frijoles refritos y queso gratinado, acompañado de huevo preparado al gusto",
            "description_en": "Open bolillo roll with refried beans and melted cheese, served with eggs prepared to taste.",
        },
        "torta_de_huevo": {
            "name_es": "Torta de Huevo",
            "name_en": "Egg sandwich",
            "extra_ingredients": ["bolillo", "mayonesa", "aguacate", "huevo"],
            "technique": "bolillo relleno de huevo revuelto o frito",
            "description_es": "Pan tipo bolillo relleno de huevo revuelto o frito, a veces acompañado de mayonesa, aguacate o frijoles.",
            "description_en": "Bolillo roll filled with scrambled or fried eggs, sometimes with mayonnaise, avocado, or beans.",
        },
    },
}

COMMON_NAMES_UPDATES: dict[str, list[str]] = {
    "arroz": [
        "arroz",
        "arroz rojo",
        "arroz blanco",
        "arroz verde",
        "arroz mexicano",
        "arroz a la mexicana",
    ],
    "chiles_rellenos": ["chiles en nogada", "chiles rellenos"],
    "chilaquiles": [
        "chilaquiles",
        "chilaquiles verdes",
        "chilaquiles rojos",
        "chilaquiles divorciados",
        "chilaquiles con pollo",
        "chilaquiles con huevo",
        "chilaquiles con arrachera o bistec",
        "chilaquiles con cecina",
        "chilaquiles con chorizo",
        "chilaquiles de mole",
        "torta de chilaquiles",
    ],
    "albondigas": [
        "albóndigas",
        "abóndigas",
        "sopa de albóndigas",
        "albóndigas en chipotle",
        "albóndigas en salsa roja",
        "albóndigas en salsa verde",
        "albóndigas a la jardinera",
        "albóndigas en caldo",
        "albóndigas rellenas de huevo",
        "albóndigas en mole",
        "albóndigas de res",
        "albóndigas mixtas",
        "albóndigas de pollo",
        "albóndigas vegetarianas",
    ],
}

BASE_INGREDIENTS_UPDATES: dict[str, list[str]] = {
    "arroz": ["arroz", "agua", "sal", "aceite", "ajo", "cebolla"],
    "chiles_rellenos": [
        "chile_poblano",
        "jitomate",
        "tomatillo",
        "cebolla",
        "ajo",
        "aceite",
        "sal",
    ],
}


def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"^#+\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


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
        if not part or part.startswith("- "):
            continue
        lines = part.splitlines()
        variant = {
            "name": lines[0].strip(),
            "name_en": "",
            "description": "",
        }
        for ln in lines[1:]:
            m_en = re.match(r"- \*\*Nombre EN:\*\*\s*(.*)", ln, re.I)
            if m_en:
                variant["name_en"] = m_en.group(1).strip()
                continue
            m_desc = re.match(r"- \*\*Descripción:\*\*\s*(.*)", ln, re.I)
            if m_desc:
                variant["description"] = m_desc.group(1).strip()
                continue
        if variant["description"]:
            data["variants"].append(variant)
    return data


def resolve_variant_key(yaml_key: str, variant_name: str) -> str | None:
    mapping = VARIANT_KEY_MAP.get(yaml_key, {})
    key = mapping.get(norm(variant_name))
    if key:
        return key
    slug = re.sub(r"[^a-z0-9]+", "_", norm(variant_name)).strip("_")
    return slug or None


def _folded(key: str, value: str, indent: int = 4) -> list[str]:
    pad = " " * indent
    lines = [f"{pad}{key}: >"]
    for part in value.split("\n"):
        part = part.strip()
        if part:
            lines.append(f"{pad}  {part}")
    return lines


def _dump_variant(key: str, variant: dict, indent: int = 2) -> list[str]:
    pad = " " * indent
    lines = [f"{pad}{key}:"]
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
            lines.extend(_folded(desc_key, str(variant[desc_key]).strip(), indent + 2))
    return lines


def dump_yaml(data: dict) -> str:
    lines: list[str] = []
    lines.append(f"canonical_name: {data['canonical_name']}")
    lines.append("common_names:")
    for name in data.get("common_names", []):
        lines.append(f"  - {name}")
    lines.append(f"category: {data['category']}")
    lines.append("base_ingredients:")
    for ing in data.get("base_ingredients", []):
        lines.append(f"  - {ing}")
    lines.append("variants:")
    for key, variant in (data.get("variants") or {}).items():
        lines.extend(_dump_variant(key, variant))
    return "\n".join(lines) + "\n"


def sync_file(txt_key: str, yaml_key: str) -> list[str]:
    logs: list[str] = []
    txt_path = next((f for f in DOWNLOADS.glob("*.txt") if f.stem.lower() == txt_key), None)
    if not txt_path:
        return [f"skip {yaml_key}: txt not found"]

    parsed = parse_txt(txt_path)
    yaml_path = KB / f"{yaml_key}.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    variants = data.setdefault("variants", {})

    if yaml_key in COMMON_NAMES_UPDATES:
        data["common_names"] = COMMON_NAMES_UPDATES[yaml_key]

    if yaml_key in BASE_INGREDIENTS_UPDATES:
        data["base_ingredients"] = BASE_INGREDIENTS_UPDATES[yaml_key]

    chicharron_keys = ["roja", "verde"]
    for idx, tv in enumerate(parsed["variants"]):
        if yaml_key == "chicharron_en_salsa" and idx < len(chicharron_keys):
            vkey = chicharron_keys[idx]
        else:
            vkey = resolve_variant_key(yaml_key, tv["name"])
        if not vkey:
            logs.append(f"{yaml_key}: unmapped variant {tv['name']!r}")
            continue
        entry = variants.setdefault(vkey, {})
        if not entry.get("name_es"):
            entry["name_es"] = tv["name"].title()
        entry["description_es"] = tv["description"]
        logs.append(f"{yaml_key}.{vkey}: updated description_es")

    for vkey, entry in NEW_VARIANTS.get(yaml_key, {}).items():
        existing = variants.get(vkey, {})
        merged = {**entry, **existing}
        merged["description_es"] = existing.get("description_es", entry.get("description_es"))
        if entry.get("name_en") and not existing.get("name_en"):
            merged["name_en"] = entry["name_en"]
        variants[vkey] = merged
        logs.append(f"{yaml_key}.{vkey}: ensured variant")

    yaml_path.write_text(dump_yaml(data), encoding="utf-8")
    return logs


def main() -> None:
    all_logs: list[str] = []
    for txt_key, yaml_key in TXT_TO_YAML.items():
        all_logs.extend(sync_file(txt_key, yaml_key))
    for line in all_logs:
        print(line)


if __name__ == "__main__":
    main()
