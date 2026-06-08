"""Import platillos from client PDF into kb/platillos YAML + entities_index.json."""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_platillos_pdf import load_pdf_text, parse_entries  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "kb" / "platillos"
INDEX_PATH = ROOT / "kb" / "entities_index.json"
PDF_TEXT = ROOT / ".tmp_platillos_pdf.txt"

# Parent phrase (normalized) -> canonical yaml key
PARENT_MAP: dict[str, str] = {
    "taco": "tacos",
    "tacos": "tacos",
    "enchilada": "enchiladas",
    "enchiladas": "enchiladas",
    "chile relleno": "chiles_rellenos",
    "chiles rellenos": "chiles_rellenos",
    "mole": "mole",
    "tinga": "tinga",
    "picadillo": "picadillo",
    "albondiga": "albondigas",
    "albondigas": "albondigas",
    "bistec": "bistec",
    "caldo": "caldo",
    "pozole": "pozole",
    "birria": "birria",
    "quesadilla": "quesadillas",
    "quesadillas": "quesadillas",
    "gordita": "gordita_frita",
    "sopes": "sopes",
    "sope": "sopes",
    "huarache": "huaraches",
    "huaraches": "huaraches",
    "tostada": "tostadas",
    "tostadas": "tostadas",
    "flauta": "tacos_dorados",
    "flautas": "tacos_dorados",
    "enfrijolada": "enchiladas",
    "enfrijoladas": "enchiladas",
    "entomatada": "enchiladas",
    "entomatadas": "enchiladas",
    "chilaquiles": "chilaquiles",
    "chilaquile": "chilaquiles",
    "mollete": "molletes",
    "molletes": "molletes",
    "tamal": "tamales",
    "tamales": "tamales",
    "cochinita": "cochinita_pibil",
    "cochinita pibil": "cochinita_pibil",
    "carne asada": "carne_asada",
    "arrachera": "carne_asada",
    "frijoles": "frijoles",
    "nopal": "nopales",
    "nopales": "nopales",
    "rajas": "rajas_con_crema",
    "calabacita": "calabacitas",
    "calabacitas": "calabacitas",
    "tortita de papa": "tortitas_de_papa",
    "tortitas de papa": "tortitas_de_papa",
    "tortita de camaron": "tortitas_de_camaron",
    "tortitas de camaron": "tortitas_de_camaron",
    "tortita de camarón": "tortitas_de_camaron",
    "tortitas de camarón": "tortitas_de_camaron",
    "pescado": "pescado",
    "ceviche": "ceviche",
    "arroz": "arroz",
    "arroz con leche": "arroz_con_leche",
    "huevo": "huevo",
    "huevos": "huevo",
    "machaca": "machaca",
    "costilla": "costillas",
    "costillas": "costillas",
    "cerdo": "cerdo",
    "pollo": "pollo",
    "mixiote": "mixiote",
    "pipian": "pipian",
    "pipián": "pipian",
    "mole de olla": "mole_de_olla",
    "barbacoa": "barbacoa",
    "consome": "consome",
    "consomé": "consome",
    "consomé de barbacoa": "consome",
    "menudo": "pancita",
    "sopa": "sopas",
    "crema": "crema",
    "ensalada": "ensaladas",
    "chorizo": "chorizo",
    "camarones": "camarones",
    "camaron": "camarones",
    "camarón": "camarones",
    "aguachile": "aguachile",
    "coctel de camaron": "coctel_de_camaron",
    "cóctel de camarón": "coctel_de_camaron",
    "pulpo": "pulpo",
    "discada": "discada",
    "papadzules": "papadzules",
    "salbutes": "salbutes",
    "panuchos": "panuchos",
    "migas": "migas",
    "asado de boda": "asado_de_boda",
    "asado de puerco": "asado_de_puerco",
    "carne de puerco con verdolagas": "carne_con_verdolagas",
    "carne en su jugo": "carne_en_su_jugo",
    "carne adobada": "carne_adobada",
    "carne en salsa verde": "carne_en_salsa_verde",
    "milanesa": "milanesa",
    "pechuga": "pechuga",
    "huachinango": "pescado",
    "gelatina": "gelatina",
    "flan": "flan",
    "churros": "churros",
    "empanada": "empanada_frita",
    "pambazo": "pambazos",
    "pambazos": "pambazos",
    "chimichanga": "chimichanga",
    "carnitas": "carnitas",
    "moronga": "moronga",
    "romeritos": "romeritos",
    "lentejas": "lentejas_con_tocino",
    "jericalla": "jericalla",
    "capirotada": "capirotada",
    "ate con queso": "ate_con_queso",
    "aguas frescas": "aguas_frescas",
    "agua fresca": "aguas_frescas",
}

ING_SLUG: dict[str, str] = {
    "cerdo": "carne_de_cerdo",
    "carne de cerdo": "carne_de_cerdo",
    "carne de res": "carne_de_res",
    "pollo": "pollo",
    "huevo": "huevo",
    "huevos": "huevo",
    "tortilla de maíz": "tortilla_de_maiz",
    "tortilla de maiz": "tortilla_de_maiz",
    "tortillas de maíz": "tortilla_de_maiz",
    "tortillas de maiz": "tortilla_de_maiz",
    "queso fresco": "queso_fresco",
    "queso": "queso",
    "frijoles": "frijoles",
    "jitomate": "jitomate",
    "cebolla": "cebolla",
    "ajo": "ajo",
    "chile guajillo": "chile_guajillo",
    "chile ancho": "chile_ancho",
    "chile serrano": "chile_serrano",
    "cilantro": "cilantro",
    "crema": "crema",
    "aguacate": "aguacate",
    "limón": "limon",
    "limon": "limon",
    "arroz": "arroz",
    "masa": "masa_de_maiz",
    "masa de maíz": "masa_de_maiz",
    "papa": "papa",
    "papas": "papa",
    "camaron": "camaron",
    "camarón": "camaron",
    "camarones": "camaron",
    "pescado": "pescado",
    "nopales": "nopales",
    "chorizo": "chorizo",
    "jamón": "jamon",
    "jamon": "jamon",
    "sal": "sal",
    "aceite": "aceite",
    "manteca": "manteca",
}

CATEGORY_KEYWORDS: list[tuple[str, str]] = [
    ("postre", "postre"),
    ("flan", "postre"),
    ("gelatina", "postre"),
    ("jericalla", "postre"),
    ("churro", "postre"),
    ("arroz con leche", "postre"),
    ("capirotada", "postre"),
    ("ate con queso", "postre"),
    ("agua fresca", "bebida"),
    ("aguas frescas", "bebida"),
    ("sopa", "caldo"),
    ("caldo", "caldo"),
    ("consomé", "caldo"),
    ("consome", "caldo"),
    ("pozole", "caldo"),
    ("menudo", "caldo"),
    ("birria", "caldo"),
    ("mole de olla", "caldo"),
    ("crema de", "caldo"),
    ("taco", "antojito"),
    ("tostada", "antojito"),
    ("quesadilla", "antojito"),
    ("gordita", "antojito"),
    ("sope", "antojito"),
    ("huarache", "antojito"),
    ("tamal", "antojito"),
    ("enchilada", "antojito"),
    ("chilaquiles", "antojito"),
    ("mollete", "antojito"),
    ("flauta", "antojito"),
    ("empanada", "antojito"),
    ("pambazo", "antojito"),
    ("arroz", "guarnicion"),
    ("frijoles", "guarnicion"),
    ("nopales", "guarnicion"),
    ("ensalada", "guarnicion"),
]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower().strip())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", norm(s)).strip("_")


def slug_ingredient(raw: str) -> str:
    key = norm(raw)
    if key in ING_SLUG:
        return ING_SLUG[key]
    return slug(key) or "ingrediente"


def parse_ingredients(raw: str) -> list[str]:
    if not raw:
        return []
    parts = re.split(r"[,;]|\s+y\s+|\s+con\s+", raw, flags=re.I)
    result: list[str] = []
    seen: set[str] = set()
    for part in parts:
        part = part.strip(" .●-")
        if not part or part.lower() in {"etc", "etc."}:
            continue
        ing = slug_ingredient(part)
        if ing not in seen:
            seen.add(ing)
            result.append(ing)
    return result


def infer_category(name_es: str) -> str:
    n = norm(name_es)
    for needle, cat in CATEGORY_KEYWORDS:
        if needle in n:
            return cat
    return "platillo_principal"


def _primary_from_compound(n: str) -> str | None:
    """Extract primary dish from 'X con Y' when Y is accompaniment."""
    m = re.match(r"^(.+?)\s+con\s+(huevo|pollo|res|cerdo|camaron|queso|frijol|nopal|nopales)\s*$", n)
    if m:
        return m.group(1).strip()
    m = re.match(r"^(huevo|huevos|chorizo|machaca|nopales?)\s+con\s+.+$", n)
    if m:
        return m.group(1).strip()
    return None


def resolve_canonical(name_es: str, index: dict[str, str]) -> str | None:
    n = norm(name_es)
    primary = _primary_from_compound(n)
    if primary:
        if primary in PARENT_MAP:
            return PARENT_MAP[primary]
        if primary in index:
            return index[primary]
        primary_slug = slug(primary)
        if primary_slug in {p.stem for p in KB.glob("*.yaml")}:
            return primary_slug
    if n in index:
        return index[n]
    for parent, canonical in sorted(PARENT_MAP.items(), key=lambda x: -len(x[0])):
        if n == parent or n.startswith(parent + " "):
            return canonical
    for alias, canonical in index.items():
        if len(alias) >= 6 and (alias in n or n in alias):
            return canonical
    return None


def variant_key(canonical: str, name_es: str) -> str:
    n = norm(name_es)
    canon_display = norm(canonical.replace("_", " "))
    if n == canon_display:
        return canonical
    for parent in sorted(PARENT_MAP, key=len, reverse=True):
        p = norm(parent)
        if n.startswith(p + " "):
            suffix = n[len(p) :].strip()
            if suffix:
                return slug(suffix)
    if " " in n:
        return slug(n)
    return slug(name_es) or canonical


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def folded(key: str, value: str, indent: int = 4) -> list[str]:
    pad = " " * indent
    lines = [f"{pad}{key}: >"]
    for part in value.split("\n"):
        part = part.strip()
        if part:
            lines.append(f"{pad}  {part}")
    return lines


def dump_yaml(data: dict) -> str:
    lines = [f"canonical_name: {data['canonical_name']}", "common_names:"]
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
            if desc_key in variant and variant[desc_key]:
                lines.extend(folded(desc_key, str(variant[desc_key]).strip(), 4))
    return "\n".join(lines) + "\n"


def ensure_entity(entities: dict[str, dict], canonical: str, name_es: str) -> dict:
    if canonical not in entities:
        entities[canonical] = {
            "canonical_name": canonical,
            "common_names": [],
            "category": infer_category(name_es),
            "base_ingredients": [],
            "variants": {},
        }
    ent = entities[canonical]
    if name_es not in ent["common_names"]:
        ent["common_names"].append(name_es)
    return ent


def apply_entry(
    entities: dict[str, dict],
    entry: dict,
    index: dict[str, str],
) -> tuple[str, str]:
    name_es = entry["name_es"]
    canonical = resolve_canonical(name_es, index)
    if not canonical:
        canonical = slug(name_es)
    ent = ensure_entity(entities, canonical, name_es)
    vkey = variant_key(canonical, name_es)
    ingredients = parse_ingredients(entry["ingredients_raw"])
    if not ent["base_ingredients"] and ingredients:
        ent["base_ingredients"] = ingredients[:6]
    existing = ent["variants"].get(vkey, {})
    merged_ingredients = list(existing.get("extra_ingredients") or [])
    for ing in ingredients:
        if ing not in merged_ingredients:
            merged_ingredients.append(ing)
    pdf_name_en = (entry["name_en"] or "").strip()
    existing_name_en = (existing.get("name_en") or "").strip()
    if existing_name_en and pdf_name_en:
        name_en = existing_name_en if len(existing_name_en) >= len(pdf_name_en) else pdf_name_en
    else:
        name_en = pdf_name_en or existing_name_en or name_es
    ent["variants"][vkey] = {
        "name_es": name_es,
        "name_en": name_en,
        "extra_ingredients": merged_ingredients,
        "technique": existing.get("technique") or "",
        "description_es": entry["description_es"] or existing.get("description_es", ""),
        "description_en": entry["description_es"] or existing.get("description_en", ""),
    }
    return canonical, vkey


def rebuild_index(entities: dict[str, dict], old_index: dict[str, str]) -> dict[str, str]:
    index = dict(old_index)
    for canonical, ent in entities.items():
        index[norm(canonical.replace("_", " "))] = canonical
        for name in ent.get("common_names", []):
            index[norm(name)] = canonical
        for vkey, variant in (ent.get("variants") or {}).items():
            name_es = variant.get("name_es", "")
            if name_es:
                index[norm(name_es)] = canonical
            name_en = variant.get("name_en", "")
            if name_en:
                index[norm(name_en)] = canonical
    return dict(sorted(index.items(), key=lambda x: x[0]))


def main() -> None:
    text = load_pdf_text()
    entries = parse_entries(text)
    old_index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))

    entities: dict[str, dict] = {}
    for yaml_path in KB.glob("*.yaml"):
        data = load_yaml(yaml_path)
        if data.get("canonical_name"):
            entities[data["canonical_name"]] = data

    stats = defaultdict(int)
    for entry in entries:
        canonical, vkey = apply_entry(entities, entry, old_index)
        stats[canonical] += 1

    KB.mkdir(parents=True, exist_ok=True)
    for canonical, data in sorted(entities.items()):
        out = KB / f"{canonical}.yaml"
        out.write_text(dump_yaml(data), encoding="utf-8")

    new_index = rebuild_index(entities, old_index)
    INDEX_PATH.write_text(
        json.dumps(new_index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"imported {len(entries)} pdf entries")
    print(f"canonical entities: {len(entities)}")
    print(f"index aliases: {len(new_index)}")
    print(f"new yaml files: {sum(1 for c in entities if not (KB / f'{c}.yaml').exists())}")
    top = sorted(stats.items(), key=lambda x: -x[1])[:15]
    print("top entities by variant count:")
    for c, n in top:
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
