#!/usr/bin/env python3
"""Second pass: fix description_en quality in batch-3 curated platillos."""

import re
import unicodedata
from pathlib import Path

import yaml

PLATILLOS_DIR = Path(__file__).resolve().parent.parent / "kb" / "platillos"

SKIP = {
    "albondigas.yaml", "frijoles.yaml", "mole.yaml", "tamales.yaml",
    "aguas_frescas.yaml", "tacos.yaml", "crema.yaml", "huevo.yaml",
    "pechuga.yaml", "tacos_dorados.yaml", "consome.yaml", "arroz.yaml",
    "pollo.yaml", "sopas.yaml", "chilaquiles.yaml", "enchiladas.yaml",
    "nopales.yaml", "pescado.yaml", "gordita_frita.yaml", "papas.yaml",
    "bistec.yaml", "chiles_rellenos.yaml", "pozole.yaml", "barbacoa.yaml",
    "carnitas.yaml", "birria.yaml", "caldo.yaml", "picadillo.yaml",
    "tinga.yaml", "chicharron_en_salsa.yaml", "costillas.yaml",
}

ING_EN = {
    "masa_de_maiz": "corn masa",
    "frijoles": "beans",
    "frijol": "beans",
    "pollo": "chicken",
    "res": "beef",
    "carne_de_cerdo": "pork",
    "cerdo": "pork",
    "chorizo": "chorizo",
    "crema": "Mexican crema",
    "queso": "cheese",
    "queso_oaxaca": "Oaxaca cheese",
    "jitomate": "tomato",
    "tomatillo": "tomatillo",
    "chile_serrano": "serrano chile",
    "chile_guajillo": "guajillo chile",
    "chile_chipotle": "chipotle",
    "chipotle": "chipotle",
    "tomate_verde": "tomatillo",
    "elote": "corn",
    "leche": "milk",
    "grenetina": "gelatin",
    "guayaba": "guava",
    "pinole": "toasted corn meal",
    "piloncillo": "piloncillo",
    "chocolate": "chocolate",
    "achiote": "achiote",
    "naranja_agria": "sour orange",
    "flor_de_calabaza": "squash blossom",
    "huitlacoche": "corn truffle",
    "champinones": "mushrooms",
    "papa": "potato",
    "arroz": "rice",
    "verduras": "vegetables",
    "lechuga": "lettuce",
    "salsa": "salsa",
    "tortilla": "tortilla",
    "tortilla_de_maiz": "corn tortilla",
    "cafe": "coffee",
    "azucar": "sugar",
    "fruta": "fruit",
    "pasas": "raisins",
    "nuez": "walnuts",
    "leche_condensada": "condensed milk",
    "gelatina": "gelatin",
    "atun": "tuna",
    "at_n": "tuna",
    "pepino": "cucumber",
    "mayonesa": "mayonnaise",
    "cebolla": "onion",
    "chile": "chile",
    "vinagre": "vinegar",
    "menta": "mint",
    "calabaza": "zucchini",
    "zanahoria": "carrot",
    "manteca": "lard",
    "epazote": "epazote",
    "tocino": "bacon",
    "mole": "mole sauce",
    "suadero": "suadero beef",
    "cecina": "cecina",
    "tripa": "tripe",
    "lengua": "beef tongue",
    "camarón": "shrimp",
    "camaron": "shrimp",
    "pescado": "fish",
    "huevo": "egg",
    "pan_molido": "breadcrumbs",
    "carne_molida": "ground meat",
    "carne_molida_de_res": "ground beef",
    "carne_molida_de_cerdo": "ground pork",
    "pollo_molido": "ground chicken",
    "res_molida": "ground beef",
    "soya": "soy protein",
    "cerveza": "beer",
    "chayote": "chayote",
    "ejotes": "green beans",
    "pescado": "fish",
    "huevo_cocido": "hard-boiled egg",
    "salsa_roja": "red sauce",
    "queso_fresco": "fresh cheese",
    "ajonjoli": "sesame seeds",
    "arroz_rojo": "red rice",
    "chile_chipotle": "chipotle chile",
}

SPANISH_MARKERS = re.compile(
    r"\b(con|de|en|el|la|los|las|servido|preparado|elaborado|cocido|"
    r"frito|guisado|bañado|bañada|tradicion|aromatizado|mezclado|"
    r"generalmente|acompañ|acompañad|hasta quedar|hecha|hecho|"
    r"bebida|postre|tortilla|salsa|carne|cerdo|pollo|frijol|"
    r"queso|crema|jitomate|cebolla|cilantro|chile|maíz|maiz|"
    r"normalmente|puede|distintos|para darle|puede incluir|"
    r"sencillos|sencillas|combinada|combinado|como|se combina|"
    r"darle|ingredientes|base de|a base|disuelta|disuelto|"
    r"aromatizada|rellenas|rellena|relleno|cocida|cocido|"
    r"deshebrada|deshebrado|frescas|frescos|colores|cremosa|"
    r"cremoso|sola|solo|jugos|sabores|acompañada|servirse|"
    r"prepararse|doblada|rellena|fríe|freída|freído|cubiert)\b",
    re.IGNORECASE,
)


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(text.lower().strip()))


def ing_list(ings: list) -> str:
    parts = []
    for i in (ings or [])[:5]:
        parts.append(ING_EN.get(i, i.replace("_", " ")))
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def needs_fix(desc_en: str, desc_es: str) -> bool:
    if not desc_en or not desc_en.strip():
        return True
    if normalize(desc_en) == normalize(desc_es):
        return True
    if SPANISH_MARKERS.search(desc_en):
        return True
    if re.search(r"\bof (corn|beef|cheese|bean)\b.*\bof\b", desc_en, re.I):
        return True
    return False


def generate_en(name_en: str, desc_es: str, extras: list, technique: str) -> str:
    d = normalize(desc_es)
    name = name_en.strip()

    # Exact-pattern library (normalized desc_es -> English)
    EXACT = {
        "cerdo marinado en achiote y cocido lentamente.":
            "Pork marinated in achiote paste and slow-cooked until tender.",
        "atole espeso con chocolate y piloncillo.":
            "Thick corn masa drink with chocolate and piloncillo.",
        "atole aromatizado con pulpa de guayaba.":
            "Warm atole flavored with guava pulp.",
        "bebida tradicional hecha con maiz tostado molido.":
            "Traditional warm drink made with toasted ground corn.",
        "bebida espesa hecha con granos de elote.":
            "Thick warm drink made with fresh corn kernels.",
        "gelatina aromatizada con cafe.":
            "Coffee-flavored gelatin dessert.",
        "postre suave preparado con leche y grenetina.":
            "Smooth milk gelatin dessert.",
        "postre elaborado con cubos de gelatina de colores en base cremosa.":
            "Colorful gelatin cubes set in a creamy base.",
        "elaborado a base de grenetina o gelatina disuelta en agua, leche o jugos de fruta. puede prepararse en distintos sabores y colores, y servirse sola o acompanada con fruta, crema o leche condensada.":
            "Gelatin dessert made with water, milk, or fruit juice. Served plain or with fruit, crema, or condensed milk.",
        "tortillas gruesas cubiertas con asiento, salsa y queso.":
            "Thick corn tortillas topped with lard, salsa, and fresh cheese.",
        "tortilla doblada con queso o guisos.":
            "Folded tortilla filled with cheese or stew.",
        "tortillas de maiz o harina con queso derretido.":
            "Corn or flour tortillas filled with melted cheese.",
        "quesadillas rellenas de champinones.":
            "Quesadillas filled with sautéed mushrooms.",
        "quesadilla rellena con huitlacoche y queso fundido.":
            "Quesadilla filled with huitlacoche and melted Oaxaca cheese.",
        "tortilla rellena de queso y flor de calabaza cocida.":
            "Tortilla filled with cheese and cooked squash blossom.",
        "tortilla de maiz doblada y rellena normalmente de queso, papa, hongos o carne. se frie en aceite hasta quedar dorada y crujiente.":
            "Folded corn tortilla filled with cheese, potato, mushrooms, or meat, then fried until golden and crisp.",
        "base gruesa de maiz cubierta con frijoles, carne y vegetales.":
            "Thick corn masa bases topped with beans, meat, and vegetables.",
        "sopes con chorizo frito y crema.":
            "Thick corn sopes topped with fried chorizo and crema.",
        "sopes con pollo deshebrado.":
            "Thick corn sopes topped with shredded chicken.",
        "sopes con carne de res deshebrada.":
            "Thick corn sopes topped with shredded beef.",
        "sopes con pollo en chipotle.":
            "Thick corn sopes topped with chicken in chipotle sauce.",
        "sopes con chorizo y papa.":
            "Thick corn sopes topped with chorizo and potato.",
        "sopes sencillos de frijol y queso.":
            "Simple sopes with beans and cheese.",
        "carne deshebrada en salsa verde.":
            "Shredded beef simmered in green tomatillo sauce.",
        "filete fino de res o cerdo cubierto con pan molido y huevo, frito hasta quedar dorado.":
            "Thin beef or pork cutlet breaded and fried until golden.",
        "atun mezclado con verduras picadas como jitomate, cebolla, zanahoria, chicharos o papa. generalmente se combina con mayonesa, limon y especias para darle sabor.":
            "Tuna mixed with diced tomato, onion, carrot, peas, or potato, bound with mayonnaise, lime, and spices.",
        "pepino en rodajas o cubos, mezclado con ingredientes como jitomate, cebolla, limon y sal, puede incluir chile, queso, crema, hierbas frescas o vinagre.":
            "Sliced or diced cucumber with tomato, onion, lime, and salt; may include chile, cheese, crema, fresh herbs, or vinegar.",
    }

    if d in EXACT:
        return EXACT[d]
    # Also try without trailing period
    if d.rstrip(".") + "." in EXACT:
        return EXACT[d.rstrip(".") + "."]

    # Heuristic patterns
    if d.startswith("sopes con "):
        rest = desc_es.split(" con ", 1)[-1].rstrip(".")
        tr = translate_short(rest).rstrip(".")
        return f"Thick corn sopes topped with {tr.lower()}."
    if d.startswith("sopes sencillos de "):
        rest = desc_es.split(" de ", 1)[-1].rstrip(".")
        return f"Simple sopes with {translate_short(rest).rstrip('.').lower()}."
    if "quesadilla rellena" in d or "tortilla rellena" in d or "tortilla de maiz doblada" in d:
        return translate_short(desc_es)
    if "atole" in d:
        return translate_short(desc_es)
    if "postre" in d or "gelatina" in d or "elaborado a base" in d:
        return translate_short(desc_es)
    if "bebida" in d:
        return translate_short(desc_es)
    if "base gruesa" in d:
        return translate_short(desc_es)

    tech = (technique or "").strip()
    if tech and tech.lower() != "none":
        return f"{name}. {translate_short(tech)}"

    ings = ing_list(extras)
    if ings:
        return f"{name} made with {ings}."
    return f"{name}."


def translate_short(text: str) -> str:
    """Translate short Spanish menu phrases to clean English."""
    t = text.strip().rstrip(".")
    replacements = [
        (r"^Carne deshebrada en ", "Shredded beef in "),
        (r"^Albóndigas ", "Meatballs "),
        (r"^Albondigas ", "Meatballs "),
        (r" en salsa verde$", " in green tomatillo sauce"),
        (r" en salsa roja$", " in red tomato sauce"),
        (r" en chipotle$", " in chipotle sauce"),
        (r" en mole$", " in mole sauce"),
        (r" con ", " with "),
        (r" y ", " and "),
        (r" de ", " of "),
        (r" en ", " in "),
        (r"cocido lentamente", "slow-cooked"),
        (r"cocida lentamente", "slow-cooked"),
        (r"marinado en achiote", "marinated in achiote"),
        (r"deshebrado", "shredded"),
        (r"deshebrada", "shredded"),
        (r"frito", "fried"),
        (r"frita", "fried"),
        (r"guisado", "stewed"),
        (r"guisada", "stewed"),
        (r"^Tortilla de maíz doblada", "Folded corn tortilla"),
        (r"^Tortilla de maiz doblada", "Folded corn tortilla"),
        (r"^Tortilla rellena de queso y flor de calabaza cocida", "Tortilla filled with cheese and cooked squash blossom"),
        (r"^Quesadilla rellena con huitlacoche y queso fundido", "Quesadilla filled with huitlacoche and melted cheese"),
        (r"^Tortillas de maíz o harina con queso derretido", "Corn or flour tortillas filled with melted cheese"),
        (r"^Base gruesa de maíz cubierta con frijoles, carne y vegetales", "Thick corn masa bases topped with beans, meat, and vegetables"),
        (r"^Elaborado a base de grenetina", "Gelatin dessert made with gelatin dissolved in water, milk, or fruit juice"),
        (r"normalmente de ", "typically with "),
        (r"Se fríe en aceite hasta quedar dorada y crujiente", "Fried in oil until golden and crisp"),
        (r"Se frie en aceite hasta quedar dorada y crujiente", "Fried in oil until golden and crisp"),
        (r"Puede prepararse en distintos sabores y colores, y servirse sola o acompañada con fruta, crema o leche condensada", "Available in various flavors; served plain or with fruit, crema, or condensed milk"),
        (r"flor de calabaza cocida", "cooked squash blossom"),
        (r"queso fundido", "melted cheese"),
        (r"carne de res deshebrada", "shredded beef"),
        (r"chorizo frito", "fried chorizo"),
        (r"pollo deshebrado", "shredded chicken"),
        (r"aromatizado con", "flavored with"),
        (r"aromatizada con", "flavored with"),
        (r"preparado con", "prepared with"),
        (r"preparada con", "prepared with"),
        (r"elaborado con", "made with"),
        (r"elaborada con", "made with"),
        (r"espeso con", "thickened with"),
        (r"espesa con", "thickened with"),
    ]
    result = t
    for pat, repl in replacements:
        result = re.sub(pat, repl, result, flags=re.IGNORECASE)
    if result and result[0].islower():
        result = result[0].upper() + result[1:]
    return result + "."


def dump_yaml(data: dict) -> str:
    lines = [f"canonical_name: {data['canonical_name']}"]
    lines.append("common_names:")
    for cn in data.get("common_names") or []:
        lines.append(f"  - {cn}")
    lines.append(f"category: {data.get('category', 'platillo_principal')}")
    lines.append("base_ingredients:")
    for ing in data.get("base_ingredients") or []:
        lines.append(f"  - {ing}")
    lines.append("variants:")
    for key, v in (data.get("variants") or {}).items():
        lines.append(f"  {key}:")
        lines.append(f"    name_es: {v.get('name_es', '')}")
        lines.append(f"    name_en: {v.get('name_en', '')}")
        lines.append("    extra_ingredients:")
        for ing in v.get("extra_ingredients") or []:
            lines.append(f"      - {ing}")
        technique = v.get("technique") or ""
        lines.append(f"    technique: {technique}")
        desc_es = (v.get("description_es") or "").strip()
        desc_en = (v.get("description_en") or "").strip()
        lines.append("    description_es: >")
        lines.append(f"      {desc_es}")
        lines.append("    description_en: >")
        lines.append(f"      {desc_en}")
    return "\n".join(lines) + "\n"


def main():
    fixed = 0
    for path in sorted(PLATILLOS_DIR.glob("*.yaml")):
        if path.name in SKIP:
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        changed = False
        for v in (data.get("variants") or {}).values():
            desc_es = (v.get("description_es") or "").strip()
            desc_en = (v.get("description_en") or "").strip()
            if desc_es and desc_es[0].islower():
                v["description_es"] = desc_es[0].upper() + desc_es[1:]
                changed = True
            desc_es = (v.get("description_es") or "").strip()
            desc_en = (v.get("description_en") or "").strip()
            if needs_fix(desc_en, desc_es):
                new_en = generate_en(
                    v.get("name_en", ""),
                    desc_es,
                    v.get("extra_ingredients") or [],
                    v.get("technique") or "",
                )
                if new_en != desc_en:
                    v["description_en"] = new_en
                    changed = True
        if changed:
            path.write_text(dump_yaml(data), encoding="utf-8")
            fixed += 1
    print(f"DESCRIPTION_FIX_FILES={fixed}")


if __name__ == "__main__":
    main()
