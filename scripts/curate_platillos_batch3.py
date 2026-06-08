#!/usr/bin/env python3
"""Curate platillos YAML files — batch 3 (remaining files)."""

import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import yaml

PLATILLOS_DIR = Path(__file__).resolve().parent.parent / "kb" / "platillos"

SKIP = {
    "albondigas.yaml",
    "frijoles.yaml",
    "mole.yaml",
    "tamales.yaml",
    # batch 1
    "aguas_frescas.yaml",
    "tacos.yaml",
    "crema.yaml",
    "huevo.yaml",
    "pechuga.yaml",
    "tacos_dorados.yaml",
    "consome.yaml",
    "arroz.yaml",
    "pollo.yaml",
    "sopas.yaml",
    "chilaquiles.yaml",
    "enchiladas.yaml",
    # batch 2
    "nopales.yaml",
    "pescado.yaml",
    "gordita_frita.yaml",
    "papas.yaml",
    "bistec.yaml",
    "chiles_rellenos.yaml",
    "pozole.yaml",
    "barbacoa.yaml",
    "carnitas.yaml",
    "birria.yaml",
    "caldo.yaml",
    "picadillo.yaml",
    "tinga.yaml",
    "chicharron_en_salsa.yaml",
    "costillas.yaml",
}

ARTICLES = {"el", "la", "los", "las", "un", "una", "de", "del", "al"}

BROTH_WORDS = {"sopa", "caldo", "caldillo", "consome", "consomé"}

OTHER_DISH_STARTERS = {
    "taco",
    "tacos",
    "tamal",
    "tamales",
    "sope",
    "sopes",
    "quesadilla",
    "quesadillas",
    "enchilada",
    "enchiladas",
    "tostada",
    "tostadas",
    "mollete",
    "molletes",
    "gordita",
    "gorditas",
    "tlacoyo",
    "tlacoyos",
    "memela",
    "memelas",
    "flauta",
    "flautas",
    "burrito",
    "burritos",
    "torta",
    "tortas",
    "sandwich",
    "mollete",
    "pan",
    "pay",
    "pastel",
    "empanada",
    "empanadas",
    "volovan",
    "sincronizada",
    "chimichanga",
    "huarache",
    "huaraches",
    "pambazo",
    "pambazos",
    "lonche",
    "discada",
    "pollo",
    "pescado",
    "camarón",
    "camaron",
    "camarones",
    "res",
    "cerdo",
    "puerco",
    "cabrito",
    "lengua",
    "hígado",
    "higado",
    "mollete",
    "molletes",
    "elote",
    "esquite",
    "esquites",
    "tostiloco",
    "tostilocos",
    "doriloco",
    "dorilocos",
}

PROTEIN_IN_MOLE_PATTERN = re.compile(
    r"^(pollo|carne|cerdo|puerco|res|pavo|pato|conejo)\s+en\s+",
    re.IGNORECASE,
)

ESTILO_PATTERN = re.compile(r"\bestilo\s+\w+", re.IGNORECASE)

SPANISH_MARKERS = re.compile(
    r"\b(con|de|en|el|la|los|las|servido|preparado|elaborado|cocido|"
    r"frito|guisado|bañado|bañada|tradicion|aromatizado|mezclado|"
    r"generalmente|acompañ|acompañad|hasta quedar|hecha|hecho|"
    r"bebida|postre|tortilla|salsa|carne|cerdo|pollo|frijol|"
    r"queso|crema|jitomate|cebolla|cilantro|chile|maíz|maiz)\b",
    re.IGNORECASE,
)

# Common Spanish → English for menu descriptions
TRANSLATIONS = [
    (r"\bElaborado\b", "Made"),
    (r"\belaborado\b", "made"),
    (r"\bPreparado\b", "Prepared"),
    (r"\bpreparado\b", "prepared"),
    (r"\bPostre\b", "Dessert"),
    (r"\bpostre\b", "dessert"),
    (r"\bBebida\b", "Drink"),
    (r"\bbebida\b", "drink"),
    (r"\bTortilla\b", "Tortilla"),
    (r"\btortilla\b", "tortilla"),
    (r"\bTortillas\b", "Tortillas"),
    (r"\btortillas\b", "tortillas"),
    (r"\bCarne\b", "Beef"),
    (r"\bcarne\b", "beef"),
    (r"\bCerdo\b", "Pork"),
    (r"\bcerdo\b", "pork"),
    (r"\bPollo\b", "Chicken"),
    (r"\bpollo\b", "chicken"),
    (r"\bFrijoles\b", "Beans"),
    (r"\bfrijoles\b", "beans"),
    (r"\bFrijol\b", "Bean"),
    (r"\bfrijol\b", "bean"),
    (r"\bQueso\b", "Cheese"),
    (r"\bqueso\b", "cheese"),
    (r"\bCrema\b", "Crema"),
    (r"\bcrema\b", "crema"),
    (r"\bSalsa\b", "Sauce"),
    (r"\bsalsa\b", "sauce"),
    (r"\bAtole\b", "Atole"),
    (r"\batole\b", "atole"),
    (r"\bGelatina\b", "Gelatin"),
    (r"\bgelatina\b", "gelatin"),
    (r"\bcon\b", "with"),
    (r"\bde\b", "of"),
    (r"\ben\b", "in"),
    (r"\by\b", "and"),
    (r"\bo\b", "or"),
    (r"\bServido\b", "Served"),
    (r"\bservido\b", "served"),
    (r"\bCocido\b", "Cooked"),
    (r"\bcocido\b", "cooked"),
    (r"\bFrito\b", "Fried"),
    (r"\bfrito\b", "fried"),
    (r"\bGuisado\b", "Stewed"),
    (r"\bguisado\b", "stewed"),
    (r"\bBañad[oa]\b", "Topped"),
    (r"\bbañad[oa]\b", "topped"),
    (r"\bDeshebrad[oa]\b", "Shredded"),
    (r"\bdeshebrad[oa]\b", "shredded"),
    (r"\bRelleno\b", "Stuffed"),
    (r"\brelleno\b", "stuffed"),
    (r"\bRellena\b", "Stuffed"),
    (r"\brellena\b", "stuffed"),
    (r"\bMarinado\b", "Marinated"),
    (r"\bmarinado\b", "marinated"),
    (r"\bAromatizado\b", "Flavored"),
    (r"\baromatizado\b", "flavored"),
    (r"\bMezclado\b", "Mixed"),
    (r"\bmezclado\b", "mixed"),
    (r"\bHasta quedar\b", "Until"),
    (r"\bhasta quedar\b", "until"),
    (r"\bGeneralmente\b", "Typically"),
    (r"\bgeneralmente\b", "typically"),
    (r"\bAcompañad[oa]\b", "Served with"),
    (r"\bacompañad[oa]\b", "served with"),
    (r"\bHecha\b", "Made"),
    (r"\bhecha\b", "made"),
    (r"\bHecho\b", "Made"),
    (r"\bhecho\b", "made"),
    (r"\bBase\b", "Base"),
    (r"\bbase\b", "base"),
    (r"\bVerduras\b", "Vegetables"),
    (r"\bverduras\b", "vegetables"),
    (r"\bFruta\b", "Fruit"),
    (r"\bfruta\b", "fruit"),
    (r"\bLeche\b", "Milk"),
    (r"\bleche\b", "milk"),
    (r"\bAgua\b", "Water"),
    (r"\bagua\b", "water"),
    (r"\bArroz\b", "Rice"),
    (r"\barroz\b", "rice"),
    (r"\bPapa\b", "Potato"),
    (r"\bpapa\b", "potato"),
    (r"\bPapas\b", "Potatoes"),
    (r"\bpapas\b", "potatoes"),
    (r"\bChile\b", "Chile"),
    (r"\bchile\b", "chile"),
    (r"\bJitomate\b", "Tomato"),
    (r"\bjitomate\b", "tomato"),
    (r"\bCebolla\b", "Onion"),
    (r"\bcebolla\b", "onion"),
    (r"\bCilantro\b", "Cilantro"),
    (r"\bcilantro\b", "cilantro"),
    (r"\bMaíz\b", "Corn"),
    (r"\bmaíz\b", "corn"),
    (r"\bMaiz\b", "Corn"),
    (r"\bmaiz\b", "corn"),
    (r"\bMasa\b", "Masa"),
    (r"\bmasa\b", "masa"),
    (r"\bGranos\b", "Kernels"),
    (r"\bgranos\b", "kernels"),
    (r"\bPulpa\b", "Pulp"),
    (r"\bpulpa\b", "pulp"),
    (r"\bEspeso\b", "Thick"),
    (r"\bespeso\b", "thick"),
    (r"\bEspesa\b", "Thick"),
    (r"\bespesa\b", "thick"),
    (r"\bSuave\b", "Smooth"),
    (r"\bsuave\b", "smooth"),
    (r"\bTradicional\b", "Traditional"),
    (r"\btradicional\b", "traditional"),
    (r"\bDorado\b", "Golden"),
    (r"\bdorado\b", "golden"),
    (r"\bDorada\b", "Golden"),
    (r"\bdorada\b", "golden"),
    (r"\bCrujiente\b", "Crispy"),
    (r"\bcrujiente\b", "crispy"),
    (r"\bFilete\b", "Cutlet"),
    (r"\bfilete\b", "cutlet"),
    (r"\bAtún\b", "Tuna"),
    (r"\batún\b", "tuna"),
    (r"\batun\b", "tuna"),
    (r"\bPepino\b", "Cucumber"),
    (r"\bpepino\b", "cucumber"),
    (r"\bMayonesa\b", "Mayonnaise"),
    (r"\bmayonesa\b", "mayonnaise"),
    (r"\bLimón\b", "Lime"),
    (r"\blimón\b", "lime"),
    (r"\blimon\b", "lime"),
    (r"\bVinagre\b", "Vinegar"),
    (r"\bvinagre\b", "vinegar"),
    (r"\bEspecias\b", "Spices"),
    (r"\bespecias\b", "spices"),
    (r"\bChampiñones\b", "Mushrooms"),
    (r"\bchampiñones\b", "mushrooms"),
    (r"\bChampinones\b", "Mushrooms"),
    (r"\bchampinones\b", "mushrooms"),
    (r"\bHongos\b", "Mushrooms"),
    (r"\bhongos\b", "mushrooms"),
    (r"\bDerretido\b", "Melted"),
    (r"\bderretido\b", "melted"),
    (r"\bFundido\b", "Melted"),
    (r"\bfundido\b", "melted"),
    (r"\bLentamente\b", "Slowly"),
    (r"\blentamente\b", "slowly"),
    (r"\bCubiert[oa]\b", "Topped"),
    (r"\bcubiert[oa]\b", "topped"),
    (r"\bDoblada\b", "Folded"),
    (r"\bdoblada\b", "folded"),
    (r"\bDoblado\b", "Folded"),
    (r"\bdoblado\b", "folded"),
    (r"\bFríe\b", "Fried"),
    (r"\bfríe\b", "fried"),
    (r"\bFreída\b", "Fried"),
    (r"\bfreída\b", "fried"),
    (r"\bFreído\b", "Fried"),
    (r"\bfreído\b", "fried"),
    (r"\bNuez\b", "Walnut"),
    (r"\bnuez\b", "walnut"),
    (r"\bCafé\b", "Coffee"),
    (r"\bcafé\b", "coffee"),
    (r"\bCafe\b", "Coffee"),
    (r"\bcafe\b", "coffee"),
    (r"\bGrenetina\b", "Gelatin"),
    (r"\bgrenetina\b", "gelatin"),
    (r"\bChocolate\b", "Chocolate"),
    (r"\bchocolate\b", "chocolate"),
    (r"\bPiloncillo\b", "Piloncillo"),
    (r"\bpiloncillo\b", "piloncillo"),
    (r"\bAsiento\b", "Lard"),
    (r"\basiento\b", "lard"),
    (r"\bGruesas\b", "Thick"),
    (r"\bgruesas\b", "thick"),
    (r"\bGruesa\b", "Thick"),
    (r"\bgruesa\b", "thick"),
    (r"\bRodajas\b", "Slices"),
    (r"\b rodajas\b", " slices"),
    (r"\bcubos\b", "cubes"),
    (r"\bCubos\b", "Cubes"),
    (r"\bPicadas\b", "Diced"),
    (r"\bpicadas\b", "diced"),
    (r"\bZanahoria\b", "Carrot"),
    (r"\bzanahoria\b", "carrot"),
    (r"\bChícharos\b", "Peas"),
    (r"\bchícharos\b", "peas"),
    (r"\bChicharos\b", "Peas"),
    (r"\bchicharos\b", "peas"),
    (r"\bSal\b", "Salt"),
    (r"\bsal\b", "salt"),
    (r"\bHierbas\b", "Herbs"),
    (r"\bhierbas\b", "herbs"),
    (r"\bFrescas\b", "Fresh"),
    (r"\bfrescas\b", "fresh"),
    (r"\bFresco\b", "Fresh"),
    (r"\bfresco\b", "fresh"),
    (r"\bColores\b", "Colors"),
    (r"\bcolores\b", "colors"),
    (r"\bCremosa\b", "Creamy"),
    (r"\bcremosa\b", "creamy"),
    (r"\bCremoso\b", "Creamy"),
    (r"\bcremoso\b", "creamy"),
    (r"\bSencillos\b", "Simple"),
    (r"\bsencillos\b", "simple"),
    (r"\bSencillas\b", "Simple"),
    (r"\bsencillas\b", "simple"),
    (r"\bDeshebrado\b", "Shredded"),
    (r"\bdeshebrado\b", "shredded"),
    (r"\bRes\b", "Beef"),
    (r"\bres\b", "beef"),
    (r"\bChorizo\b", "Chorizo"),
    (r"\bchorizo\b", "chorizo"),
    (r"\bVegetales\b", "Vegetables"),
    (r"\bvegetales\b", "vegetables"),
    (r"\bPan molido\b", "Breadcrumbs"),
    (r"\bpan molido\b", "breadcrumbs"),
    (r"\bHuevo\b", "Egg"),
    (r"\bhuevo\b", "egg"),
    (r"\bEnsalada\b", "Salad"),
    (r"\bensalada\b", "salad"),
    (r"\bEmpanizada\b", "Breaded"),
    (r"\bempanizada\b", "breaded"),
    (r"\bEmpanizado\b", "Breaded"),
    (r"\bempanizado\b", "breaded"),
    (r"\bNaranja agria\b", "Sour orange"),
    (r"\bnaranja agria\b", "sour orange"),
    (r"\bAchiote\b", "Achiote"),
    (r"\bachiote\b", "achiote"),
    (r"\bPinole\b", "Pinole"),
    (r"\bpinole\b", "pinole"),
    (r"\bGuayaba\b", "Guava"),
    (r"\bguayaba\b", "guava"),
    (r"\bElote\b", "Corn"),
    (r"\belote\b", "corn"),
    (r"\bFlor de calabaza\b", "Squash blossom"),
    (r"\bflor de calabaza\b", "squash blossom"),
    (r"\bHuitlacoche\b", "Corn truffle"),
    (r"\bhuitlacoche\b", "corn truffle"),
    (r"\bOaxaca\b", "Oaxaca"),
    (r"\boaxaca\b", "Oaxaca"),
    (r"\bHarina\b", "Flour"),
    (r"\bharina\b", "flour"),
    (r"\bGuisos\b", "Stews"),
    (r"\bguisos\b", "stews"),
    (r"\bGuiso\b", "Stew"),
    (r"\bguiso\b", "stew"),
    (r"\bDisuelta\b", "Dissolved"),
    (r"\bdisuelta\b", "dissolved"),
    (r"\bDisuelto\b", "Dissolved"),
    (r"\bdisuelto\b", "dissolved"),
    (r"\bJugos\b", "Juices"),
    (r"\bjugos\b", "juices"),
    (r"\bJugo\b", "Juice"),
    (r"\bjugo\b", "juice"),
    (r"\bSabores\b", "Flavors"),
    (r"\bsabores\b", "flavors"),
    (r"\bSola\b", "Alone"),
    (r"\bsola\b", "alone"),
    (r"\bSolo\b", "Alone"),
    (r"\bsolo\b", "alone"),
    (r"\bCondensada\b", "Condensed"),
    (r"\bcondensada\b", "condensed"),
    (r"\bPasas\b", "Raisins"),
    (r"\bpasas\b", "raisins"),
    (r"\bAzúcar\b", "Sugar"),
    (r"\bazúcar\b", "sugar"),
    (r"\bAzucar\b", "Sugar"),
    (r"\bazucar\b", "sugar"),
]


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def normalize(text: str) -> str:
    text = strip_accents(text.lower().strip())
    text = re.sub(r"\s+", " ", text)
    return text


def singular(word: str) -> str:
    w = normalize(word)
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and len(w) > 3:
        return w[:-1]
    return w


def canonical_roots(canonical: str) -> set[str]:
    base = normalize(canonical.replace("_", " "))
    parts = base.split()
    roots = {base, singular(base)}
    for p in parts:
        roots.add(p)
        roots.add(singular(p))
    return roots


def first_word(name_es: str) -> str:
    words = normalize(name_es).split()
    for w in words:
        if w not in ARTICLES:
            return w
    return words[0] if words else ""


def name_starts_with_canonical(name_es: str, roots: set[str]) -> bool:
    norm = normalize(name_es)
    fw = first_word(name_es)
    if fw in roots or singular(fw) in roots:
        return True
    for root in roots:
        if norm.startswith(root + " ") or norm == root:
            return True
        if norm.startswith("sopa de " + root) or norm.startswith("caldo de " + root):
            return True
        if norm.startswith("caldillo de " + root):
            return True
    return False


def is_derived_variant(name_es: str, canonical: str) -> bool:
    roots = canonical_roots(canonical)
    fw = first_word(name_es)
    fw_sing = singular(fw)

    if name_starts_with_canonical(name_es, roots):
        return False

    norm = normalize(name_es)
    words = norm.split()

    # Broth of canonical: "sopa de albóndigas"
    if words and words[0] in BROTH_WORDS:
        rest = " ".join(words[1:])
        if rest.startswith("de "):
            rest = rest[3:]
        for root in roots:
            if rest == root or rest.startswith(root + " ") or singular(rest.split()[0]) in roots:
                return False

    # Protein en sauce: "pollo en mole"
    if PROTEIN_IN_MOLE_PATTERN.match(norm):
        return True

    # Other dish + con/de canonical ingredient
    if fw in OTHER_DISH_STARTERS or fw_sing in OTHER_DISH_STARTERS:
        canon_in_name = any(r in norm for r in roots if len(r) > 3)
        if canon_in_name and not name_starts_with_canonical(name_es, roots):
            return True
        if fw not in roots and fw_sing not in roots:
            return True

    # Name starts with unrelated dish noun
    if fw in OTHER_DISH_STARTERS and fw not in roots and fw_sing not in roots:
        return True

    return False


def suggested_canonical(name_es: str) -> str:
    fw = first_word(name_es)
    return singular(fw).replace(" ", "_")


def variant_score(v: dict) -> int:
    score = 0
    for field in ("description_es", "description_en", "technique"):
        val = v.get(field, "")
        if val and str(val).strip() not in ("", "None"):
            score += len(str(val))
    score += len(v.get("extra_ingredients") or []) * 10
    return score


def merge_variants(variants: dict) -> tuple[dict, int]:
    """Merge near-duplicate variants by normalized name_es."""
    groups: dict[str, list[str]] = defaultdict(list)
    for key, v in variants.items():
        name = normalize(v.get("name_es", key))
        groups[name].append(key)

    merged = dict(variants)
    removed = 0
    for _norm, keys in groups.items():
        if len(keys) < 2:
            continue
        best = max(keys, key=lambda k: variant_score(variants[k]))
        for k in keys:
            if k != best:
                del merged[k]
                removed += 1
    return merged, removed


def consolidate_combinatorial(variants: dict) -> tuple[dict, int]:
    """Merge keys like de_elote_con_X, de_elote_con_Y into de_elote."""
    groups: dict[str, list[str]] = defaultdict(list)
    for key in variants:
        if "_con_" in key:
            base = key.split("_con_")[0]
            groups[base].append(key)

    merged = dict(variants)
    removed = 0
    for base, keys in groups.items():
        if len(keys) < 2:
            continue
        # Keep the shortest key variant or create consolidated
        keep_key = min(keys, key=len)
        best_v = max((variants[k] for k in keys), key=variant_score)
        consolidated = dict(best_v)
        # Merge extra ingredients
        all_extras = set()
        for k in keys:
            all_extras.update(variants[k].get("extra_ingredients") or [])
        consolidated["extra_ingredients"] = sorted(all_extras)
        for k in keys:
            if k in merged:
                del merged[k]
                removed += 1
        merged[base] = consolidated
        removed -= 1  # we kept one
    return merged, max(0, removed)


def remove_derived(variants: dict, canonical: str) -> tuple[dict, list[str], int]:
    kept = {}
    derived = []
    removed = 0
    for key, v in variants.items():
        name_es = v.get("name_es", key)
        if is_derived_variant(name_es, canonical):
            derived.append(f"{name_es} → {suggested_canonical(name_es)}")
            removed += 1
        else:
            kept[key] = v
    return kept, derived, removed


def clean_common_names(names: list, variants: dict, canonical: str) -> tuple[list, int]:
    if not names:
        return names, 0
    original_count = len(names)
    variant_names = {normalize(v.get("name_es", "")) for v in variants.values()}
    roots = canonical_roots(canonical)

    seen = set()
    cleaned = []
    for name in names:
        if not name or not str(name).strip():
            continue
        n = str(name).strip()
        norm = normalize(n)

        if ESTILO_PATTERN.search(n):
            continue
        if norm in seen:
            continue
        if norm in variant_names:
            continue
        if norm in roots or singular(norm.split()[0]) in roots:
            if norm == normalize(canonical.replace("_", " ")):
                continue

        seen.add(norm)
        cleaned.append(n)

    return cleaned, original_count - len(cleaned)


def is_spanish_text(text: str) -> bool:
    if not text:
        return False
    return bool(SPANISH_MARKERS.search(text))


def translate_description(desc_es: str, name_en: str) -> str:
    if not desc_es or not desc_es.strip():
        return f"{name_en}."
    result = desc_es.strip()
    for pattern, repl in TRANSLATIONS:
        result = re.sub(pattern, repl, result)
    result = re.sub(r"\s+", " ", result).strip()
    if result and result[0].islower():
        result = result[0].upper() + result[1:]
    if result and not result.endswith("."):
        result += "."
    return result


def fix_descriptions(variant: dict) -> bool:
    changed = False
    desc_es = (variant.get("description_es") or "").strip()
    desc_en = (variant.get("description_en") or "").strip()
    name_en = variant.get("name_en", "")

    if desc_es and desc_es[0].islower():
        variant["description_es"] = desc_es[0].upper() + desc_es[1:]
        changed = True

    desc_es = (variant.get("description_es") or "").strip()
    desc_en = (variant.get("description_en") or "").strip()

    if not desc_en or desc_en == desc_es or is_spanish_text(desc_en):
        new_en = translate_description(desc_es, name_en)
        if new_en != desc_en:
            variant["description_en"] = new_en
            changed = True

    technique = variant.get("technique")
    if technique == "None":
        variant["technique"] = ""
        changed = True

    return changed


def dump_yaml(data: dict) -> str:
    """Render YAML matching existing platillos style."""

    def block_scalar(text: str) -> str:
        text = text.strip()
        if not text:
            return "''"
        if "\n" in text:
            lines = text.split("\n")
            return ">\n      " + "\n      ".join(lines)
        return text

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


def is_minimal(data: dict) -> bool:
    variants = data.get("variants") or {}
    common = data.get("common_names") or []
    if len(variants) != 1:
        return False
    if len(common) > 8:
        return False
    for cn in common:
        if ESTILO_PATTERN.search(str(cn)):
            return False
    return True


def curate_file(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    canonical = data["canonical_name"]
    orig_variants = len(data.get("variants") or {})
    orig_common = len(data.get("common_names") or [])
    all_derived = []

    minimal = is_minimal(data)

    if not minimal:
        variants = data.get("variants") or {}

        variants, derived, _ = remove_derived(variants, canonical)
        all_derived.extend(derived)

        variants, comb_removed = consolidate_combinatorial(variants)
        variants, dup_removed = merge_variants(variants)

        common, cn_removed = clean_common_names(
            data.get("common_names") or [], variants, canonical
        )
        data["common_names"] = common
        data["variants"] = variants
    else:
        cn_removed = 0
        comb_removed = 0
        dup_removed = 0
        # Still dedupe common_names lightly
        common, cn_removed = clean_common_names(
            data.get("common_names") or [],
            data.get("variants") or {},
            canonical,
        )
        data["common_names"] = common

    for v in (data.get("variants") or {}).values():
        fix_descriptions(v)

    new_variants = len(data.get("variants") or {})
    new_common = len(data.get("common_names") or {})
    variants_removed = orig_variants - new_variants

    path.write_text(dump_yaml(data), encoding="utf-8")

    return {
        "file": path.name,
        "variants_before": orig_variants,
        "variants_after": new_variants,
        "variants_removed": variants_removed,
        "common_before": orig_common,
        "common_after": new_common,
        "common_removed": orig_common - new_common,
        "derived": all_derived,
        "significant": variants_removed > 2 or (orig_common - new_common) > 5,
        "minimal": minimal,
    }


def main():
    results = []
    for path in sorted(PLATILLOS_DIR.glob("*.yaml")):
        if path.name in SKIP:
            continue
        results.append(curate_file(path))

    print(f"TOTAL_PROCESSED={len(results)}")
    print("\nSIGNIFICANT_CHANGES:")
    for r in results:
        if r["significant"]:
            print(
                f"  {r['file']}: variants {r['variants_before']}->{r['variants_after']}, "
                f"common_names {r['common_before']}->{r['common_after']}"
            )

    all_derived = []
    for r in results:
        all_derived.extend(r["derived"])
    if all_derived:
        print("\nDERIVED_DISHES:")
        for d in sorted(set(all_derived)):
            print(f"  {d}")


if __name__ == "__main__":
    main()
