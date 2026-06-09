#!/usr/bin/env python3
"""Report curation stats by comparing current files to git HEAD originals."""

import re
import subprocess
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


def git_original(name: str) -> dict | None:
    try:
        raw = subprocess.check_output(
            ["git", "show", f"HEAD:kb/platillos/{name}"],
            stderr=subprocess.DEVNULL,
        )
        return yaml.safe_load(raw)
    except subprocess.CalledProcessError:
        return None


def main():
    results = []
    all_derived = []
    for path in sorted(PLATILLOS_DIR.glob("*.yaml")):
        if path.name in SKIP:
            continue
        orig = git_original(path.name)
        if not orig:
            continue
        curr = yaml.safe_load(path.read_text(encoding="utf-8"))
        ov = len(orig.get("variants") or {})
        cv = len(curr.get("variants") or {})
        oc = len(orig.get("common_names") or [])
        cc = len(curr.get("common_names") or {})
        vr = ov - cv
        cr = oc - cc
        significant = vr > 2 or cr > 5
        results.append({
            "file": path.name,
            "variants_before": ov,
            "variants_after": cv,
            "variants_removed": vr,
            "common_before": oc,
            "common_after": cc,
            "common_removed": cr,
            "significant": significant,
        })
        # detect removed variant names as potential derived dishes
        orig_names = {v.get("name_es") for v in (orig.get("variants") or {}).values()}
        curr_names = {v.get("name_es") for v in (curr.get("variants") or {}).values()}
        for n in orig_names - curr_names:
            if n:
                fw = n.split()[0].lower()
                all_derived.append(f"{n} (from {path.name})")

    print(f"TOTAL_PROCESSED={len(results)}")
    print("\nSIGNIFICANT_CHANGES:")
    for r in results:
        if r["significant"]:
            print(
                f"  {r['file']}: variants {r['variants_before']}->{r['variants_after']} "
                f"(-{r['variants_removed']}), common_names {r['common_before']}->{r['common_after']} "
                f"(-{r['common_removed']})"
            )
    if all_derived:
        print("\nDERIVED_DISHES_REMOVED:")
        for d in sorted(set(all_derived))[:50]:
            print(f"  {d}")
        if len(set(all_derived)) > 50:
            print(f"  ... and {len(set(all_derived)) - 50} more")


if __name__ == "__main__":
    main()
