"""
merr_abstraktet.py – Shkarkon abstraktet reale të mini-korpusit nga Europe PMC REST API.

Përdorimi (nga rrënja e repository-t):
    python src/merr_abstraktet.py

Çfarë bën:
  1. Lexon data/manifests/02_Korpusi_Pilot_Manifest.csv (8 burime REAL06).
  2. Për çdo burim kërkon në Europe PMC sipas PMID, PMCID ose DOI (resultType=core).
  3. Krahason titullin e API-së me titullin në manifest (mbrojtje nga ID të gabuara).
  4. Plotëson fushat që mungojnë (PMID/PMCID/DOI, autorët, revista), licencën nga API
     dhe vendos shareable=True vetëm për licenca CC BY / CC0.
  5. Ruan abstraktin FJALË PËR FJALË (pa HTML) te data/private/02_Abstraktet_Reale.csv
     dhe përgjigjen bruto JSON te data/private/raw/ (për auditim).
  6. Përditëson manifestin (api_verified, data e marrjes).

data/private/ është në .gitignore: tekstet nuk rishpërndahen në repository. Në CSV-të
e rezultateve, fragmentet e burimeve pa licencë të hapur zëvendësohen me një shënim.

Nuk shpiken burime: nëse një burim nuk gjendet, shënohet NOT_FOUND dhe nuk përdoret.
"""
from __future__ import annotations

import datetime as dt
import difflib
import html
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/02_Korpusi_Pilot_Manifest.csv"
OUT = ROOT / "data/private/02_Abstraktet_Reale.csv"
RAW = ROOT / "data/private/raw"
API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def clean_abstract(raw: str) -> str:
    """clean_v1: heq etiketat HTML, ruan titujt e seksioneve si 'Titull: ', normalizon hapësirat."""
    s = raw or ""
    s = re.sub(r"<h4>\s*(.*?)\s*</h4>", r" \1: ", s, flags=re.I | re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    return s


def is_open_license(lic: str) -> bool:
    lic = (lic or "").lower().replace("-", " ")
    return lic.startswith("cc by") or lic.startswith("cc0")


def query_for(row: pd.Series) -> str:
    if pd.notna(row.get("PMID")) and str(row["PMID"]).strip():
        return f"EXT_ID:{int(float(row['PMID']))} AND SRC:MED"
    if pd.notna(row.get("PMCID")) and str(row["PMCID"]).strip():
        return f"PMCID:{row['PMCID']}"
    return f'DOI:"{row["DOI"]}"'


def fetch(query: str) -> dict:
    r = requests.get(API, params={"query": query, "resultType": "core", "format": "json",
                                  "pageSize": 1}, timeout=30)
    r.raise_for_status()
    return r.json()


def main() -> int:
    man = pd.read_csv(MANIFEST, dtype=str)
    RAW.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    rows, problems = [], []

    for i, row in man.iterrows():
        sid = row["source_id"]
        q = query_for(row)
        try:
            data = fetch(q)
        except Exception as exc:
            problems.append(f"{sid}: gabim API {exc!r}")
            man.loc[i, "api_verified"] = "API_ERROR"
            continue
        (RAW / f"{sid}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        res = (data.get("resultList") or {}).get("result") or []
        if not res:
            problems.append(f"{sid}: nuk u gjet me query {q}")
            man.loc[i, "api_verified"] = "NOT_FOUND"
            continue
        a = res[0]
        api_title = clean_abstract(a.get("title", "")).rstrip(".")
        sim = difflib.SequenceMatcher(None, api_title.lower(), str(row["title"]).lower().rstrip(".")).ratio()
        if sim < 0.85:
            problems.append(f"{sid}: titulli i API-së ndryshon nga manifesti (ngjashmëri {sim:.2f}): {api_title}")
            man.loc[i, "api_verified"] = f"TITLE_MISMATCH({sim:.2f})"
            continue

        for col, key in [("PMID", "pmid"), ("PMCID", "pmcid"), ("DOI", "doi"),
                         ("authors", "authorString"), ("publication_year", "pubYear")]:
            if (pd.isna(row.get(col)) or not str(row.get(col)).strip()) and a.get(key):
                man.loc[i, col] = str(a[key])
        if pd.isna(row.get("journal")) or not str(row.get("journal")).strip():
            man.loc[i, "journal"] = ((a.get("journalInfo") or {}).get("journal") or {}).get("title", "")

        api_license = a.get("license") or ""
        lic = api_license if api_license else f"{row['license_status']} (API: pa licencë)"
        man.loc[i, "license_status"] = lic
        man.loc[i, "shareable"] = str(is_open_license(api_license))
        if is_open_license(api_license):
            man.loc[i, "inclusion_decision"] = "candidate"
        man.loc[i, "api_verified"] = f"OK {today}"

        abstract = clean_abstract(a.get("abstractText", ""))
        if not abstract:
            problems.append(f"{sid}: API nuk ka abstrakt")
            man.loc[i, "inclusion_decision"] = "excluded (pa abstrakt)"
            continue
        rows.append({
            "source_id": sid, "PMID": man.loc[i, "PMID"], "PMCID": man.loc[i, "PMCID"],
            "DOI": man.loc[i, "DOI"], "title": api_title, "source_url": row["source_url"],
            "license_status": lic, "shareable": is_open_license(api_license),
            "api_query": q, "retrieval_date": today, "cleaning_version": "clean_v1",
            "abstract_text": abstract,
        })
        time.sleep(0.3)  # sjellje e mirë ndaj API-së

    if not rows:
        print("Asnjë abstrakt nuk u shkarkua. Manifesti NUK u ndryshua. Kontrolloni internetin.")
        for p in problems:
            print("  KUJDES:", p)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")
    man.to_csv(MANIFEST, index=False, encoding="utf-8")

    print(f"U ruajtën {len(rows)} abstrakte te {OUT.relative_to(ROOT)}")
    print(f"Manifesti u përditësua: {MANIFEST.relative_to(ROOT)}")
    for p in problems:
        print("  KUJDES:", p)
    if len(rows) < 5:
        print("  KUJDES: më pak se 5 abstrakte – kriteri i javës 2 NUK plotësohet.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
