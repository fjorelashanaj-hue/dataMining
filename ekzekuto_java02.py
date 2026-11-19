"""
ekzekuto_java02.py – Gjithë Java 2 me një komandë.

    python ekzekuto_java02.py

Hapat: (1) shkarkon abstraktet reale nga Europe PMC, (2) retrieval + baseline + RAG mbi SIM06
dhe korpusin real, (3) rindërton notebook-ët me rezultatet, (4) ekzekuton testet,
(5) shkruan një përmbledhje të asaj që mbetet për kontroll manual.

Para ekzekutimit: pip install -r requirements.txt  dhe vendosni çelësin LLM, p.sh.
    Windows PowerShell:  $env:GROQ_API_KEY="..."
    macOS/Linux:         export GROQ_API_KEY="..."
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def step(title, args):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    r = subprocess.run([PY, *args], cwd=ROOT)
    return r.returncode == 0


ok_fetch = step("1/4  Shkarkimi i abstrakteve reale (Europe PMC)", ["src/merr_abstraktet.py"])
real = ROOT / "data/private/02_Abstraktet_Reale.csv"
corpus = "both" if real.exists() and real.stat().st_size > 50 else "sim"
ok_run = step(f"2/4  Retrieval + baseline + RAG (--corpus {corpus})", ["src/run_java02.py", "--corpus", corpus])
ok_nb = step("3/4  Rindërtimi i notebook-ëve", ["tools/build_notebooks.py"])
ok_test = step("4/4  Testet", ["-m", "pytest", "-q"])

print(f"\n{'=' * 70}\nPËRMBLEDHJE\n{'=' * 70}")
log = json.loads((ROOT / "results/02_run_log.json").read_text(encoding="utf-8"))
for c, r in log["runs"].items():
    print(f"{c}: {r['n_documents']} dokumente, {r['n_chunks']} chunk-e | embedder: {r['embedder']} | "
          f"indeksi: {r['index']} | LLM: {r['llm']['model']}")
warn = []
if not ok_fetch:
    warn.append("Abstraktet reale nuk u shkarkuan (shih mesazhet KUJDES më lart).")
if "REAL06" not in log["runs"]:
    warn.append("Korpusi real nuk u ekzekutua – porta go/no-go NUK kalohet.")
if any("FALLBACK" in r["embedder"] or "FALLBACK" in r["index"] for r in log["runs"].values()):
    warn.append("U përdor fallback (TF-IDF/numpy): pip install sentence-transformers faiss-cpu")
if any("stub" in r["llm"]["model"] for r in log["runs"].values()):
    warn.append("LLM = stub (JO LLM): vendosni GROQ_API_KEY ose LLM_PROVIDER=ollama dhe riekzekutoni.")
if not ok_test:
    warn.append("Disa teste dështuan.")
for w in warn:
    print("  KUJDES:", w)
if not warn:
    print("  Gjithçka u ekzekutua me burime reale, model embeddings, FAISS dhe LLM real.")
print("\nMbetet për ekipin: kontrolli manual (Evis), licencat 'për verifikim' (Fjorela),")
print("demonstrimi dhe vendimi go/no-go (Gerald + të gjithë). Shih docs/02_Backlog_Rolet_Java02.docx.")
