"""Ndërton dhe ekzekuton notebook-ët e javës 2 (pa nevojë për Jupyter).
Çdo qelizë ekzekutohet në rend; stdout dhe figurat ruhen si output-e në .ipynb.
    python tools/build_notebooks.py
"""
import base64
import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n")}


def execute(cells, name):
    os.chdir(NB)
    ns = {"__name__": "__main__"}
    n = 0
    for c in cells:
        if c["cell_type"] != "code":
            continue
        n += 1
        c["execution_count"] = n
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(c["source"], f"{name}[{n}]", "exec"), ns)
        except Exception:
            buf.write(traceback.format_exc())
            print(f"GABIM te {name} qeliza {n}:\n{buf.getvalue()}", file=sys.stderr)
        text = buf.getvalue()
        if text:
            c["outputs"].append({"output_type": "stream", "name": "stdout", "text": text})
        for num in plt.get_fignums():
            fig = plt.figure(num)
            b = io.BytesIO()
            fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
            c["outputs"].append({"output_type": "display_data", "metadata": {},
                                 "data": {"image/png": base64.b64encode(b.getvalue()).decode(),
                                          "text/plain": "<Figure>"}})
        plt.close("all")
    for c in cells:  # formati nbformat kërkon listë rreshtash
        c["source"] = [l + "\n" for l in c["source"].split("\n")]
        c["source"][-1] = c["source"][-1].rstrip("\n")
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (NB / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("U ndërtua", name)


SETUP = '''
import sys, json
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path("..").resolve() / "src"))
import minirag as mr
pd.set_option("display.max_colwidth", 90); pd.set_option("display.width", 200)

# Korpusi: REAL06 nëse ekzistojnë abstraktet reale (pas `python src/merr_abstraktet.py`), përndryshe SIM06.
REAL_PATH = mr.ROOT / "data/private/02_Abstraktet_Reale.csv"
CORPUS = "REAL06" if REAL_PATH.exists() else "SIM06"
print("Korpusi aktiv:", CORPUS)
'''

# --------------------------------------------------------------------------- #
nb1 = [
    md("""
# 02 – Indeksi dhe chunk-et (Detyra B)
Projekti 06 • Java 2 • Roli B (Klara Belulaj), zëvendësues Gerald Paja

**Qëllimi:** abstraktet ndahen në fragmente (chunks) me ID të gjurmueshme, krijohen embeddings me model të
gatshëm dhe ndërtohet një indeks vektorial (FAISS). Testohet që një fragment i gjetur hapet si tekst dhe i
përket `source_id` të regjistruar.

> Kufizim: provë teknike mësimore. Pa të dhëna pacientësh, pa diagnozë ose këshillë mjekësore.
"""),
    code(SETUP),
    md("## 1. Ngarkimi i korpusit dhe lidhja me manifestin"),
    code('''
docs = mr.load_real_corpus() if CORPUS == "REAL06" else mr.load_sim_corpus()
print(f"{len(docs)} dokumente")
print(docs[["source_id", "title", "license_status", "shareable"]].to_string(index=False))

manifest = pd.read_csv(mr.ROOT / "data/manifests/02_Korpusi_Pilot_Manifest.csv")
print(f"\\nManifesti real ka {len(manifest)} burime; versioni: {manifest.corpus_version.iloc[0]}")
'''),
    md("""
## 2. Chunking
- **SIM06:** çdo dokument është tashmë një fragment i shkurtër → 1 chunk/dokument.
- **REAL06:** abstrakti ndahet në grupe me 1–2 fjali (maks. ~600 karaktere), pa mbivendosje.

Çdo chunk ruan `chunk_id = <source_id>-Cnn`, `start_offset`/`end_offset` te teksti origjinal dhe versionet e
pastrimit/segmentimit.
"""),
    code('''
chunks = mr.build_chunks(docs, single_chunk=(CORPUS == "SIM06"))
print(f"{len(chunks)} chunk-e  |  versioni: {chunks.chunking_version.iloc[0]}")
print(chunks[["chunk_id", "source_id", "start_offset", "end_offset", "n_chars"]].head(12).to_string(index=False))
'''),
    md("### Kontrolli i gjurmueshmërisë: `text[start:end]` i burimit duhet të jetë identik me chunk-un"),
    code('''
errors = mr.verify_chunk_provenance(chunks, docs)
print("Gabime gjurmueshmërie:", errors if errors else "asnjë ✔")
'''),
    md("## 3. Embeddings dhe indeksi vektorial"),
    code('''
rag = mr.MiniRAG.build(chunks)
print("Embedder:", rag.embedder.backend)
print("Indeksi: ", rag.index.backend, "| vektorë:", len(rag.index), "| dimensioni:", rag.index.dim)
if not rag.embedder.is_semantic:
    print("\\nKUJDES: sentence-transformers nuk është instaluar në këtë mjedis -> u përdor TF-IDF (fallback).")
    print("Në mjedisin e ekipit: pip install -r requirements.txt dhe riekzekutoni notebook-un.")
'''),
    md("## 4. Shembull përputhjeje: një pyetje → fragmenti më i afërt hapet si tekst"),
    code('''
q = ("Cili rezultat është regjistruar në momentin e daljes nga spitali?" if CORPUS == "SIM06"
     else "What was the crude annual incidence of aneurysmal subarachnoid hemorrhage in Norway between 2008 and 2014?")
top = rag.retrieve(q, k=3)
print("Pyetja:", q, "\\n")
for r in top.itertuples():
    print(f"#{r.rank}  {r.chunk_id}  (score {r.retrieval_score})  burimi={r.source_id}")
    print("    ", r.chunk_text[:220] + ("…" if len(r.chunk_text) > 220 else ""))

best = top.iloc[0]
src_text = docs.set_index("source_id").loc[best.source_id, "text"]
row = chunks.set_index("chunk_id").loc[best.chunk_id]
assert src_text[row.start_offset:row.end_offset] == best.chunk_text
print(f"\\n✔ {best.chunk_id} i përket {best.source_id} dhe gjendet te pozicioni {row.start_offset}–{row.end_offset}.")
'''),
    md("## 5. Ruajtja e `02_Chunket_Pilot.csv`\nTeksti i burimeve pa licencë të hapur nuk shkruhet në repository."),
    code('''
mr.upsert_csv(mr.export_chunks(chunks), mr.ROOT / "results/02_Chunket_Pilot.csv", CORPUS)
saved = pd.read_csv(mr.ROOT / "results/02_Chunket_Pilot.csv")
print(saved.groupby("corpus").size().to_string())
'''),
    md("""
## 6. Zgjedhjet teknike dhe kufizimet
| Zgjedhja | Arsyeja | Kufizimi |
|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | model i gatshëm, i vogël, shumëgjuhësh (pyetje shqip ↔ abstrakte anglisht) | jo i specializuar për mjekësi |
| FAISS `IndexFlatIP` | kërkim i saktë (jo i përafërt) për pak vektorë | për korpus të madh duhet indeks i përafërt |
| 1–2 fjali/chunk | fragmente të shkurtra → citime të sakta | konteksti mund të ndahet mes dy chunk-eve |
| TF-IDF (fallback) | punon pa internet | leksikor: nuk kap sinonime dhe nuk kalon gjuhë |
"""),
]

# --------------------------------------------------------------------------- #
nb2 = [
    md("""
# 02 – Mini-RAG: retrieval dhe krahasimi LLM pa/me fragmente (Detyrat C dhe D)
Projekti 06 • Java 2 • Roli B (retrieval/LLM) + Roli C (kontrolli manual)

Rrjedha: **pyetje → top-3 fragmente (FAISS) → (a) LLM pa fragmente = baseline, (b) i njëjti LLM me fragmentet = RAG
→ kontroll automatik i citimit → kontroll manual.**

Modeli dhe parametrat lexohen nga `config/llm_config.json`. Pa çelës API përdoret `stub_extractive`, i cili
**nuk është LLM** dhe shërben vetëm për të provuar tubacionin.
"""),
    code(SETUP + '''
import run_java02 as runner
cfg = mr.load_llm_config()
llm = mr.LLMClient(cfg)
print("LLM:", llm.model_version)
print("Parametrat (të njëjtë për baseline dhe RAG):", json.dumps(llm.settings(), ensure_ascii=False))
'''),
    md("## 1. Pyetjet e provës"),
    code('''
docs, qs, chunks = runner.load(CORPUS)
rag = mr.MiniRAG.build(chunks)
print("Embedder:", rag.embedder.backend, "| Indeksi:", rag.index.backend)
print(qs[["question_id", "question", "expected_behavior"]].to_string(index=False))
'''),
    md("## 2. Retrieval: top-3 fragmente për çdo pyetje, krahasuar me kërkimin me fjalë kyçe"),
    code('''
for q in qs.itertuples():
    top = rag.retrieve(q.question, k=3)
    kw = mr.keyword_search(chunks, q.question, k=3)
    abst, rule = mr.needs_abstention(q.question, top)
    print(f"\\n{q.question_id}: {q.question}")
    for r in top.itertuples():
        mark = "  ← burimi i pritur" if r.source_id == q.expected_source_id else ""
        txt = r.chunk_text if r.shareable else "[tekst vetëm lokal]"
        print(f"   #{r.rank} {r.chunk_id} score={r.retrieval_score:.3f}{mark}\\n       {txt[:160]}")
    print("   fjalë kyçe top-3:", [f"{c} ({n} fjalë)" for c, n in kw])
    print("   abstention:", abst, rule)
'''),
    md("""
**Leximi i rezultatit:** pikëzimi (score) nuk është provë. Rishikuesi lexon tekstin e fragmentit dhe plotëson
kolonën `manual_relevance` te `results/02_Retrieval_Prova.csv` (përkatës / pjesërisht / jo përkatës).
"""),
    md("## 3. Prompt-et: e njëjta pyetje, e njëjta LLM, me dhe pa fragmente"),
    code('''
q0 = qs.iloc[1]
top0 = rag.retrieve(q0.question, k=3)
prompts = mr.build_prompts(q0.question, top0)
print("SYSTEM:", mr.SYSTEM_PROMPT, "\\n")
print("----- BASELINE -----\\n" + prompts["baseline"])
print("\\n----- RAG -----\\n" + runner.redact_prompt(prompts["rag"], chunks))
'''),
    md("## 4. Ekzekutimi i plotë dhe ruajtja e `02_Retrieval_Prova.csv` + `02_Krahasimi_LLM.csv`"),
    code('''
import datetime as dt
run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
log_path = mr.ROOT / "results/02_run_log.json"
log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {"runs": {}}
runner.run(CORPUS, llm, run_id, log)
log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
'''),
    md("## 5. Baseline kundrejt RAG dhe kontrolli automatik i citimit"),
    code('''
cmp = pd.read_csv(mr.ROOT / "results/02_Krahasimi_LLM.csv")
cmp = cmp[cmp.corpus == CORPUS]
for qid, g in cmp.groupby("question_id", sort=False):
    print(f"\\n{qid}: {g.question_text.iloc[0]}")
    for r in g.itertuples():
        print(f"   [{r.variant:8}] {str(r.output)[:230]}")
        print(f"              citim: {r.citation_check}")
'''),
    md("""
## 6. Kontrolli manual (Roli C – Evis Bakiu)
Kontrolli automatik provon vetëm që (1) ID e cituar ishte mes fragmenteve të dërguara dhe (2) teksti në thonjëza
gjendet fjalë për fjalë. **Nuk provon që pohimi është i saktë në kuptim.**

Për të paktën një citim:
1. Hapni `results/02_Krahasimi_LLM.csv`, rreshtin RAG të pyetjes.
2. Hapni fragmentin te `results/02_Chunket_Pilot.csv` sipas `cited_ids` (ose burimin origjinal nga `source_url`).
3. Plotësoni `manual_verdict` = *e mbështetur / pjesërisht / e pambështetur / abstention e saktë* dhe `manual_note`.
4. Për baseline: shënoni çdo pohim ose referencë pa bazë si *e pambështetur*.
"""),
    code('''
pending = cmp[cmp.manual_verdict.isna()][["question_id", "variant", "citation_check"]]
print(f"Rreshta që presin kontroll manual: {len(pending)}")
print(pending.to_string(index=False))
'''),
]

# --------------------------------------------------------------------------- #
nb3 = [
    md("""
# 02 – DAG didaktik (Detyra E)
Projekti 06 • Java 2 • Roli D (Gerald Paja)

Shembull **neutral**, i ndarë nga literatura aSAH (sipas seksionit 10.6 të detyrës):
*Sezoni i verës → Temperatura e lartë → Shitje akulloresh* dhe *Temperatura e lartë → Vizita në plazh*.

> Shigjetat janë **supozime** të modeluesit, jo fakte të nxjerra nga të dhënat. Mungesa e cikleve është kusht
> teknik i një DAG-u, jo provë shkakësie. Ky nuk është DoWhy dhe nuk përdoret për relacione klinike.
"""),
    code('''
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

G = nx.DiGraph()
supozimet = [
    ("Sezoni i verës", "Temperatura e lartë", "S1: vera sjell ditë më të nxehta"),
    ("Temperatura e lartë", "Shitje akulloresh", "S2: kur është nxehtë blihen më shumë akullore"),
    ("Temperatura e lartë", "Vizita në plazh", "S3: kur është nxehtë shkohet më shumë në plazh"),
]
for a, b, s in supozimet:
    G.add_edge(a, b, supozim=s)

print("Nyje:", list(G.nodes))
print("Shigjeta:")
for a, b, d in G.edges(data=True):
    print(f"   {a} → {b}   [{d['supozim']}]")
print("\\nA është DAG (pa cikle)?", nx.is_directed_acyclic_graph(G))
print("Rendi topologjik:", " → ".join(nx.topological_sort(G)))
'''),
    md("## 1. Roja kundër cikleve\nÇdo shigjetë e re shtohet vetëm nëse grafi mbetet pa cikle (si do të bëjë paneli i javës 8)."),
    code('''
def shto_shigjete(G, a, b, supozim):
    G.add_edge(a, b, supozim=supozim)
    if not nx.is_directed_acyclic_graph(G):
        cikli = nx.find_cycle(G)
        G.remove_edge(a, b)
        print(f"REFUZUAR: {a} → {b} krijon ciklin {cikli}")
        return False
    print(f"PRANUAR: {a} → {b}")
    return True

shto_shigjete(G, "Shitje akulloresh", "Temperatura e lartë", "supozim i gabuar: akullorja ngroh motin")
shto_shigjete(G, "Shitje akulloresh", "Vizita në plazh", "supozim i padokumentuar")
G.remove_edge("Shitje akulloresh", "Vizita në plazh")  # e heqim: nuk ka arsyetim për këtë shigjetë
print("Pas rishikimit, DAG:", nx.is_directed_acyclic_graph(G), "| shigjeta:", G.number_of_edges())
'''),
    md("## 2. Vizatimi i grafit"),
    code('''
pos = {"Sezoni i verës": (0, 1), "Temperatura e lartë": (1, 1),
       "Shitje akulloresh": (2, 1.6), "Vizita në plazh": (2, 0.4)}
fig, ax = plt.subplots(figsize=(8, 3.6))
nx.draw_networkx_nodes(G, pos, node_color="#dbe9f6", edgecolors="#2b5d8a", node_size=4200, ax=ax)
nx.draw_networkx_labels(G, pos, labels={n: n.replace(" ", "\\n", 1) for n in G}, font_size=9, ax=ax)
nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=18, node_size=4200, edge_color="#2b5d8a", ax=ax)
ax.annotate("", xy=(2, 0.62), xytext=(2, 1.38),
            arrowprops=dict(arrowstyle="<->", ls="--", color="#b03a2e"))
ax.text(2.08, 1.0, "lëvizin bashkë\\n(asociacion,\\njo shkak)", color="#b03a2e", fontsize=8, va="center")
ax.set_title("DAG didaktik – shembull neutral, jo pretendim klinik", fontsize=10)
ax.set_xlim(-0.4, 2.7); ax.set_ylim(0, 2); ax.axis("off")
fig.savefig(Path("..") / "results/02_DAG_Didaktik.png", dpi=150, bbox_inches="tight")
print("Figura u ruajt: results/02_DAG_Didaktik.png")
'''),
    md("""
## 3. Pse asociacion ≠ shkakësi (simulim i vogël me të dhëna të sajuara)
Krijojmë të dhëna ku **akullorja nuk ndikon fare** te plazhi; të dyja varen vetëm nga temperatura.
Megjithatë ato dalin të korreluara. Kur krahasojmë ditë me temperaturë të ngjashme, lidhja pothuajse zhduket.
Të dhënat janë sintetike dhe shërbejnë vetëm për ilustrim.
"""),
    code('''
rng = np.random.default_rng(42)
n = 2000
temperatura = rng.normal(25, 6, n)
akullore = 2.0 * temperatura + rng.normal(0, 8, n)   # varet vetëm nga temperatura
plazh = 1.5 * temperatura + rng.normal(0, 8, n)      # varet vetëm nga temperatura

r_total = np.corrcoef(akullore, plazh)[0, 1]
mask = (temperatura > 24) & (temperatura < 26)        # "kontrollojmë" confounder-in
r_strata = np.corrcoef(akullore[mask], plazh[mask])[0, 1]
print(f"Korrelacioni akullore–plazh (të gjitha ditët):        r = {r_total:.2f}")
print(f"Korrelacioni brenda ditëve me temperaturë 24–26 °C:  r = {r_strata:.2f}")
print("\\nPërfundimi: lidhja vjen nga confounder-i (temperatura), jo nga akullorja.")
'''),
    md("""
## 4. Supozimet dhe kufijtë e interpretimit
| Kodi | Supozimi | Si mund të jetë i gabuar |
|---|---|---|
| S1 | Sezoni → Temperatura | ka verë me ditë të freskëta; edhe gjeografia ndikon |
| S2 | Temperatura → Akullore | çmimet, festat ose turizmi mund të ndikojnë po aq |
| S3 | Temperatura → Plazh | fundjava dhe pushimet janë faktorë të tjerë që mungojnë në graf |

- Grafi paraqet **hipoteza** të deklaruara. Nuk u vlerësua asnjë efekt shkakësor (DoWhy vjen në javët 9–10).
- Në projekt, çdo shigjetë mbi literaturën aSAH do të kërkojë pasazh burimor dhe **rishikim njerëzor** (java 8).
"""),
]

if __name__ == "__main__":
    execute(nb1, "02_Indeksi_Dhe_Chunket.ipynb")
    execute(nb2, "02_Mini_RAG_Prova.ipynb")
    execute(nb3, "02_DAG_Didaktik.ipynb")
