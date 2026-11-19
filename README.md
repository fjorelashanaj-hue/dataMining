# Projekti 06 – Studim pilot Causal AI për literaturën cerebrovaskulare (aSAH)

**Java 2: Mini-laborator RAG dhe vendimi teknik go/no-go**

Asistent kërkimor demonstrues. Ai gjen fragmente nga abstrakte shkencore për aSAH (*aneurysmal subarachnoid hemorrhage*) dhe krahason përgjigjen e të njëjtit LLM në dy kushte: pa fragmente (baseline) dhe me fragmente të rikthyera (mini-RAG). Çdo citim kontrollohet në fragmentin origjinal.

Ekipi: Fjorela Shanaj (A – burimet/licencat), Klara Belulaj (B – retrieval/RAG/baseline), Evis Bakiu (C – vlerësimi/gold set), Gerald Paja (D – DAG/paneli/DoWhy).

## Kufizimet etike (të detyrueshme)
- Puna mbështetet vetëm te literatura shkencore. Nuk përdoren të dhëna pacientësh dhe nuk jepet diagnozë, trajtim ose këshillë mjekësore.
- Asociacioni nuk trajtohet si shkakësi. DAG-u është shembull didaktik dhe nuk përbën pretendim klinik.
- Tekstet e plota nuk shkarkohen. Abstraktet ruhen vetëm lokalisht (`data/private/`, në `.gitignore`). Në repository publikohet vetëm teksti i burimeve me licencë CC BY/CC0.
- Dataset-i **SIM06** është fiktiv. Shërben vetëm për të testuar tubacionin dhe nuk numërohet si korpus real.

## Struktura
```
config/llm_config.json            ofruesi LLM + parametrat e përbashkët (temperature=0, max_tokens=300, seed=42)
data/02_Projekti06_Dataset.csv    SIM06 – 8 fragmente fiktive (nga detyra e javës 2)
data/02_Projekti06_Pyetje_Testimi.csv   Q01–Q06 për SIM06
data/02_Pyetje_Reale.csv          Q-REAL-01..04 për korpusin real (përfshirë 1 abstention)
data/manifests/01_Regjistri_Burimeve.csv       regjistri i javës 1
data/manifests/02_Korpusi_Pilot_Manifest.csv   8 burime reale REAL06 me ID, URL, licencë, version
data/private/                     abstraktet reale + JSON bruto i API-së (NUK ngarkohet në Git)
src/minirag.py                    chunking, embeddings, FAISS, retrieval, abstention, prompt-e, LLM, kontroll citimi
src/merr_abstraktet.py            shkarkon abstraktet nga Europe PMC dhe verifikon manifestin
src/run_java02.py                 ekzekuton gjithë provën dhe shkruan results/
notebooks/02_Indeksi_Dhe_Chunket.ipynb   Detyra B
notebooks/02_Mini_RAG_Prova.ipynb        Detyrat C dhe D
notebooks/02_DAG_Didaktik.ipynb          Detyra E
results/                          02_Chunket_Pilot.csv, 02_Retrieval_Prova.csv, 02_Krahasimi_LLM.csv, 02_DAG_Didaktik.png, 02_run_log.json
tests/test_minirag.py             10 teste (pytest)
tools/build_notebooks.py          rindërton dhe ekzekuton notebook-ët pa Jupyter
docs/02_Fjalorth_Dhe_Go_NoGo.docx fjalori, kufizimet, gjendja dhe vendimi i portës 1
docs/02_Backlog_Rolet_Java02.docx  detyrat e javës 2 me emra, statusi dhe demonstrimi
ekzekuto_java02.py                gjithë java 2 me një komandë
```

## Instalimi
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Në ekzekutimin e parë, modeli `paraphrase-multilingual-MiniLM-L12-v2` (~470 MB) shkarkohet automatikisht. Ai është shumëgjuhësh, prandaj pyetjet shqip funksionojnë edhe mbi abstraktet në anglisht.

## Ekzekutimi me një komandë
```bash
pip install -r requirements.txt
export GROQ_API_KEY="..."          # Windows PowerShell: $env:GROQ_API_KEY="..."
python ekzekuto_java02.py
```
Në fund del një përmbledhje. Nëse ka rreshta **KUJDES** (p.sh. LLM stub, fallback, mungojnë abstraktet), porta go/no-go nuk kalohet ende.

## Kush bën çfarë në javën 2
| Detyra | Përgjegjësi | Zëvendësuesi |
|---|---|---|
| T08 Shkarkimi i abstrakteve reale (`python ekzekuto_java02.py`) | Fjorela Shanaj | Klara Belulaj |
| T02 Kontrolli i licencave “për verifikim” te manifesti | Klara Belulaj | Fjorela Shanaj |
| T03 Kontrolli i chunk-eve reale | Evis Bakiu | Gerald Paja |
| T04 Notebook embeddings + FAISS (riekzekutim me modelin real) | Gerald Paja | Evis Bakiu |
| T05 Baseline vs RAG me LLM real | Klara Belulaj | Gerald Paja |
| T06 DAG didaktik + shpjegimi në demo | Fjorela Shanaj | Evis Bakiu |
| T09 Kontrolli manual i fragmenteve dhe citimeve | Evis Bakiu | Klara Belulaj |
| T10 Demonstrimi dhe vendimi go/no-go | Gerald Paja | Fjorela Shanaj |
| T07 Rishikim kryq dhe integrimi | Të gjithë | Fjorela Shanaj |

Detajet: `docs/02_Backlog_Rolet_Java02.docx` dhe `docs/02_Backlog_Rolet.csv`.

## Ekzekutimi hap pas hapi (alternativë)
```bash
# 1. Shkarko 8 abstraktet reale nga Europe PMC dhe verifiko manifestin (Roli A)
python src/merr_abstraktet.py

# 2. Vendos çelësin e LLM-së (shembull me Groq, falas). Mos e shkruaj çelësin në kod.
export GROQ_API_KEY="..."          # Windows PowerShell: $env:GROQ_API_KEY="..."
#    ose: export LLM_PROVIDER=ollama   (model lokal, pa çelës; "ollama pull llama3.1:8b")
#    ose: export LLM_PROVIDER=openai OPENAI_API_KEY="..."

# 3. Ekzekuto retrieval + baseline + RAG mbi SIM06 dhe korpusin real (Roli B)
python src/run_java02.py --corpus both

# 4. Rindërto notebook-ët me rezultatet e reja (ose hapi me `jupyter notebook`)
python tools/build_notebooks.py

# 5. Testet
pytest -q
```
Pa çelës API, programi kalon në `stub_extractive`. Ky **nuk është LLM**: kthen fjalinë më të përshtatshme të fragmentit ose abstention. Shërben vetëm për të provuar tubacionin. Krahasimi baseline/RAG kërkon një LLM real.

## Kontrolli manual (Roli C)
1. Te `results/02_Retrieval_Prova.csv`, plotësoni `manual_relevance` (*përkatës / pjesërisht / jo*) për fragmentet e Q-REAL-01 dhe Q-REAL-02.
2. Te `results/02_Krahasimi_LLM.csv`, për të paktën një rresht RAG, hapni fragmentin sipas `cited_ids` dhe burimin nga `source_url`. Pastaj plotësoni `manual_verdict` (*e mbështetur / pjesërisht / e pambështetur / abstention e saktë*) dhe `manual_note`.
3. Për baseline-in, shënoni si *e pambështetur* çdo pohim, shifër ose referencë që nuk vjen nga burimet.

Kontrolli automatik (`citation_check`) provon vetëm dy gjëra: që ID-ja e cituar ishte mes fragmenteve të dërguara, dhe që teksti në thonjëza gjendet fjalë për fjalë në fragment. Ai **nuk** provon kuptimin.

## Zgjedhjet teknike
| Hapi | Zgjedhja | Fallback (shënohet në `02_run_log.json`) |
|---|---|---|
| Chunking | 1–2 fjali/chunk, ≤600 karaktere, `start/end_offset` te teksti origjinal | – |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | TF-IDF me n-grama karakteresh |
| Indeksi | FAISS `IndexFlatIP` (kosinus mbi vektorë të normalizuar) | numpy (i njëjti kërkim i saktë) |
| Retrieval | top-k = 3, krahasim me kërkim me fjalë kyçe | – |
| LLM | API e përputhshme me OpenAI (Groq/OpenAI/Ollama), të njëjtat parametra për të dy variantet | `stub_extractive` (JO LLM) |
| Abstention | rregulla heuristike R1–R3 + udhëzim në prompt | – |

## Burimet
[R1] Lewis et al. (2020), RAG. https://arxiv.org/abs/2005.11401 · [R2] Europe PMC RESTful Web Service. https://europepmc.org/RestfulWebService · [R4] Hoh et al. (2023), Guideline for aSAH. doi:10.1161/STR.0000000000000436. Burimet e korpusit pilot janë te `data/manifests/02_Korpusi_Pilot_Manifest.csv`.
