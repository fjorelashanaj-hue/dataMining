"""
minirag.py – Mini-laborator RAG i Projektit 06 (Java 2)

Rrjedha: abstrakt -> fragment (chunk) -> embedding -> indeks vektorial
         -> retrieval (top-k) -> LLM pa/me fragmente -> kontroll citimi.

Parimet:
  * Çdo fragment mban chunk_id, source_id dhe pozicionin (start/end) te teksti origjinal.
  * Embeddings: sentence-transformers (model shumëgjuhësh i gatshëm). Nëse paketa
    mungon, përdoret TF-IDF me n-grama karakteresh dhe kjo shënohet në log.
  * Indeksi: FAISS (IndexFlatIP). Nëse FAISS mungon, përdoret numpy (rezultat i njëjtë
    për kërkim të saktë "flat") dhe kjo shënohet në log.
  * Baseline dhe RAG përdorin TË NJËJTIN model dhe TË NJËJTAT parametra.
  * Kontrolli automatik i citimit provon vetëm që ID ekziston dhe cituesi gjendet
    fjalë për fjalë; nuk provon kuptimin. Kontrolli manual mbetet i detyrueshëm.

Ky kod është provë mësimore. Nuk jep diagnozë, trajtim ose këshillë mjekësore
dhe nuk vërteton shkakësi.
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ABSTENTION_TEXT = "Nuk ka evidencë të mjaftueshme"
CLEANING_VERSION = "clean_v1"
CHUNKING_VERSION = "sent2_max600_v1"


# --------------------------------------------------------------------------- #
# 1. Ngarkimi i korpuseve
# --------------------------------------------------------------------------- #
def load_sim_corpus(path: Path | str = ROOT / "data/02_Projekti06_Dataset.csv") -> pd.DataFrame:
    """Dataset-i didaktik FIKTIV (SIM06). Çdo rresht është tashmë një fragment."""
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "corpus": "SIM06",
        "source_id": df["source_id"],
        "title": df["title_illustrative"],
        "text": df["fragment_text"],
        "source_url": df["source_url"].fillna(""),
        "license_status": df["license_status"],
        "shareable": True,  # tekst origjinal ilustrues i dokumentit të detyrës
    })
    return out


def load_real_corpus(path: Path | str = ROOT / "data/private/02_Abstraktet_Reale.csv") -> pd.DataFrame:
    """Abstraktet reale të shkarkuara nga src/merr_abstraktet.py (Europe PMC)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} nuk ekziston. Ekzekutoni fillimisht: python src/merr_abstraktet.py"
        )
    df = pd.read_csv(path)
    df = df[df["abstract_text"].fillna("").str.len() > 0].copy()
    return pd.DataFrame({
        "corpus": "REAL06",
        "source_id": df["source_id"],
        "title": df["title"],
        "text": df["abstract_text"],
        "source_url": df["source_url"],
        "license_status": df["license_status"],
        "shareable": df["shareable"].astype(bool),
    })


# --------------------------------------------------------------------------- #
# 2. Chunking me pozicione të gjurmueshme
# --------------------------------------------------------------------------- #
# Fund fjalie: . ! ? të ndjekura nga hapësirë + shkronjë e madhe/kllapë, ose fundi i tekstit.
# Kështu "5.7", "p = 0.001" ose "et al. (2020)" nuk ndahen gabimisht.
_SENT_END = re.compile(r"[.!?]+(?=\s+[A-ZÇË\[(]|\s*$)")


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Kthen (start, end) për çdo fjali, duke ruajtur pozicionet te teksti origjinal."""
    spans, start = [], 0
    for m in _SENT_END.finditer(text):
        spans.append((start, m.end()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    clean = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            clean.append((s, e))
    spans = clean
    # bashko "fjali" shumë të shkurtra (p.sh. "95% CI 1.05-2.23)") me të mëparshmen
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and (e - s) < 25:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def chunk_text(text: str, max_sentences: int = 2, max_chars: int = 600) -> list[tuple[int, int]]:
    """Grupon fjalitë në fragmente me 1–2 fjali (max ~600 karaktere), pa mbivendosje."""
    spans = _sentence_spans(text)
    chunks, cur = [], []
    for s, e in spans:
        if cur and (len(cur) >= max_sentences or (e - cur[0][0]) > max_chars):
            chunks.append((cur[0][0], cur[-1][1]))
            cur = []
        cur.append((s, e))
    if cur:
        chunks.append((cur[0][0], cur[-1][1]))
    return chunks


def build_chunks(corpus: pd.DataFrame, single_chunk: bool = False) -> pd.DataFrame:
    """Krijon tabelën e fragmenteve. single_chunk=True për SIM06 (1 fragment/dokument)."""
    rows = []
    for _, r in corpus.iterrows():
        text = str(r["text"])
        spans = [(0, len(text))] if single_chunk else chunk_text(text)
        for i, (s, e) in enumerate(spans, start=1):
            rows.append({
                "corpus": r["corpus"],
                "chunk_id": f"{r['source_id']}-C{i:02d}",
                "source_id": r["source_id"],
                "chunk_index": i,
                "start_offset": s,
                "end_offset": e,
                "chunk_text": text[s:e],
                "n_chars": e - s,
                "source_url": r["source_url"],
                "license_status": r["license_status"],
                "shareable": bool(r["shareable"]),
                "cleaning_version": CLEANING_VERSION,
                "chunking_version": "1_fragment_per_dokument" if single_chunk else CHUNKING_VERSION,
            })
    return pd.DataFrame(rows)


def verify_chunk_provenance(chunks: pd.DataFrame, corpus: pd.DataFrame) -> list[str]:
    """Kontrollon që teksti i çdo fragmenti = text[start:end] i burimit të tij."""
    texts = dict(zip(corpus["source_id"], corpus["text"]))
    errors = []
    for _, c in chunks.iterrows():
        src = texts.get(c["source_id"])
        if src is None:
            errors.append(f"{c['chunk_id']}: source_id i panjohur")
        elif src[c["start_offset"]:c["end_offset"]] != c["chunk_text"]:
            errors.append(f"{c['chunk_id']}: offset-et nuk përputhen me tekstin origjinal")
    return errors


def export_chunks(chunks: pd.DataFrame) -> pd.DataFrame:
    """Version për repository: teksti fshihet kur licenca nuk lejon rishpërndarje."""
    out = chunks.copy()
    out["chunk_text"] = np.where(out["shareable"], out["chunk_text"],
                                 "[tekst vetëm lokal – licenca për verifikim]")
    return out


# --------------------------------------------------------------------------- #
# 3. Embeddings dhe indeksi vektorial
# --------------------------------------------------------------------------- #
DEFAULT_ST_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder:
    """Model embeddings i gatshëm; fallback TF-IDF kur sentence-transformers mungon."""

    def __init__(self, model_name: str = DEFAULT_ST_MODEL, force_backend: str | None = None):
        self.model_name = model_name
        self.backend = None
        self._model = None
        if force_backend != "tfidf":
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._model = SentenceTransformer(model_name)
                self.backend = f"sentence-transformers:{model_name}"
            except Exception as exc:  # paketa mungon ose modeli s'shkarkohet
                self._fallback_reason = repr(exc)[:200]
        if self._model is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._model = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                          lowercase=True, sublinear_tf=True)
            self.backend = "tfidf-char(3,5) [FALLBACK – jo model semantik]"
        self._fitted = False

    @property
    def is_semantic(self) -> bool:
        return self.backend.startswith("sentence-transformers")

    def fit(self, texts: list[str]) -> "Embedder":
        if not self.is_semantic:
            self._model.fit(texts)
        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.is_semantic:
            v = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(v, dtype="float32")
        v = self._model.transform(texts).toarray().astype("float32")
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return v / n


class VectorIndex:
    """Indeks FAISS IndexFlatIP (ngjashmëri kosinusi mbi vektorë të normalizuar)."""

    def __init__(self, dim: int, force_backend: str | None = None):
        self.dim = dim
        self._ids: list[str] = []
        self._faiss = None
        if force_backend != "numpy":
            try:
                import faiss  # type: ignore
                self._faiss = faiss.IndexFlatIP(dim)
                self.backend = "faiss.IndexFlatIP"
            except Exception:
                pass
        if self._faiss is None:
            self._matrix = np.zeros((0, dim), dtype="float32")
            self.backend = "numpy-flat-ip [FALLBACK – FAISS mungon]"

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if self._faiss is not None:
            self._faiss.add(vectors)
        else:
            self._matrix = np.vstack([self._matrix, vectors])
        self._ids.extend(ids)

    def search(self, qvec: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        qvec = np.ascontiguousarray(qvec.reshape(1, -1), dtype="float32")
        k = min(k, len(self._ids))
        if self._faiss is not None:
            scores, idx = self._faiss.search(qvec, k)
            return [(self._ids[i], float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]
        sims = (self._matrix @ qvec.T).ravel()
        order = np.argsort(-sims, kind="stable")[:k]
        return [(self._ids[i], float(sims[i])) for i in order]

    def __len__(self) -> int:
        return len(self._ids)


@dataclass
class MiniRAG:
    chunks: pd.DataFrame
    embedder: Embedder = field(default=None)
    index: VectorIndex = field(default=None)

    @classmethod
    def build(cls, chunks: pd.DataFrame, force_embedder: str | None = None,
              force_index: str | None = None) -> "MiniRAG":
        texts = chunks["chunk_text"].tolist()
        emb = Embedder(force_backend=force_embedder).fit(texts)
        vecs = emb.encode(texts)
        idx = VectorIndex(vecs.shape[1], force_backend=force_index)
        idx.add(chunks["chunk_id"].tolist(), vecs)
        return cls(chunks=chunks.set_index("chunk_id", drop=False), embedder=emb, index=idx)

    def retrieve(self, question: str, k: int = 3) -> pd.DataFrame:
        hits = self.index.search(self.embedder.encode([question])[0], k=k)
        rows = []
        for rank, (cid, score) in enumerate(hits, start=1):
            c = self.chunks.loc[cid]
            rows.append({"rank": rank, "chunk_id": cid, "source_id": c["source_id"],
                         "source_url": c["source_url"], "retrieval_score": round(score, 4),
                         "chunk_text": c["chunk_text"], "shareable": c["shareable"]})
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. Kërkim me fjalë kyçe (për krahasim me retrieval-in vektorial)
# --------------------------------------------------------------------------- #
_STOP = set("""a e i në me për të që nga se si sa cili cila cilat çfarë është janë ka kanë
the of and in to a an is are was were what which how does do did for with on by from that this
""".split())


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", _norm(s)) if t not in _STOP and len(t) > 2]


def keyword_search(chunks: pd.DataFrame, question: str, k: int = 3) -> list[tuple[str, int]]:
    """Numëron fjalët e pyetjes (pa stopwords) që shfaqen në fragment."""
    q = set(_tokens(question))
    scored = [(cid, len(q & set(_tokens(t)))) for cid, t in zip(chunks["chunk_id"], chunks["chunk_text"])]
    scored.sort(key=lambda x: -x[1])
    return scored[:k]


# --------------------------------------------------------------------------- #
# 5. Abstention (heuristikë e deklaruar, jo gjykim i LLM-së)
# --------------------------------------------------------------------------- #
_NUMERIC_Q = re.compile(r"\b(sa |përqindj|perqindj|numri|shkall|percentage|proportion|how many|"
                        r"rate|incidence|frequency|mortality rate)", re.I)


# rezultat sasior: përqindje, numër dhjetor ose normë "per 100 000"
_QUANT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*%|\d+[.,]\d+|p[eë]r 100", re.I)


def needs_abstention(question: str, retrieved: pd.DataFrame, min_score: float = 0.05) -> tuple[bool, str]:
    """Kthen (True, arsyeja) kur fragmentet e rikthyera nuk mjaftojnë.

    Rregullat (heuristike, të dokumentuara në fjalor):
      R1 pyetja kërkon numër/përqindje, por asnjë fragment nuk raporton rezultat sasior
         (përqindje, numër dhjetor ose normë "për 100 000"; "30 ditë" nuk numërohet);
      R2 pyetja përmend emër të përveçëm (p.sh. vend) që nuk gjendet në fragmente;
      R3 ngjashmëria më e lartë është nën pragun min_score.
    """
    if retrieved.empty:
        return True, "R0: asnjë fragment"
    joined = " ".join(retrieved["chunk_text"])
    if _NUMERIC_Q.search(question) and not _QUANT_RE.search(joined):
        return True, "R1: kërkohet shifër/përqindje, fragmentet nuk raportojnë asnjë"
    words = re.findall(r"\b[A-ZÇË][a-zçë]{3,}\b", question)[1:]  # anashkalo fjalën e parë
    missing = [w for w in words if _norm(w) not in _norm(joined)]
    if missing:
        return True, f"R2: emri '{missing[0]}' nuk gjendet në fragmente"
    if retrieved["retrieval_score"].max() < min_score:
        return True, f"R3: ngjashmëria maksimale < {min_score}"
    return False, ""


# --------------------------------------------------------------------------- #
# 6. Prompt-et (të fiksuara dhe të versionuara)
# --------------------------------------------------------------------------- #
PROMPT_VERSION = "p02_v1"

SYSTEM_PROMPT = (
    "Je asistent kërkimor për literaturë shkencore (provë akademike). "
    "Nuk jep diagnozë, trajtim apo këshillë mjekësore. "
    "Nuk e paraqet asociacionin si shkakësi. Përgjigju shqip, shkurt (maksimumi 3 fjali)."
)

BASELINE_TEMPLATE = (
    "Pyetja: {question}\n\n"
    "Nëse nuk e di përgjigjen me siguri, shkruaj: \"" + ABSTENTION_TEXT + "\"."
)

RAG_TEMPLATE = (
    "Përgjigju VETËM duke u mbështetur te fragmentet më poshtë.\n"
    "Rregulla:\n"
    "1. Pas çdo pohimi shëno ID-në e fragmentit në kllapa katrore, p.sh. [{example_id}].\n"
    "2. Për çdo pohim jep një citim të shkurtër fjalë për fjalë nga fragmenti, brenda thonjëzave \"...\", "
    "në gjuhën origjinale të fragmentit.\n"
    "3. Nëse fragmentet nuk e përmbajnë përgjigjen, shkruaj vetëm: \"" + ABSTENTION_TEXT + "\".\n"
    "4. Mos shto informacion që nuk është në fragmente.\n\n"
    "Fragmentet:\n{context}\n\n"
    "Pyetja: {question}"
)


def build_context(retrieved: pd.DataFrame) -> str:
    return "\n".join(f"[{r.chunk_id}] {r.chunk_text}" for r in retrieved.itertuples())


def build_prompts(question: str, retrieved: pd.DataFrame) -> dict[str, str]:
    ex = retrieved["chunk_id"].iloc[0] if not retrieved.empty else "ID"
    return {
        "baseline": BASELINE_TEMPLATE.format(question=question),
        "rag": RAG_TEMPLATE.format(question=question, context=build_context(retrieved), example_id=ex),
    }


# --------------------------------------------------------------------------- #
# 7. Klienti LLM (API e përputhshme me OpenAI: OpenAI, Groq, Ollama, LM Studio...)
# --------------------------------------------------------------------------- #
def load_llm_config(path: Path | str = ROOT / "config/llm_config.json") -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    preset = os.environ.get("LLM_PROVIDER", cfg.get("active_provider", "stub"))
    p = dict(cfg["providers"].get(preset, {}))
    p["provider"] = preset
    p.update(cfg["generation"])
    return p


class LLMClient:
    """Thërret /chat/completions. Pa çelës/server → modaliteti stub_extractive (JO LLM)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.api_key = os.environ.get(cfg.get("api_key_env", ""), "") if cfg.get("api_key_env") else ""
        self.is_stub = cfg["provider"] == "stub" or (cfg.get("api_key_env") and not self.api_key)
        self.model_version = "stub_extractive (JO LLM)" if self.is_stub else cfg["model"]

    def settings(self) -> dict:
        keys = ("temperature", "max_tokens", "seed", "top_p")
        prov = self.cfg["provider"]
        if self.is_stub and prov != "stub":
            prov = f"stub (mungon {self.cfg.get('api_key_env')})"
        return {"provider": prov, "model": self.model_version,
                **{k: self.cfg.get(k) for k in keys}}

    def chat(self, user_prompt: str, variant: str, question: str = "",
             retrieved: pd.DataFrame | None = None) -> dict:
        t0 = time.time()
        if self.is_stub:
            text = stub_extractive(variant, question, retrieved)
            return {"text": text, "latency_s": 0.0, "prompt_tokens": None,
                    "completion_tokens": None, "error": ""}
        import requests
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.cfg["model"],
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user_prompt}],
            "temperature": self.cfg["temperature"],
            "max_tokens": self.cfg["max_tokens"],
            "top_p": self.cfg.get("top_p", 1.0),
        }
        if self.cfg.get("seed") is not None:
            body["seed"] = self.cfg["seed"]
        try:
            r = requests.post(self.cfg["base_url"].rstrip("/") + "/chat/completions",
                              headers=headers, json=body, timeout=120)
            r.raise_for_status()
            data = r.json()
            usage = data.get("usage", {}) or {}
            return {"text": data["choices"][0]["message"]["content"].strip(),
                    "latency_s": round(time.time() - t0, 2),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "model_returned": data.get("model", ""), "error": ""}
        except Exception as exc:
            return {"text": "", "latency_s": round(time.time() - t0, 2), "prompt_tokens": None,
                    "completion_tokens": None, "error": repr(exc)[:300]}


def stub_extractive(variant: str, question: str, retrieved: pd.DataFrame | None) -> str:
    """NUK është LLM. Kontroll i tubacionit: kthen fjalinë më të përshtatshme të fragmentit
    të parë me citim, ose abstention sipas heuristikës. Baseline-i i stub-it nuk ka përgjigje."""
    if variant == "baseline":
        return "[stub] Baseline-i kërkon LLM real; pa fragmente stub-i nuk ka çfarë të thotë."
    abstain, _ = needs_abstention(question, retrieved)
    if abstain:
        return ABSTENTION_TEXT + "."
    top = retrieved.iloc[0]
    q = set(_tokens(question))
    sents = [top["chunk_text"][s:e] for s, e in _sentence_spans(top["chunk_text"])]
    best = max(sents, key=lambda s: len(q & set(_tokens(s))))
    return f"Sipas fragmentit: \"{best}\" [{top['chunk_id']}]"


# --------------------------------------------------------------------------- #
# 8. Kontrolli automatik i citimit
# --------------------------------------------------------------------------- #
_ID_RE = re.compile(r"\[((?:SIM06|REAL06)-\d{3}-C\d{2})\]")
_QUOTE_RE = re.compile(r"[\"“”„«»]([^\"“”„«»]{8,})[\"“”„«»]")


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", _norm(s)).strip(" .,:;")


def is_abstention(answer: str) -> bool:
    return _norm(ABSTENTION_TEXT) in _norm(answer)


def citation_check(answer: str, sent_chunks: pd.DataFrame | None) -> dict:
    """Kontroll automatik: (1) ID e cituar ishte mes fragmenteve të dërguara;
    (2) teksti brenda thonjëzave gjendet fjalë për fjalë në fragmentet e cituara."""
    ids = _ID_RE.findall(answer or "")
    quotes = _QUOTE_RE.findall(answer or "")
    res = {"cited_ids": ";".join(dict.fromkeys(ids)), "n_quotes": len(quotes),
           "ids_valid": None, "quotes_found": None, "citation_check": ""}
    if not answer:
        res["citation_check"] = "PA_PËRGJIGJE"
        return res
    if is_abstention(answer) and not ids:
        res["citation_check"] = "ABSTENTION"
        return res
    if not ids:
        res["citation_check"] = "PA_CITIM (pohim i pambështetur, kontrollo manualisht)"
        return res
    sent = {} if sent_chunks is None else dict(zip(sent_chunks["chunk_id"], sent_chunks["chunk_text"]))
    bad = [i for i in ids if i not in sent]
    res["ids_valid"] = not bad
    cited_text = _ws(" ".join(sent.get(i, "") for i in ids))
    found = [q for q in quotes if _ws(q) in cited_text]
    res["quotes_found"] = f"{len(found)}/{len(quotes)}"
    if bad:
        res["citation_check"] = f"ID_JASHTË_KONTEKSTIT: {','.join(bad)}"
    elif quotes and len(found) < len(quotes):
        res["citation_check"] = "CITUESI_NUK_GJENDET_FJALË_PËR_FJALË"
    elif not quotes:
        res["citation_check"] = "ID_OK_PA_CITUES (kontrollo manualisht)"
    else:
        res["citation_check"] = "OK_AUTOMATIK (kërkon ende kontroll manual)"
    return res


# --------------------------------------------------------------------------- #
# 9. Ndihmës për ruajtjen e rezultateve (zëvendëson vetëm rreshtat e të njëjtit korpus)
# --------------------------------------------------------------------------- #
def upsert_csv(df: pd.DataFrame, path: Path | str, corpus: str) -> None:
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path)
        old = old[old["corpus"] != corpus]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8")
