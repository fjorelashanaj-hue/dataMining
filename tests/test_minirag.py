"""Teste automatike të mini-laboratorit (pytest -q)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import minirag as mr  # noqa: E402
import merr_abstraktet as ma  # noqa: E402

# Tekst TESTI i shkruar për testim njësie (jo abstrakt real, nuk përdoret në rezultate).
FAKE = ("Background: This is a unit-test text, not a real abstract. Methods: We test chunking. "
        "Results: The crude rate was 5.7 per 100,000 person-years. Conclusions: Offsets must match.")


@pytest.fixture(scope="module")
def sim():
    docs = mr.load_sim_corpus()
    chunks = mr.build_chunks(docs, single_chunk=True)
    return docs, chunks, mr.MiniRAG.build(chunks)


def test_sim_dataset_has_8_fragments_and_mapping(sim):
    docs, chunks, _ = sim
    assert len(docs) == 8 and len(chunks) == 8
    assert (chunks["chunk_id"].str.rsplit("-C", n=1).str[0] == chunks["source_id"]).all()


def test_chunk_offsets_trace_back_to_source():
    docs = pd.DataFrame([{"corpus": "REAL06", "source_id": "REAL06-999", "title": "t", "text": FAKE,
                          "source_url": "", "license_status": "test", "shareable": False}])
    chunks = mr.build_chunks(docs)
    assert len(chunks) >= 2
    assert mr.verify_chunk_provenance(chunks, docs) == []
    exported = mr.export_chunks(chunks)
    assert exported["chunk_text"].str.contains("vetëm lokal").all()  # licenca e mbyllur → pa tekst


def test_decimal_numbers_not_split():
    spans = mr._sentence_spans("Incidence was 5.7 per 100,000 (p = 0.007). The second sentence is long enough here.")
    assert len(spans) == 2 and "5.7 per 100,000" in "Incidence was 5.7 per 100,000 (p = 0.007)."


def test_retrieval_returns_expected_fragment_for_all_sim_questions(sim):
    _, _, rag = sim
    qs = pd.read_csv(mr.ROOT / "data/02_Projekti06_Pyetje_Testimi.csv")
    for q in qs.itertuples():
        top = rag.retrieve(q.question, k=3)
        assert len(top) == 3
        assert q.expected_chunk_id in set(top["chunk_id"]), q.question_id


def test_negative_control_not_top1(sim):
    _, _, rag = sim
    top = rag.retrieve("Cila ngjarje e vëzhguar gjatë ndjekjes përmendet?", k=3)
    assert top.iloc[0]["chunk_id"] != "SIM06-008-C01"


def test_abstention_q06_and_not_for_q02(sim):
    _, _, rag = sim
    q06 = "Sa është përqindja e vdekshmërisë brenda 30 ditëve?"
    assert mr.needs_abstention(q06, rag.retrieve(q06))[0]
    q02 = "Cili rezultat është regjistruar në momentin e daljes nga spitali?"
    assert not mr.needs_abstention(q02, rag.retrieve(q02))[0]


def test_abstention_unknown_country():
    top = pd.DataFrame([{"chunk_text": "Incidence in Norway was 5.7 per 100,000.", "retrieval_score": 0.5}])
    assert mr.needs_abstention("What percentage in Albania were coiled?", top)[0]
    assert not mr.needs_abstention("What was the incidence in Norway?", top)[0]


def test_citation_check(sim):
    _, _, rag = sim
    top = rag.retrieve("Cili rezultat është regjistruar në momentin e daljes nga spitali?")
    ok = 'Gjendja funksionale "rezultati i regjistruar është gjendja funksionale" [SIM06-002-C01]'
    assert mr.citation_check(ok, top)["citation_check"].startswith("OK")
    fake_id = 'Diçka "rezultati i regjistruar" [SIM06-008-C01]'
    assert mr.citation_check(fake_id, top)["citation_check"].startswith("ID_JASHTË")
    fake_quote = 'Shifra "mortaliteti ishte 12 përqind" [SIM06-002-C01]'
    assert mr.citation_check(fake_quote, top)["citation_check"].startswith("CITUESI_NUK")
    assert mr.citation_check("Nuk ka evidencë të mjaftueshme.", top)["citation_check"] == "ABSTENTION"
    assert mr.citation_check("Diçka pa referencë.", None)["citation_check"].startswith("PA_CITIM")


def test_same_settings_for_both_variants():
    cfg = mr.load_llm_config()
    llm = mr.LLMClient(cfg)
    s = llm.settings()
    assert s["temperature"] == cfg["temperature"] and s["max_tokens"] == cfg["max_tokens"]


def test_clean_abstract_and_license():
    raw = "<h4>Background</h4>Text &amp; more. <h4>Results</h4>Rate was 5.7 ."
    assert ma.clean_abstract(raw) == "Background: Text & more. Results: Rate was 5.7."
    assert ma.is_open_license("cc by") and ma.is_open_license("CC-BY-NC") and not ma.is_open_license("")
