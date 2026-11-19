"""
run_java02.py – Ekzekuton gjithë mini-provën e javës 2 dhe ruan rezultatet.

    python src/run_java02.py --corpus sim     # dataset-i didaktik SIM06 (kontroll tubacioni)
    python src/run_java02.py --corpus real    # 5–10 abstraktet reale (pas merr_abstraktet.py)
    python src/run_java02.py --corpus both

Ofruesi LLM zgjidhet te config/llm_config.json ose me LLM_PROVIDER=groq|openai|ollama|stub.

Prodhon (në results/):
    02_Chunket_Pilot.csv      chunk_id, source_id, pozicioni, prejardhja
    02_Retrieval_Prova.csv    top-3 fragmente për çdo pyetje + krahasim me fjalë kyçe
    02_Krahasimi_LLM.csv      baseline vs RAG, prompt, parametra, kontroll citimi
    02_run_log.json           backend-et, versionet, parametrat, koha
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minirag as mr  # noqa: E402

ROOT = mr.ROOT
RES = ROOT / "results"
TOP_K = 3


def load(corpus: str):
    if corpus == "SIM06":
        docs = mr.load_sim_corpus()
        qs = pd.read_csv(ROOT / "data/02_Projekti06_Pyetje_Testimi.csv")
        qs["expected_source_id"] = qs["expected_chunk_id"].str.rsplit("-C", n=1).str[0]
        chunks = mr.build_chunks(docs, single_chunk=True)
    else:
        docs = mr.load_real_corpus()
        qs = pd.read_csv(ROOT / "data/02_Pyetje_Reale.csv")
        qs["expected_chunk_id"] = ""
        chunks = mr.build_chunks(docs)
    return docs, qs, chunks


def redact_prompt(prompt: str, chunks: pd.DataFrame) -> str:
    for r in chunks[~chunks["shareable"]].itertuples():
        prompt = prompt.replace(r.chunk_text, "[tekst vetëm lokal – licenca për verifikim]")
    return prompt


def run(corpus: str, llm: mr.LLMClient, run_id: str, log: dict) -> None:
    docs, qs, chunks = load(corpus)
    errs = mr.verify_chunk_provenance(chunks, docs)
    if errs:
        raise SystemExit("Gabime gjurmueshmërie:\n" + "\n".join(errs))
    rag = mr.MiniRAG.build(chunks)
    mr.upsert_csv(mr.export_chunks(chunks), RES / "02_Chunket_Pilot.csv", corpus)

    ret_rows, llm_rows = [], []
    for q in qs.itertuples():
        top = rag.retrieve(q.question, k=TOP_K)
        kw = mr.keyword_search(chunks, q.question, k=TOP_K)
        kw_rank = {cid: i + 1 for i, (cid, _) in enumerate(kw)}
        abstain, rule = mr.needs_abstention(q.question, top)
        exp_src = q.expected_source_id if isinstance(q.expected_source_id, str) else ""
        for r in top.itertuples():
            ret_rows.append({
                "corpus": corpus, "run_id": run_id, "question_id": q.question_id,
                "question_text": q.question, "rank": r.rank, "chunk_id": r.chunk_id,
                "source_id": r.source_id, "source_url": r.source_url,
                "retrieval_score": r.retrieval_score,
                "excerpt": r.chunk_text if r.shareable else "[tekst vetëm lokal – licenca për verifikim]",
                "keyword_rank": kw_rank.get(r.chunk_id, ""),
                "is_expected_source": bool(exp_src) and r.source_id == exp_src,
                "abstention_flag": abstain, "abstention_rule": rule,
                "embedder": rag.embedder.backend, "index": rag.index.backend,
                "manual_relevance": "", "manual_note": "",
            })
        prompts = mr.build_prompts(q.question, top)
        for variant in ("baseline", "rag"):
            out = llm.chat(prompts[variant], variant, question=q.question, retrieved=top)
            sent = top if variant == "rag" else None
            chk = mr.citation_check(out["text"], sent)
            if llm.is_stub and variant == "baseline":
                chk["citation_check"] = "N/A – baseline kërkon LLM real (stub)"
            llm_rows.append({
                "corpus": corpus, "run_id": run_id, "question_id": q.question_id,
                "question_text": q.question, "prompt_id": f"{mr.PROMPT_VERSION}_{variant}",
                "variant": "baseline" if variant == "baseline" else "RAG",
                "model_version": llm.model_version, "settings": json.dumps(llm.settings()),
                "sent_chunk_ids": ";".join(top["chunk_id"]) if variant == "rag" else "",
                "prompt_text": redact_prompt(prompts[variant], chunks),
                "output": out["text"], "error": out["error"],
                "latency_s": out["latency_s"], "prompt_tokens": out["prompt_tokens"],
                "completion_tokens": out["completion_tokens"],
                "heuristic_abstention": f"{abstain} {rule}".strip(),
                "is_abstention": mr.is_abstention(out["text"]),
                **chk,
                "expected_answer": q.expected_answer, "expected_behavior": q.expected_behavior,
                "manual_check_instruction": q.manual_check,
                "manual_verdict": "", "manual_note": "",
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            })
        hit = top["source_id"].eq(exp_src).any() if exp_src else None
        print(f"  {q.question_id}: top-1={top.iloc[0].chunk_id} (score {top.iloc[0].retrieval_score})"
              f"  burimi i pritur në top-3: {hit}  abstention: {abstain} {rule}")

    mr.upsert_csv(pd.DataFrame(ret_rows), RES / "02_Retrieval_Prova.csv", corpus)
    mr.upsert_csv(pd.DataFrame(llm_rows), RES / "02_Krahasimi_LLM.csv", corpus)
    log["runs"][corpus] = {
        "run_id": run_id, "n_documents": len(docs), "n_chunks": len(chunks),
        "n_questions": len(qs), "embedder": rag.embedder.backend, "index": rag.index.backend,
        "chunking_version": chunks["chunking_version"].iloc[0], "top_k": TOP_K,
        "prompt_version": mr.PROMPT_VERSION, "llm": llm.settings(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["sim", "real", "both"], default="sim")
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    llm = mr.LLMClient(mr.load_llm_config())
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = RES / "02_run_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {"runs": {}}
    print(f"LLM: {llm.model_version}  |  run_id: {run_id}")
    if llm.is_stub:
        print("  KUJDES: po përdoret stub_extractive (JO LLM). Vendosni çelësin API për provë reale.")
    corpora = {"sim": ["SIM06"], "real": ["REAL06"], "both": ["SIM06", "REAL06"]}[args.corpus]
    for c in corpora:
        print(f"\n== {c} ==")
        run(c, llm, run_id, log)
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nRezultatet u ruajtën në results/.")


if __name__ == "__main__":
    main()
