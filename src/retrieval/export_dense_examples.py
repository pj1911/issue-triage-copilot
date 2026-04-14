"""
Export comparison examples: dense wins, dense failures, both fail.

Buckets (classified by R@10):
    dense_win     : dense R@10 > 0  AND lexical R@10 == 0  (10 examples)
    dense_fail    : lexical R@10 > 0 AND dense R@10 == 0   (10 examples)
    both_fail     : both R@10 == 0                         (5 examples)

For each selected query we re-run both retrievers and show the top-5
retrieved docs for each so the difference is directly readable.

Requires on HPC:
    data/processed/retrieval_corpus.parquet
    data/processed/retrieval_eval.json
    models/tfidf_retrieval_vectorizer.joblib   (lexical)
    models/tfidf_retrieval_matrix.npz          (lexical)
    models/dense_doc_embeddings.npy            (dense)
    models/dense_doc_ids.npy                   (dense)
    models/dense_model_config.json             (dense)
    reports/retrieval_metrics.json             (Day 5 per-query)
    reports/dense_retrieval_metrics.json       (Day 6 per-query)

Output:
    reports/dense_examples.json
    reports/dense_examples.txt

Run from project root:
    python -m src.retrieval.export_dense_examples
"""
import json
import textwrap

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from src.retrieval.dense_retriever import (
    MODEL_NAME, _get_device, encode_texts, DenseRetriever,
)
from src.utils.paths import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR

# ── paths ──────────────────────────────────────────────────────────────────────
EVAL_PATH          = PROCESSED_DIR / "retrieval_eval.json"
CORPUS_PATH        = PROCESSED_DIR / "retrieval_corpus.parquet"
LEX_METRICS_PATH   = REPORTS_DIR   / "retrieval_metrics.json"
DENSE_METRICS_PATH = REPORTS_DIR   / "dense_retrieval_metrics.json"
VEC_PATH           = MODELS_DIR    / "tfidf_retrieval_vectorizer.joblib"
MATRIX_PATH        = MODELS_DIR    / "tfidf_retrieval_matrix.npz"
JSON_OUT           = REPORTS_DIR   / "dense_examples.json"
TXT_OUT            = REPORTS_DIR   / "dense_examples.txt"

N_WIN  = 10   # dense wins to export
N_FAIL = 10   # dense failures (lexical wins) to export
N_BOTH = 5    # both-fail cases to export
TOP_K  = 5    # retrieved docs shown per method per example


# ── bucket classification ──────────────────────────────────────────────────────

def classify(lex_rows: list[dict], dense_rows: list[dict]) -> dict[str, list[dict]]:
    """
    Join per-query results and assign each query to exactly one bucket.
    Returns dict with keys: dense_win, dense_fail, both_fail, both_hit.
    Rows include both lex_ and dense_ prefixed metrics.
    """
    lex_map   = {r["query_id"]: r for r in lex_rows}
    dense_map = {r["query_id"]: r for r in dense_rows}

    all_ids = set(lex_map) | set(dense_map)
    buckets: dict[str, list[dict]] = {
        "dense_win":  [],
        "dense_fail": [],
        "both_fail":  [],
        "both_hit":   [],
    }

    for qid in all_ids:
        lr = lex_map.get(qid,   {"recall@10": 0, "rr": 0, "query_repo": "?"})
        dr = dense_map.get(qid, {"recall@10": 0, "rr": 0, "query_repo": "?"})

        row = {
            "query_id":      qid,
            "query_repo":    lr.get("query_repo", dr.get("query_repo", "?")),
            "split":         lr.get("split",      dr.get("split", "?")),
            "labels":        lr.get("labels",     dr.get("labels", [])),
            "lex_r10":       lr["recall@10"],
            "lex_mrr":       lr["rr"],
            "dense_r10":     dr["recall@10"],
            "dense_mrr":     dr["rr"],
            "mrr_delta":     dr["rr"] - lr["rr"],
        }

        lex_hit   = lr["recall@10"] > 0
        dense_hit = dr["recall@10"] > 0

        if dense_hit and not lex_hit:
            buckets["dense_win"].append(row)
        elif lex_hit and not dense_hit:
            buckets["dense_fail"].append(row)
        elif not lex_hit and not dense_hit:
            buckets["both_fail"].append(row)
        else:
            buckets["both_hit"].append(row)

    # rank within each bucket
    buckets["dense_win"]  = sorted(buckets["dense_win"],
                                   key=lambda r: r["dense_mrr"], reverse=True)[:N_WIN]
    buckets["dense_fail"] = sorted(buckets["dense_fail"],
                                   key=lambda r: r["lex_mrr"],   reverse=True)[:N_FAIL]
    buckets["both_fail"]  = sorted(buckets["both_fail"],
                                   key=lambda r: r["query_repo"])[:N_BOTH]

    return buckets


# ── retrieval helpers ──────────────────────────────────────────────────────────

def lex_top_k(query: str, matrix: sp.csr_matrix, vectorizer, k: int):
    q_vec  = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, matrix).ravel()
    idx    = np.argpartition(scores, -k)[-k:]
    idx    = idx[np.argsort(scores[idx])[::-1]]
    return idx, scores[idx]


def dense_top_k(query: str, embeddings: np.ndarray,
                tokenizer, model, device, k: int):
    q_emb  = encode_texts([query], tokenizer, model, device,
                          batch_size=1, show_progress=False)
    scores = (embeddings @ q_emb.T).ravel()
    idx    = np.argpartition(scores, -k)[-k:]
    idx    = idx[np.argsort(scores[idx])[::-1]]
    return idx, scores[idx]


def format_hits(indices, scores, corpus: pd.DataFrame,
                gold_ids: set, k: int) -> list[dict]:
    hits = []
    for rank, (idx, score) in enumerate(zip(indices[:k], scores[:k]), 1):
        row = corpus.iloc[int(idx)]
        hits.append({
            "rank":        rank,
            "score":       round(float(score), 4),
            "is_gold":     int(row["id"]) in gold_ids,
            "corpus_id":   int(row["id"]),
            "corpus_repo": row["repo"],
            "labels":      list(row["labels_clean"]),
            "snippet":     row["text"][:200],
        })
    return hits


# ── build one example record ───────────────────────────────────────────────────

def build_example(meta: dict, eval_entry: dict,
                  corpus: pd.DataFrame,
                  matrix: sp.csr_matrix, vectorizer,
                  embeddings: np.ndarray, tokenizer, model, device) -> dict:
    query    = eval_entry["query_text"]
    gold_ids = {g["corpus_id"] for g in eval_entry["gold_relevant"]}

    lex_idx, lex_scores     = lex_top_k(query, matrix, vectorizer, TOP_K)
    dense_idx, dense_scores = dense_top_k(query, embeddings, tokenizer, model, device, TOP_K)

    return {
        "query_id":       meta["query_id"],
        "query_repo":     meta["query_repo"],
        "query_split":    meta["split"],
        "query_labels":   meta["labels"],
        "query_snippet":  query[:300],
        "lex_r10":        meta["lex_r10"],
        "lex_mrr":        round(meta["lex_mrr"], 4),
        "dense_r10":      meta["dense_r10"],
        "dense_mrr":      round(meta["dense_mrr"], 4),
        "gold_relevant":  eval_entry["gold_relevant"],
        "lexical_top5":   format_hits(lex_idx, lex_scores, corpus, gold_ids, TOP_K),
        "dense_top5":     format_hits(dense_idx, dense_scores, corpus, gold_ids, TOP_K),
    }


# ── text formatting ────────────────────────────────────────────────────────────

def fmt_example(ex: dict, bucket: str, num: int) -> str:
    sep = "─" * 72
    lines = [
        sep,
        f"[{bucket.upper()} #{num}]  repo={ex['query_repo']}  "
        f"split={ex['query_split']}",
        f"labels={ex['query_labels']}",
        f"Lexical  R@10={ex['lex_r10']:.3f}  MRR={ex['lex_mrr']:.4f}   "
        f"Dense  R@10={ex['dense_r10']:.3f}  MRR={ex['dense_mrr']:.4f}",
        "",
        "QUERY:",
        *textwrap.wrap(ex["query_snippet"].replace("\n", " "), 70),
        "",
        "GOLD RELEVANT:",
    ]
    for g in ex["gold_relevant"]:
        lines.append(
            f"  [{g['corpus_repo']}]  shared={g['shared_labels']}"
        )
        lines.append(
            "  " + g["text_snippet"].splitlines()[0][:70]
        )

    def fmt_hits(hits: list[dict], label: str) -> list[str]:
        out = ["", f"{label} TOP-{TOP_K}:"]
        for r in hits:
            hit = " ← GOLD" if r["is_gold"] else ""
            out.append(
                f"  #{r['rank']} score={r['score']:.4f}  "
                f"[{r['corpus_repo']}]  labels={r['labels']}{hit}"
            )
            out.append("  " + r["snippet"].splitlines()[0][:70])
        return out

    lines += fmt_hits(ex["lexical_top5"], "LEXICAL")
    lines += fmt_hits(ex["dense_top5"],   "DENSE")
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading eval set and metrics …")
    with open(EVAL_PATH)          as f: eval_set     = json.load(f)
    with open(LEX_METRICS_PATH)   as f: lex_metrics  = json.load(f)
    with open(DENSE_METRICS_PATH) as f: dense_metrics = json.load(f)

    eval_map = {e["query_id"]: e for e in eval_set}

    # ── classify all 37 queries ────────────────────────────────────────────
    buckets = classify(lex_metrics["per_query"], dense_metrics["per_query"])

    print(f"  dense_win  : {len(buckets['dense_win'])}")
    print(f"  dense_fail : {len(buckets['dense_fail'])}")
    print(f"  both_fail  : {len(buckets['both_fail'])}")
    print(f"  both_hit   : {len(buckets['both_hit'])}")

    # ── load retrieval indexes ─────────────────────────────────────────────
    print("Loading lexical index …")
    corpus     = pd.read_parquet(CORPUS_PATH)
    vectorizer = joblib.load(VEC_PATH)
    matrix     = sp.load_npz(str(MATRIX_PATH))

    print("Loading dense index …")
    retriever  = DenseRetriever.load()
    embeddings = retriever.embeddings

    device    = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    print(f"Device: {device}")

    # ── build examples ─────────────────────────────────────────────────────
    output: dict[str, list] = {}
    for bucket_name, metas in [
        ("dense_win",  buckets["dense_win"]),
        ("dense_fail", buckets["dense_fail"]),
        ("both_fail",  buckets["both_fail"]),
    ]:
        print(f"Building {len(metas)} {bucket_name} examples …")
        examples = []
        for meta in metas:
            entry = eval_map[meta["query_id"]]
            ex = build_example(meta, entry, corpus,
                               matrix, vectorizer,
                               embeddings, tokenizer, model, device)
            examples.append(ex)
        output[bucket_name] = examples

    # ── save JSON ──────────────────────────────────────────────────────────
    with open(JSON_OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved JSON → {JSON_OUT}")

    # ── save TXT ───────────────────────────────────────────────────────────
    header = (
        "DENSE vs LEXICAL — RETRIEVAL COMPARISON EXAMPLES\n"
        "BAAI/bge-small-en-v1.5  vs  TF-IDF + Cosine\n"
        "Eval set: 37 queries | 3 gold docs/query\n"
    )
    sections = [header, "=" * 72]

    bucket_labels = [
        ("dense_win",  "DENSE WINS  (dense hit, lexical missed)"),
        ("dense_fail", "DENSE FAILURES  (lexical hit, dense missed)"),
        ("both_fail",  "BOTH FAIL  (neither retriever found gold in top-10)"),
    ]
    for key, title in bucket_labels:
        sections.append(f"\n{'=' * 72}")
        sections.append(f"{title:^72}")
        sections.append("=" * 72)
        for i, ex in enumerate(output[key], 1):
            sections.append(fmt_example(ex, key, i))

    TXT_OUT.write_text("\n".join(sections) + "\n")
    print(f"Saved TXT  → {TXT_OUT}")

    # ── console summary ────────────────────────────────────────────────────
    for key, title in bucket_labels:
        print(f"\n── {title} ──")
        for ex in output[key][:3]:
            print(f"  [{ex['query_repo']:<12}] lex_mrr={ex['lex_mrr']:.3f}  "
                  f"dense_mrr={ex['dense_mrr']:.3f}  labels={ex['query_labels']}")


if __name__ == "__main__":
    main()
