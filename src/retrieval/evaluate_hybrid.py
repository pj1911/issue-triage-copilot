"""
Day 7 evaluation: compare Lexical, Dense, and Hybrid (RRF) retrieval
on the human-annotated eval set (retrieval_eval_human.json).

Metrics per query, then averaged:
  Recall@k  — fraction of gold relevant docs found in top-k results
  MRR       — mean reciprocal rank of the first gold hit

Run from project root:
    python -m src.retrieval.evaluate_hybrid

Output:
    reports/hybrid_metrics.json
"""
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.dense_retriever import DenseRetriever, _get_device, encode_texts, MODEL_NAME
from src.retrieval.hybrid_retriever import HybridRetriever, RRF_K, RETRIEVAL_POOL
from src.retrieval.lexical_retriever import LexicalRetriever
from src.utils.paths import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from transformers import AutoModel, AutoTokenizer

EVAL_PATH   = PROCESSED_DIR / "retrieval_eval_human.json"
OUT_PATH    = REPORTS_DIR   / "hybrid_metrics.json"

KS = [5, 10, 20]
POOL = RETRIEVAL_POOL   # top-20 from each method before RRF


# ── metric helpers ─────────────────────────────────────────────────────────────

def recall_at_k(gold_ids: set[int], ranked_ids: list[int], k: int) -> float:
    if not gold_ids:
        return 0.0
    hits = sum(1 for cid in ranked_ids[:k] if cid in gold_ids)
    return hits / len(gold_ids)


def reciprocal_rank(gold_ids: set[int], ranked_ids: list[int]) -> float:
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in gold_ids:
            return 1.0 / rank
    return 0.0


# ── per-method retrieve → ranked corpus_id list ────────────────────────────────

def lex_ranked_ids(
    query_text: str,
    vectorizer,
    matrix: sp.csr_matrix,
    corpus: pd.DataFrame,
    k: int,
) -> list[int]:
    q_vec  = vectorizer.transform([query_text])
    scores = cosine_similarity(q_vec, matrix).ravel()
    top    = np.argpartition(scores, -k)[-k:]
    top    = top[np.argsort(scores[top])[::-1]]
    return corpus.iloc[top]["id"].astype(int).tolist()


def dense_ranked_ids(
    query_text: str,
    dense: DenseRetriever,
    tokenizer,
    model,
    device,
    k: int,
) -> list[int]:
    hits = dense.retrieve(query_text, tokenizer, model, device, k=k)
    return hits["id"].astype(int).tolist()


def hybrid_ranked_ids(
    query_text: str,
    retriever: HybridRetriever,
    k: int,
) -> list[int]:
    hits = retriever.retrieve(query_text, k=k)
    return hits["id"].astype(int).tolist()


# ── aggregate helper ───────────────────────────────────────────────────────────

def aggregate(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    out = {"n": len(df)}
    for k in KS:
        out[f"recall@{k}"] = round(float(df[f"recall@{k}"].mean()), 4)
    out["mrr"] = round(float(df["rr"].mean()), 4)
    return out


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading eval set …")
    with open(EVAL_PATH) as f:
        eval_set = json.load(f)
    print(f"  {len(eval_set)} queries")

    # ── load retrievers ────────────────────────────────────────────────────────
    print("\nLoading lexical index …")
    lex_retriever = LexicalRetriever.load()
    vectorizer    = lex_retriever.vectorizer
    matrix        = lex_retriever.matrix
    corpus        = lex_retriever.corpus

    print("Loading dense index …")
    dense = DenseRetriever.load()
    device    = _get_device()
    print(f"Loading BGE encoder ({MODEL_NAME}) on {device} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    # Wrap into HybridRetriever (reuses the already-loaded objects)
    hybrid = HybridRetriever(
        lex_retriever, dense, tokenizer, model, device,
        pool=POOL, rrf_k=RRF_K,
    )

    # Build id → corpus row index for lexical (parquet order)
    id_to_lex_idx = {int(cid): i for i, cid in enumerate(corpus["id"])}

    max_k = max(KS)

    lex_rows    = []
    dense_rows  = []
    hybrid_rows = []

    print(f"\nEvaluating {len(eval_set)} queries …")
    for i, entry in enumerate(eval_set, start=1):
        gold_ids = {g["corpus_id"] for g in entry["gold_relevant"]}
        qtext    = entry["query_text"]
        repo     = entry["query_repo"]
        split    = entry["query_split"]
        labels   = entry["query_labels"]

        # ── lexical ──
        lex_ids = lex_ranked_ids(qtext, vectorizer, matrix, corpus, k=max_k)

        # ── dense ──
        den_ids = dense_ranked_ids(qtext, dense, tokenizer, model, device, k=max_k)

        # ── hybrid ──
        hyb_ids = hybrid_ranked_ids(qtext, hybrid, k=max_k)

        def make_row(ranked_ids: list[int]) -> dict:
            row = {
                "query_id":   entry["query_id"],
                "query_repo": repo,
                "split":      split,
                "labels":     labels,
                "n_gold":     len(gold_ids),
                "rr":         reciprocal_rank(gold_ids, ranked_ids),
            }
            for k in KS:
                row[f"recall@{k}"] = recall_at_k(gold_ids, ranked_ids, k)
            return row

        lex_rows.append(make_row(lex_ids))
        dense_rows.append(make_row(den_ids))
        hybrid_rows.append(make_row(hyb_ids))

        if i % 5 == 0 or i == len(eval_set):
            print(f"  {i}/{len(eval_set)} done")

    # ── aggregate ──────────────────────────────────────────────────────────────
    methods = {
        "lexical": lex_rows,
        "dense":   dense_rows,
        "hybrid":  hybrid_rows,
    }

    def full_agg(rows: list[dict]) -> dict:
        df = pd.DataFrame(rows)
        return {
            "overall":   aggregate(rows),
            "by_split":  {s: aggregate(g.to_dict("records"))
                          for s, g in df.groupby("split")},
            "by_repo":   {r: aggregate(g.to_dict("records"))
                          for r, g in df.groupby("query_repo")},
            "per_query": rows,
        }

    results = {m: full_agg(rows) for m, rows in methods.items()}
    results["meta"] = {
        "eval_set":    str(EVAL_PATH),
        "n_queries":   len(eval_set),
        "pool_size":   POOL,
        "rrf_k":       RRF_K,
        "ks":          KS,
    }

    OUT_PATH.write_text(json.dumps(results, indent=2))

    # ── print comparison table ─────────────────────────────────────────────────
    print(f"\n── Overall ({len(eval_set)} queries, human-annotated eval set) ──")
    print(f"{'Method':<10}  {'R@5':>6}  {'R@10':>6}  {'R@20':>6}  {'MRR':>6}")
    print("-" * 46)
    for method in ("lexical", "dense", "hybrid"):
        m = results[method]["overall"]
        marker = " ◀" if method == "hybrid" else ""
        print(f"{method:<10}  {m['recall@5']:>6.3f}  {m['recall@10']:>6.3f}"
              f"  {m['recall@20']:>6.3f}  {m['mrr']:>6.3f}{marker}")

    print("\n── By repo ──")
    repos = sorted({e["query_repo"] for e in eval_set})
    print(f"{'Repo':<15}  {'Lex R@10':>9}  {'Den R@10':>9}  {'Hyb R@10':>9}  "
          f"{'Lex MRR':>8}  {'Hyb MRR':>8}")
    print("-" * 70)
    for repo in repos:
        lex_r = results["lexical"]["by_repo"].get(repo, {})
        den_r = results["dense"]["by_repo"].get(repo, {})
        hyb_r = results["hybrid"]["by_repo"].get(repo, {})
        print(f"{repo:<15}  {lex_r.get('recall@10', 0):>9.3f}"
              f"  {den_r.get('recall@10', 0):>9.3f}"
              f"  {hyb_r.get('recall@10', 0):>9.3f}"
              f"  {lex_r.get('mrr', 0):>8.3f}"
              f"  {hyb_r.get('mrr', 0):>8.3f}")

    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
