"""
Evaluate the lexical retriever on the manual eval set.

Metrics (per query, then averaged):
  Recall@k   — fraction of gold relevant docs found in top-k results
  MRR        — mean reciprocal rank of the first gold hit

Run from project root:
    python -m src.retrieval.evaluate_retrieval

Output:
    reports/retrieval_metrics.json
"""
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.paths import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR

EVAL_PATH   = PROCESSED_DIR / "retrieval_eval.json"
CORPUS_PATH = PROCESSED_DIR / "retrieval_corpus.parquet"
VEC_PATH    = MODELS_DIR    / "tfidf_retrieval_vectorizer.joblib"
MATRIX_PATH = MODELS_DIR    / "tfidf_retrieval_matrix.npz"
OUT_PATH    = REPORTS_DIR   / "retrieval_metrics.json"

KS = [5, 10, 20]


# ── retrieval ──────────────────────────────────────────────────────────────────

def retrieve_top_k(query_text: str, matrix: sp.csr_matrix,
                   vectorizer, k: int) -> np.ndarray:
    """Return corpus row indices of the top-k hits (sorted by score desc)."""
    q_vec  = vectorizer.transform([query_text])
    scores = cosine_similarity(q_vec, matrix).ravel()
    top    = np.argpartition(scores, -k)[-k:]
    return top[np.argsort(scores[top])[::-1]]


# ── metrics ────────────────────────────────────────────────────────────────────

def recall_at_k(gold_indices: set[int], ranked: np.ndarray, k: int) -> float:
    """Fraction of gold docs that appear in the top-k ranked list."""
    if not gold_indices:
        return 0.0
    hits = sum(1 for idx in ranked[:k] if idx in gold_indices)
    return hits / len(gold_indices)


def reciprocal_rank(gold_indices: set[int], ranked: np.ndarray) -> float:
    """1/rank of the first gold doc in the ranked list (0 if not found)."""
    for rank, idx in enumerate(ranked, start=1):
        if idx in gold_indices:
            return 1.0 / rank
    return 0.0


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading eval set and index …")
    with open(EVAL_PATH) as f:
        eval_set = json.load(f)

    corpus     = pd.read_parquet(CORPUS_PATH)
    vectorizer = joblib.load(VEC_PATH)
    matrix     = sp.load_npz(str(MATRIX_PATH))

    # Build a lookup: corpus_id → row index in matrix
    id_to_idx = {int(cid): i for i, cid in enumerate(corpus["id"])}

    max_k    = max(KS)
    per_query_results = []

    print(f"Evaluating {len(eval_set)} queries (top-{max_k}) …")
    for entry in eval_set:
        gold_ids  = {g["corpus_id"] for g in entry["gold_relevant"]}
        gold_idxs = {id_to_idx[gid] for gid in gold_ids if gid in id_to_idx}

        if not gold_idxs:
            continue

        ranked = retrieve_top_k(entry["query_text"], matrix, vectorizer, max_k)

        row = {
            "query_id":   entry["query_id"],
            "query_repo": entry["query_repo"],
            "split":      entry["query_split"],
            "labels":     entry["query_labels"],
            "n_gold":     len(gold_idxs),
            "rr":         reciprocal_rank(gold_idxs, ranked),
        }
        for k in KS:
            row[f"recall@{k}"] = recall_at_k(gold_idxs, ranked, k)

        per_query_results.append(row)

    results_df = pd.DataFrame(per_query_results)

    # ── aggregate ──────────────────────────────────────────────────────────────
    def agg(df: pd.DataFrame, label: str) -> dict:
        d = {"n": len(df)}
        for k in KS:
            d[f"recall@{k}"] = round(float(df[f"recall@{k}"].mean()), 4)
        d["mrr"] = round(float(df["rr"].mean()), 4)
        return d

    overall = agg(results_df, "overall")
    by_split = {
        split: agg(g, split)
        for split, g in results_df.groupby("split")
    }
    by_repo = {
        repo: agg(g, repo)
        for repo, g in results_df.groupby("query_repo")
    }

    metrics = {
        "overall": overall,
        "by_split": by_split,
        "by_repo":  by_repo,
        "per_query": per_query_results,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── print ──────────────────────────────────────────────────────────────────
    print(f"\n── Overall ({overall['n']} queries) ──")
    for k in KS:
        print(f"  Recall@{k:<3} : {overall[f'recall@{k}']:.4f}")
    print(f"  MRR        : {overall['mrr']:.4f}")

    print("\n── By split ──")
    for split, m in by_split.items():
        print(f"  {split:<6}  R@5={m['recall@5']:.4f}  "
              f"R@10={m['recall@10']:.4f}  MRR={m['mrr']:.4f}  (n={m['n']})")

    print("\n── By repo ──")
    for repo, m in sorted(by_repo.items()):
        print(f"  {repo:<15}  R@5={m['recall@5']:.4f}  "
              f"R@10={m['recall@10']:.4f}  MRR={m['mrr']:.4f}  (n={m['n']})")

    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
