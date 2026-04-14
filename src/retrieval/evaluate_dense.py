"""
Evaluate the dense retriever (BAAI/bge-small-en-v1.5) on the Day 5 eval set.

Uses the exact same 37-query eval set and same metrics as evaluate_retrieval.py
so results are directly comparable.

Metrics (per query, then averaged):
    Recall@k   — fraction of gold relevant docs found in top-k results
    MRR        — mean reciprocal rank of the first gold hit

Output:
    reports/dense_retrieval_metrics.json   — full metrics (same schema as Day 5)
    reports/dense_retrieval_report.txt     — human-readable summary + comparison

Run from project root:
    python -m src.retrieval.evaluate_dense
"""
import json
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from src.retrieval.dense_retriever import (
    DenseRetriever,
    MODEL_NAME,
    _get_device,
    encode_texts,
)
from src.utils.paths import PROCESSED_DIR, REPORTS_DIR

# ── paths ──────────────────────────────────────────────────────────────────────
EVAL_PATH         = PROCESSED_DIR / "retrieval_eval.json"
LEXICAL_METRICS   = REPORTS_DIR   / "retrieval_metrics.json"        # Day 5
DENSE_METRICS_OUT = REPORTS_DIR   / "dense_retrieval_metrics.json"
DENSE_REPORT_OUT  = REPORTS_DIR   / "dense_retrieval_report.txt"

KS = [5, 10, 20]


# ── metrics ────────────────────────────────────────────────────────────────────

def recall_at_k(gold_indices: set[int], ranked: np.ndarray, k: int) -> float:
    if not gold_indices:
        return 0.0
    hits = sum(1 for idx in ranked[:k] if idx in gold_indices)
    return hits / len(gold_indices)


def reciprocal_rank(gold_indices: set[int], ranked: np.ndarray) -> float:
    for rank, idx in enumerate(ranked, start=1):
        if idx in gold_indices:
            return 1.0 / rank
    return 0.0


# ── retrieval ──────────────────────────────────────────────────────────────────

def retrieve_top_k_dense(
    query_text: str,
    embeddings: np.ndarray,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    k: int,
) -> np.ndarray:
    """Return corpus row indices of the top-k hits (sorted by score desc)."""
    q_emb  = encode_texts([query_text], tokenizer, model, device,
                           batch_size=1, show_progress=False)   # (1, H)
    scores = (embeddings @ q_emb.T).ravel()                     # (n_docs,)
    top    = np.argpartition(scores, -k)[-k:]
    return top[np.argsort(scores[top])[::-1]]


# ── aggregate helpers ──────────────────────────────────────────────────────────

def agg(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    d = {"n": len(df)}
    for k in KS:
        d[f"recall@{k}"] = round(float(df[f"recall@{k}"].mean()), 4)
    d["mrr"] = round(float(df["rr"].mean()), 4)
    return d


# ── report ─────────────────────────────────────────────────────────────────────

def _fmt_row(label: str, m: dict) -> str:
    return (f"  {label:<17}"
            f"  R@5={m['recall@5']:.4f}"
            f"  R@10={m['recall@10']:.4f}"
            f"  R@20={m['recall@20']:.4f}"
            f"  MRR={m['mrr']:.4f}"
            f"  (n={m['n']})")


def write_report(dense: dict, lexical: dict | None) -> None:
    ov_d = dense["overall"]
    lines = [
        "Day 6 — Dense Retrieval Baseline (BAAI/bge-small-en-v1.5)",
        "=" * 60,
        "",
        f"Model  : {MODEL_NAME}",
        f"Pooling: CLS token + L2 normalisation",
        f"Corpus : 71,351 train issues  |  58 repos",
        f"Eval   : 37 queries (25 val / 12 test)  |  3 gold / query",
        "",
        "Overall",
        "-------",
        f"  Recall@5   : {ov_d['recall@5']:.4f}",
        f"  Recall@10  : {ov_d['recall@10']:.4f}",
        f"  Recall@20  : {ov_d['recall@20']:.4f}",
        f"  MRR        : {ov_d['mrr']:.4f}",
        "",
    ]

    if lexical:
        ov_l = lexical["overall"]
        def delta(key: str) -> str:
            d = ov_d[key] - ov_l[key]
            return f"{d:+.4f}"

        lines += [
            "Day 5 vs Day 6 comparison (same 37-query eval set)",
            "---------------------------------------------------",
            f"  {'Metric':<12}  {'Lexical':>8}  {'Dense':>8}  {'Delta':>8}",
            f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}",
            f"  {'Recall@5':<12}  {ov_l['recall@5']:>8.4f}  {ov_d['recall@5']:>8.4f}  {delta('recall@5'):>8}",
            f"  {'Recall@10':<12}  {ov_l['recall@10']:>8.4f}  {ov_d['recall@10']:>8.4f}  {delta('recall@10'):>8}",
            f"  {'Recall@20':<12}  {ov_l['recall@20']:>8.4f}  {ov_d['recall@20']:>8.4f}  {delta('recall@20'):>8}",
            f"  {'MRR':<12}  {ov_l['mrr']:>8.4f}  {ov_d['mrr']:>8.4f}  {delta('mrr'):>8}",
            "",
        ]

    lines += ["By split", "--------"]
    for split, m in dense["by_split"].items():
        lines.append(_fmt_row(split, m))

    lines += ["", "By repo", "-------"]
    for repo, m in sorted(dense["by_repo"].items()):
        suffix = ""
        if lexical and repo in lexical.get("by_repo", {}):
            lm = lexical["by_repo"][repo]
            d5 = f"R@5 {ov_d['recall@5']:.3f}"  # placeholder; use repo-level
            delta_r5  = m["recall@5"]  - lm["recall@5"]
            delta_mrr = m["mrr"]       - lm["mrr"]
            suffix = f"  [vs lexical: R@5 {delta_r5:+.3f}  MRR {delta_mrr:+.3f}]"
        lines.append(_fmt_row(repo, m) + suffix)

    lines += [
        "",
        "Index artifacts",
        "---------------",
        "  models/dense_doc_embeddings.npy",
        "  models/dense_doc_ids.npy",
        "  models/dense_model_config.json",
    ]

    DENSE_REPORT_OUT.write_text("\n".join(lines) + "\n")
    print(f"Saved report → {DENSE_REPORT_OUT}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading eval set …")
    with open(EVAL_PATH) as f:
        eval_set = json.load(f)

    print("Loading dense index …")
    retriever = DenseRetriever.load()
    embeddings = retriever.embeddings          # (n_docs, H)  float32  L2-normed

    device    = _get_device()
    print(f"Device : {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    max_k = max(KS)
    per_query_results: list[dict] = []

    print(f"Evaluating {len(eval_set)} queries (top-{max_k}) …")
    t0 = time.perf_counter()

    for entry in eval_set:
        gold_ids  = {g["corpus_id"] for g in entry["gold_relevant"]}
        gold_idxs = {retriever.id_to_idx(gid) for gid in gold_ids
                     if retriever.id_to_idx(gid) is not None}

        if not gold_idxs:
            continue

        ranked = retrieve_top_k_dense(
            entry["query_text"], embeddings, tokenizer, model, device, k=max_k
        )

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

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s  ({elapsed/len(per_query_results)*1000:.0f} ms/query)")

    # ── aggregate ──────────────────────────────────────────────────────────────
    overall  = agg(per_query_results)
    by_split = {
        split: agg([r for r in per_query_results if r["split"] == split])
        for split in sorted({r["split"] for r in per_query_results})
    }
    by_repo  = {
        repo: agg([r for r in per_query_results if r["query_repo"] == repo])
        for repo in sorted({r["query_repo"] for r in per_query_results})
    }

    metrics = {
        "overall":    overall,
        "by_split":   by_split,
        "by_repo":    by_repo,
        "per_query":  per_query_results,
    }

    DENSE_METRICS_OUT.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics → {DENSE_METRICS_OUT}")

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

    # ── Day 5 comparison ───────────────────────────────────────────────────────
    lexical = None
    if LEXICAL_METRICS.exists():
        with open(LEXICAL_METRICS) as f:
            lexical = json.load(f)

        ov_l = lexical["overall"]
        print("\n── Day 5 vs Day 6 (overall) ──")
        print(f"  {'Metric':<12}  {'Lexical':>8}  {'Dense':>8}  {'Delta':>8}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}")
        for key in ["recall@5", "recall@10", "recall@20", "mrr"]:
            d = overall[key] - ov_l[key]
            print(f"  {key:<12}  {ov_l[key]:>8.4f}  {overall[key]:>8.4f}  {d:>+8.4f}")

    write_report(metrics, lexical)


if __name__ == "__main__":
    main()
