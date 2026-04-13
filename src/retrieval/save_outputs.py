"""
Save Day 5 retrieval outputs.

Reads existing JSON artifacts and writes:
  reports/retrieval_report.txt    — human-readable eval summary
  reports/retrieval_examples.csv  — flat CSV of good/bad examples

Run from project root:
    python -m src.retrieval.save_outputs
"""
import csv
import json

from src.utils.paths import MODELS_DIR, REPORTS_DIR

METRICS_PATH  = REPORTS_DIR / "retrieval_metrics.json"
EXAMPLES_PATH = REPORTS_DIR / "retrieval_examples.json"
REPORT_OUT    = REPORTS_DIR / "retrieval_report.txt"
CSV_OUT       = REPORTS_DIR / "retrieval_examples.csv"


# ── eval report ────────────────────────────────────────────────────────────────

def write_report(metrics: dict) -> None:
    ov = metrics["overall"]
    lines = [
        "Day 5 — Retrieval Baseline (Lexical: TF-IDF + Cosine)",
        "=" * 56,
        "",
        f"Corpus : {71351:,} train issues  |  58 repos",
        f"Vocab  : 150,000 features  |  matrix 71k × 150k",
        f"Eval   : 37 queries (25 val / 12 test)  |  3 gold / query",
        "",
        "Overall",
        "-------",
        f"  Recall@5   : {ov['recall@5']:.4f}",
        f"  Recall@10  : {ov['recall@10']:.4f}",
        f"  Recall@20  : {ov['recall@20']:.4f}",
        f"  MRR        : {ov['mrr']:.4f}",
        "",
        "By split",
        "--------",
    ]
    for split, m in metrics["by_split"].items():
        lines.append(
            f"  {split:<6}  R@5={m['recall@5']:.4f}  "
            f"R@10={m['recall@10']:.4f}  MRR={m['mrr']:.4f}  (n={m['n']})"
        )

    lines += ["", "By repo", "-------"]
    for repo, m in sorted(metrics["by_repo"].items()):
        lines.append(
            f"  {repo:<15}  R@5={m['recall@5']:.4f}  "
            f"R@10={m['recall@10']:.4f}  MRR={m['mrr']:.4f}  (n={m['n']})"
        )

    lines += [
        "",
        "Index artifacts",
        "---------------",
        f"  models/tfidf_retrieval_vectorizer.joblib  (5.8 MB)",
        f"  models/tfidf_retrieval_matrix.npz         (168 MB)",
        f"  models/tfidf_retrieval_ids.npy            (558 KB)",
        f"  data/processed/retrieval_corpus.parquet   (153 MB)",
        "",
        "Key observations",
        "----------------",
        "  Strong  : deno, neovim, vscode  (R@10 0.53–0.67)",
        "            Distinctive vocabulary matches corpus tokens directly.",
        "  Weak    : flutter, transformers  (R@10 0.00–0.07)",
        "            Semantic mismatch — repo-specific jargon and",
        "            cross-repo label noise defeat keyword overlap.",
        "  Next    : Dense retriever (encoder embeddings) should close",
        "            the gap on the weak repos.",
    ]

    REPORT_OUT.write_text("\n".join(lines) + "\n")
    print(f"Saved report → {REPORT_OUT}")


# ── examples CSV ───────────────────────────────────────────────────────────────

FIELDS = [
    "bucket",
    "query_id", "query_repo", "query_split", "query_labels",
    "mrr", "recall@5", "recall@10",
    "query_snippet",
    # gold #1
    "gold1_repo", "gold1_shared_labels", "gold1_score", "gold1_snippet",
    # top-3 retrieved
    "ret1_repo", "ret1_labels", "ret1_score", "ret1_is_gold",
    "ret2_repo", "ret2_labels", "ret2_score", "ret2_is_gold",
    "ret3_repo", "ret3_labels", "ret3_score", "ret3_is_gold",
]


def flatten(bucket: str, ex: dict) -> dict:
    gold = ex["gold_relevant"]
    ret  = ex["top5_retrieved"]

    def g(i: int, key: str):
        return gold[i][key] if i < len(gold) else ""

    def r(i: int, key: str):
        return ret[i][key] if i < len(ret) else ""

    return {
        "bucket":        bucket,
        "query_id":      ex["query_id"],
        "query_repo":    ex["query_repo"],
        "query_split":   ex["query_split"],
        "query_labels":  "|".join(ex["query_labels"]),
        "mrr":           ex["mrr"],
        "recall@5":      ex["recall@5"],
        "recall@10":     ex["recall@10"],
        "query_snippet": ex["query_snippet"].replace("\n", " ")[:200],
        "gold1_repo":          g(0, "corpus_repo"),
        "gold1_shared_labels": "|".join(g(0, "shared_labels") or []),
        "gold1_score":         g(0, "score"),
        "gold1_snippet":       (g(0, "text_snippet") or "").replace("\n", " ")[:120],
        "ret1_repo":     r(0, "corpus_repo"),
        "ret1_labels":   "|".join(r(0, "labels") or []),
        "ret1_score":    r(0, "score"),
        "ret1_is_gold":  r(0, "is_gold"),
        "ret2_repo":     r(1, "corpus_repo"),
        "ret2_labels":   "|".join(r(1, "labels") or []),
        "ret2_score":    r(1, "score"),
        "ret2_is_gold":  r(1, "is_gold"),
        "ret3_repo":     r(2, "corpus_repo"),
        "ret3_labels":   "|".join(r(2, "labels") or []),
        "ret3_score":    r(2, "score"),
        "ret3_is_gold":  r(2, "is_gold"),
    }


def write_csv(examples: dict) -> None:
    rows = []
    for bucket in ("good", "bad"):
        for ex in examples[bucket]:
            rows.append(flatten(bucket, ex))

    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV  → {CSV_OUT}  ({len(rows)} rows)")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(METRICS_PATH)  as f: metrics  = json.load(f)
    with open(EXAMPLES_PATH) as f: examples = json.load(f)

    write_report(metrics)
    write_csv(examples)

    print("\nAll Day 5 outputs saved:")
    artifacts = [
        ("index",   "models/tfidf_retrieval_vectorizer.joblib"),
        ("index",   "models/tfidf_retrieval_matrix.npz"),
        ("index",   "models/tfidf_retrieval_ids.npy"),
        ("corpus",  "data/processed/retrieval_corpus.parquet"),
        ("eval",    "data/processed/retrieval_eval.json"),
        ("metrics", "reports/retrieval_metrics.json"),
        ("report",  "reports/retrieval_report.txt"),
        ("csv",     "reports/retrieval_examples.csv"),
        ("txt",     "reports/retrieval_examples.txt"),
    ]
    for kind, path in artifacts:
        print(f"  [{kind:<7}] {path}")


if __name__ == "__main__":
    main()
