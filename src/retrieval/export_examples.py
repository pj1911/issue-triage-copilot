"""
Export 10 good and 10 bad retrieval examples.

"Good"  = queries where MRR == 1.0 (gold hit at rank 1), then by R@10 desc.
"Bad"   = queries where MRR == 0.0 (no gold in top-20), then by R@10 asc.

For each example we show:
  - Query text snippet + labels + repo
  - Gold relevant docs (from eval set)
  - Actual top-5 retrieved docs with scores

Output:
  reports/retrieval_examples.json   — machine-readable
  reports/retrieval_examples.txt    — human-readable summary
"""
import json
import textwrap

import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.paths import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR

EVAL_PATH    = PROCESSED_DIR / "retrieval_eval.json"
METRICS_PATH = REPORTS_DIR   / "retrieval_metrics.json"
CORPUS_PATH  = PROCESSED_DIR / "retrieval_corpus.parquet"
VEC_PATH     = MODELS_DIR    / "tfidf_retrieval_vectorizer.joblib"
MATRIX_PATH  = MODELS_DIR    / "tfidf_retrieval_matrix.npz"
JSON_OUT     = REPORTS_DIR   / "retrieval_examples.json"
TXT_OUT      = REPORTS_DIR   / "retrieval_examples.txt"

N = 10          # examples per bucket
TOP_K = 5       # retrieved docs to show per example


def retrieve_top_k(query_text: str, matrix, vectorizer, k: int):
    q_vec  = vectorizer.transform([query_text])
    scores = cosine_similarity(q_vec, matrix).ravel()
    idx    = np.argpartition(scores, -k)[-k:]
    idx    = idx[np.argsort(scores[idx])[::-1]]
    return idx, scores[idx]


def build_example(entry: dict, per_query_map: dict,
                  corpus: pd.DataFrame, id_to_idx: dict,
                  matrix, vectorizer) -> dict:
    pq = per_query_map[entry["query_id"]]

    # Retrieve top-K
    top_idx, top_scores = retrieve_top_k(
        entry["query_text"], matrix, vectorizer, TOP_K
    )

    gold_ids = {g["corpus_id"] for g in entry["gold_relevant"]}

    retrieved = []
    for rank, (idx, score) in enumerate(zip(top_idx, top_scores), start=1):
        row = corpus.iloc[int(idx)]
        retrieved.append({
            "rank":        rank,
            "score":       round(float(score), 4),
            "is_gold":     int(row["id"]) in gold_ids,
            "corpus_id":   int(row["id"]),
            "corpus_repo": row["repo"],
            "labels":      list(row["labels_clean"]),
            "text_snippet": row["text"][:200],
        })

    return {
        "query_id":     entry["query_id"],
        "query_repo":   entry["query_repo"],
        "query_split":  entry["query_split"],
        "query_labels": entry["query_labels"],
        "query_snippet": entry["query_text"][:300],
        "mrr":          round(pq["rr"], 4),
        "recall@5":     round(pq["recall@5"], 4),
        "recall@10":    round(pq["recall@10"], 4),
        "gold_relevant": entry["gold_relevant"],
        "top5_retrieved": retrieved,
    }


def fmt_example(ex: dict, bucket: str, num: int) -> str:
    sep = "─" * 72
    lines = [
        sep,
        f"[{bucket.upper()} #{num}]  repo={ex['query_repo']}  "
        f"split={ex['query_split']}  labels={ex['query_labels']}",
        f"MRR={ex['mrr']:.4f}  Recall@5={ex['recall@5']:.4f}  "
        f"Recall@10={ex['recall@10']:.4f}",
        "",
        "QUERY:",
        *textwrap.wrap(ex["query_snippet"].replace("\n", " "), 70),
        "",
        "GOLD RELEVANT (1–3 expected targets):",
    ]
    for g in ex["gold_relevant"]:
        lines.append(
            f"  [{g['corpus_repo']}] score={g['score']}  "
            f"shared={g['shared_labels']}"
        )
        lines.append(
            "  " + g["text_snippet"].splitlines()[0][:70]
        )
    lines += ["", f"TOP-{TOP_K} RETRIEVED:"]
    for r in ex["top5_retrieved"]:
        hit = " ← GOLD HIT" if r["is_gold"] else ""
        lines.append(
            f"  #{r['rank']} score={r['score']:.4f}  "
            f"[{r['corpus_repo']}]  labels={r['labels']}{hit}"
        )
        lines.append(
            "  " + r["text_snippet"].splitlines()[0][:70]
        )
    return "\n".join(lines)


def main() -> None:
    with open(EVAL_PATH)    as f: eval_set = json.load(f)
    with open(METRICS_PATH) as f: metrics  = json.load(f)

    corpus     = pd.read_parquet(CORPUS_PATH)
    vectorizer = joblib.load(VEC_PATH)
    matrix     = sp.load_npz(str(MATRIX_PATH))
    id_to_idx  = {int(cid): i for i, cid in enumerate(corpus["id"])}

    per_query_map = {r["query_id"]: r for r in metrics["per_query"]}
    eval_map      = {e["query_id"]: e for e in eval_set}

    rows = metrics["per_query"]

    # ── select good / bad ──────────────────────────────────────────────────
    # good: highest MRR first, break ties by R@10
    good_rows = sorted(
        [r for r in rows if r["rr"] > 0],
        key=lambda r: (r["rr"], r["recall@10"]),
        reverse=True,
    )[:N]

    # bad: MRR == 0 first, then lowest R@10
    bad_rows = sorted(
        rows,
        key=lambda r: (r["rr"], r["recall@10"]),
    )[:N]

    print(f"Good examples selected: {len(good_rows)}")
    print(f"Bad  examples selected: {len(bad_rows)}")

    # ── build ──────────────────────────────────────────────────────────────
    good_examples = [
        build_example(eval_map[r["query_id"]], per_query_map,
                      corpus, id_to_idx, matrix, vectorizer)
        for r in good_rows
    ]
    bad_examples = [
        build_example(eval_map[r["query_id"]], per_query_map,
                      corpus, id_to_idx, matrix, vectorizer)
        for r in bad_rows
    ]

    output = {"good": good_examples, "bad": bad_examples}
    with open(JSON_OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved JSON → {JSON_OUT}")

    # ── text report ────────────────────────────────────────────────────────
    sections = ["RETRIEVAL EXAMPLES — LEXICAL (TF-IDF + COSINE)\n"]
    sections.append("=" * 72)
    sections.append(f"\n{'GOOD RETRIEVALS':^72}\n")
    sections.append("=" * 72)
    for i, ex in enumerate(good_examples, 1):
        sections.append(fmt_example(ex, "good", i))

    sections.append("\n" + "=" * 72)
    sections.append(f"\n{'BAD RETRIEVALS':^72}\n")
    sections.append("=" * 72)
    for i, ex in enumerate(bad_examples, 1):
        sections.append(fmt_example(ex, "bad", i))

    txt = "\n".join(sections) + "\n"
    TXT_OUT.write_text(txt)
    print(f"Saved TXT  → {TXT_OUT}")

    # ── quick console summary ──────────────────────────────────────────────
    print("\n── Good (top 5) ──")
    for ex in good_examples[:5]:
        print(f"  [{ex['query_repo']:<12}] MRR={ex['mrr']:.3f}  "
              f"labels={ex['query_labels']}")
        print(f"    {ex['query_snippet'].splitlines()[0][7:70]}")

    print("\n── Bad (top 5) ──")
    for ex in bad_examples[:5]:
        print(f"  [{ex['query_repo']:<12}] MRR={ex['mrr']:.3f}  "
              f"labels={ex['query_labels']}")
        print(f"    {ex['query_snippet'].splitlines()[0][7:70]}")


if __name__ == "__main__":
    main()
