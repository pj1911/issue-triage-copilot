"""
Export comparison examples: hybrid wins, hybrid failures.

Buckets (classified by MRR from hybrid_metrics.json):
    hybrid_win          : hyb_mrr > lex_mrr AND hyb_mrr > dense_mrr
    dense_beats_hybrid  : dense_mrr > hyb_mrr by largest margin
    lex_beats_hybrid    : lex_mrr > hyb_mrr by largest margin

For each selected query we re-run all three retrievers and show the top-5
results so the difference is directly readable.

Run from project root:
    python -m src.retrieval.export_hybrid_examples

Output:
    reports/hybrid_examples.json
    reports/hybrid_examples.txt
"""
import json
import textwrap

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from src.retrieval.dense_retriever import (
    MODEL_NAME, _get_device, encode_texts, DenseRetriever,
)
from src.retrieval.hybrid_retriever import HybridRetriever, RRF_K, RETRIEVAL_POOL
from src.retrieval.lexical_retriever import LexicalRetriever
from src.utils.paths import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR

EVAL_PATH       = PROCESSED_DIR / "retrieval_eval_human.json"
METRICS_PATH    = REPORTS_DIR   / "hybrid_metrics.json"
JSON_OUT        = REPORTS_DIR   / "hybrid_examples.json"
TXT_OUT         = REPORTS_DIR   / "hybrid_examples.txt"

N_WIN  = 3   # hybrid win examples
N_FAIL = 3   # dense-beats-hybrid examples
N_LEX  = 2   # lex-beats-hybrid examples
TOP_K  = 5   # results shown per method


# ── bucket classification ──────────────────────────────────────────────────────

def classify(metrics: dict) -> dict[str, list[dict]]:
    """
    Read per_query rows from all three methods and assign each query to a bucket.

    Returns dict: hybrid_win, dense_beats_hybrid, lex_beats_hybrid, hybrid_ok
    """
    lex_rows = {r["query_id"]: r for r in metrics["lexical"]["per_query"]}
    den_rows = {r["query_id"]: r for r in metrics["dense"]["per_query"]}
    hyb_rows = {r["query_id"]: r for r in metrics["hybrid"]["per_query"]}

    all_ids = sorted(set(lex_rows) | set(den_rows) | set(hyb_rows))

    buckets: dict[str, list[dict]] = {
        "hybrid_win":         [],
        "dense_beats_hybrid": [],
        "lex_beats_hybrid":   [],
        "hybrid_ok":          [],
    }

    for qid in all_ids:
        lr = lex_rows.get(qid, {})
        dr = den_rows.get(qid, {})
        hr = hyb_rows.get(qid, {})

        lex_mrr = lr.get("rr", 0.0)
        den_mrr = dr.get("rr", 0.0)
        hyb_mrr = hr.get("rr", 0.0)

        row = {
            "query_id":   qid,
            "query_repo": hr.get("query_repo", lr.get("query_repo", "?")),
            "split":      hr.get("split", "?"),
            "labels":     hr.get("labels", []),
            "lex_mrr":    lex_mrr,
            "den_mrr":    den_mrr,
            "hyb_mrr":    hyb_mrr,
            "lex_r5":     lr.get("recall@5", 0.0),
            "den_r5":     dr.get("recall@5", 0.0),
            "hyb_r5":     hr.get("recall@5", 0.0),
            "lex_r10":    lr.get("recall@10", 0.0),
            "den_r10":    dr.get("recall@10", 0.0),
            "hyb_r10":    hr.get("recall@10", 0.0),
            "den_delta":  hyb_mrr - den_mrr,   # negative = dense beat hybrid
            "lex_delta":  hyb_mrr - lex_mrr,   # negative = lex beat hybrid
        }

        if hyb_mrr > lex_mrr and hyb_mrr > den_mrr:
            buckets["hybrid_win"].append(row)
        elif den_mrr > hyb_mrr and (den_mrr - hyb_mrr) >= (lex_mrr - hyb_mrr):
            buckets["dense_beats_hybrid"].append(row)
        elif lex_mrr > hyb_mrr:
            buckets["lex_beats_hybrid"].append(row)
        else:
            buckets["hybrid_ok"].append(row)

    # sort each bucket by the magnitude of the gap
    buckets["hybrid_win"] = sorted(
        buckets["hybrid_win"],
        key=lambda r: r["hyb_mrr"] - max(r["lex_mrr"], r["den_mrr"]),
        reverse=True,
    )[:N_WIN]
    buckets["dense_beats_hybrid"] = sorted(
        buckets["dense_beats_hybrid"],
        key=lambda r: r["den_mrr"] - r["hyb_mrr"],
        reverse=True,
    )[:N_FAIL]
    buckets["lex_beats_hybrid"] = sorted(
        buckets["lex_beats_hybrid"],
        key=lambda r: r["lex_mrr"] - r["hyb_mrr"],
        reverse=True,
    )[:N_LEX]

    return buckets


# ── per-method retrieve → top-K rows ──────────────────────────────────────────

def lex_top(query: str, vectorizer, matrix: sp.csr_matrix,
            corpus: pd.DataFrame, k: int) -> list[dict]:
    q_vec  = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, matrix).ravel()
    idx    = np.argpartition(scores, -k)[-k:]
    idx    = idx[np.argsort(scores[idx])[::-1]]
    rows = []
    for rank, i in enumerate(idx, 1):
        doc = corpus.iloc[int(i)]
        rows.append({
            "rank": rank, "score": round(float(scores[i]), 4),
            "corpus_id": int(doc["id"]), "repo": doc["repo"],
            "labels": list(doc["labels_clean"]),
            "snippet": doc["text"][:200],
        })
    return rows


def dense_top(query: str, dense: DenseRetriever,
              tokenizer, model, device, k: int) -> list[dict]:
    hits = dense.retrieve(query, tokenizer, model, device, k=k)
    rows = []
    for _, row in hits.iterrows():
        rows.append({
            "rank": int(row["rank"]), "score": round(float(row["score"]), 4),
            "corpus_id": int(row["id"]), "repo": row["repo"],
            "labels": list(row["labels_clean"]),
            "snippet": row["text"][:200],
        })
    return rows


def hybrid_top(query: str, hybrid: HybridRetriever, k: int) -> list[dict]:
    hits = hybrid.retrieve(query, k=k)
    rows = []
    for _, row in hits.iterrows():
        rows.append({
            "rank": int(row["rank"]), "score": round(float(row["rrf_score"]), 6),
            "lex_rank": None if pd.isna(row["lex_rank"]) else int(row["lex_rank"]),
            "dense_rank": None if pd.isna(row["dense_rank"]) else int(row["dense_rank"]),
            "corpus_id": int(row["id"]), "repo": row["repo"],
            "labels": list(row["labels_clean"]),
            "snippet": row["text"][:200],
        })
    return rows


# ── build full example record ──────────────────────────────────────────────────

def build_example(
    meta: dict,
    eval_entry: dict,
    lex: LexicalRetriever,
    dense: DenseRetriever,
    hybrid: HybridRetriever,
    tokenizer, model, device,
) -> dict:
    query    = eval_entry["query_text"]
    gold_ids = {g["corpus_id"] for g in eval_entry["gold_relevant"]}

    lex_hits   = lex_top(query, lex.vectorizer, lex.matrix, lex.corpus, TOP_K)
    dense_hits = dense_top(query, dense, tokenizer, model, device, TOP_K)
    hyb_hits   = hybrid_top(query, hybrid, TOP_K)

    def mark_gold(hits):
        for h in hits:
            h["is_gold"] = h["corpus_id"] in gold_ids
        return hits

    return {
        "query_id":    meta["query_id"],
        "query_repo":  meta["query_repo"],
        "split":       meta["split"],
        "labels":      meta["labels"],
        "query_snippet": query[:300],
        "metrics": {
            "lex":    {"mrr": meta["lex_mrr"], "r@5": meta["lex_r5"],  "r@10": meta["lex_r10"]},
            "dense":  {"mrr": meta["den_mrr"], "r@5": meta["den_r5"],  "r@10": meta["den_r10"]},
            "hybrid": {"mrr": meta["hyb_mrr"], "r@5": meta["hyb_r5"],  "r@10": meta["hyb_r10"]},
        },
        "gold_relevant": eval_entry["gold_relevant"],
        "lexical_top5":  mark_gold(lex_hits),
        "dense_top5":    mark_gold(dense_hits),
        "hybrid_top5":   mark_gold(hyb_hits),
    }


# ── text formatting ────────────────────────────────────────────────────────────

def fmt_hits(hits: list[dict], label: str, show_src: bool = False) -> list[str]:
    lines = [f"  {label} TOP-{TOP_K}:"]
    for h in hits:
        gold  = " ← GOLD" if h.get("is_gold") else ""
        score = f"score={h['score']:.5f}" if show_src else f"score={h['score']:.4f}"
        src   = ""
        if show_src:
            lex_r = f"lex:{h['lex_rank']}"   if h.get("lex_rank")   else "lex:-"
            den_r = f"den:{h['dense_rank']}"  if h.get("dense_rank") else "den:-"
            src   = f"  [{lex_r}|{den_r}]"
        lines.append(
            f"    #{h['rank']}  {score}{src}"
            f"  repo={h['repo']}  labels={h['labels']}{gold}"
        )
        lines.append("       " + h["snippet"].splitlines()[0][:68])
    return lines


def fmt_example(ex: dict, bucket: str, num: int) -> str:
    sep  = "─" * 76
    m    = ex["metrics"]
    lines = [
        sep,
        f"[{bucket.upper()} #{num}]  repo={ex['query_repo']}  split={ex['split']}",
        f"labels={ex['labels']}",
        (f"  Lex   MRR={m['lex']['mrr']:.3f}  R@5={m['lex']['r@5']:.3f}"
         f"  R@10={m['lex']['r@10']:.3f}"),
        (f"  Dense MRR={m['dense']['mrr']:.3f}  R@5={m['dense']['r@5']:.3f}"
         f"  R@10={m['dense']['r@10']:.3f}"),
        (f"  Hybrid MRR={m['hybrid']['mrr']:.3f}  R@5={m['hybrid']['r@5']:.3f}"
         f"  R@10={m['hybrid']['r@10']:.3f}"),
        "",
        "QUERY:",
        *textwrap.wrap(ex["query_snippet"].replace("\n", " "), 72),
        "",
        "GOLD RELEVANT:",
    ]
    for g in ex["gold_relevant"]:
        lines.append(f"  [{g['corpus_repo']}]  shared_labels={g['shared_labels']}")
        lines.append("  " + g["text_snippet"].splitlines()[0][:72])

    lines += fmt_hits(ex["lexical_top5"],  "LEXICAL")
    lines += fmt_hits(ex["dense_top5"],    "DENSE")
    lines += fmt_hits(ex["hybrid_top5"],   "HYBRID", show_src=True)
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading eval set and metrics …")
    with open(EVAL_PATH)    as f: eval_set = json.load(f)
    with open(METRICS_PATH) as f: metrics  = json.load(f)

    eval_map = {e["query_id"]: e for e in eval_set}

    buckets = classify(metrics)
    print(f"  hybrid_win         : {len(buckets['hybrid_win'])}")
    print(f"  dense_beats_hybrid : {len(buckets['dense_beats_hybrid'])}")
    print(f"  lex_beats_hybrid   : {len(buckets['lex_beats_hybrid'])}")
    print(f"  hybrid_ok          : {len(buckets['hybrid_ok'])}")

    print("\nLoading retrievers …")
    lex    = LexicalRetriever.load()
    dense  = DenseRetriever.load()
    device    = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    hybrid    = HybridRetriever(lex, dense, tokenizer, model, device,
                                pool=RETRIEVAL_POOL, rrf_k=RRF_K)

    # ── build examples ─────────────────────────────────────────────────────
    output: dict[str, list] = {}
    for bucket_name, metas in [
        ("hybrid_win",         buckets["hybrid_win"]),
        ("dense_beats_hybrid", buckets["dense_beats_hybrid"]),
        ("lex_beats_hybrid",   buckets["lex_beats_hybrid"]),
    ]:
        print(f"Building {len(metas)} '{bucket_name}' examples …")
        examples = []
        for meta in metas:
            entry = eval_map[meta["query_id"]]
            ex = build_example(meta, entry, lex, dense, hybrid,
                               tokenizer, model, device)
            examples.append(ex)
        output[bucket_name] = examples

    # ── save JSON ──────────────────────────────────────────────────────────
    with open(JSON_OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved JSON → {JSON_OUT}")

    # ── save TXT ───────────────────────────────────────────────────────────
    header = "\n".join([
        "HYBRID (RRF) vs LEXICAL vs DENSE — COMPARISON EXAMPLES",
        "RRF k=60 | pool=20 from each method | BAAI/bge-small-en-v1.5 + TF-IDF",
        f"Eval set: {len(eval_set)} queries (human-annotated)",
        "=" * 76,
    ])

    sections = [header]
    bucket_labels = [
        ("hybrid_win",         "HYBRID WINS  (hybrid MRR beats both lexical and dense)"),
        ("dense_beats_hybrid", "DENSE BEATS HYBRID  (RRF dilutes a strong dense signal)"),
        ("lex_beats_hybrid",   "LEXICAL BEATS HYBRID  (keyword match, RRF introduces noise)"),
    ]
    for key, title in bucket_labels:
        sections.append(f"\n{'=' * 76}")
        sections.append(f"{title:^76}")
        sections.append("=" * 76)
        for i, ex in enumerate(output[key], 1):
            sections.append(fmt_example(ex, key, i))

    TXT_OUT.write_text("\n".join(sections) + "\n")
    print(f"Saved TXT  → {TXT_OUT}")

    # ── console summary ────────────────────────────────────────────────────
    for key, title in bucket_labels:
        print(f"\n── {title} ──")
        for ex in output[key]:
            m = ex["metrics"]
            print(f"  [{ex['query_repo']:<13}]  "
                  f"lex_mrr={m['lex']['mrr']:.3f}  "
                  f"den_mrr={m['dense']['mrr']:.3f}  "
                  f"hyb_mrr={m['hybrid']['mrr']:.3f}  "
                  f"labels={ex['labels']}")


if __name__ == "__main__":
    main()
