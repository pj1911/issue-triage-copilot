"""
Build a human-reviewable candidate set for fair retrieval evaluation (Day 7).

Problem with the existing eval set
------------------------------------
Gold docs in retrieval_eval.json were selected by TF-IDF cosine similarity.
This means lexical retrieval trivially scores well because the "right answers"
are already the ones that TF-IDF finds most similar.  Dense retrieval can only
win if it happens to find those same TF-IDF-chosen docs.

Strategy for a fairer eval
---------------------------
1. Sample ~27 queries (3 per repo, all 9 repos) filtered to specific labels and
   moderate relevant-set sizes.  Avoids queries whose only label is "bug" (15k
   corpus docs) which are both noisy and biased toward lexical matching.

2. For each query, pool candidates from BOTH retrievers:
     - top-10 from lexical  (TF-IDF + cosine)
     - top-10 from dense    (BGE-small-en-v1.5)
     - up to 5 random label-matched docs not already in the pool

   Neither method dominates candidate selection, so human judgments can't be
   gamed by either retriever.

3. A human reviews each candidate and marks `human_relevant: true/false`.

4. Run finalize_human_eval.py to produce retrieval_eval_human.json, which the
   Day 7 evaluation scripts will use.

Outputs
-------
  data/processed/human_eval_candidates.json   structured for annotation
  data/processed/human_eval_review.md         human-readable review document

Run from project root:
    python -m src.retrieval.build_human_eval
"""
import json
import random
import time

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from src.retrieval.dense_retriever import _get_device, encode_texts, MODEL_NAME
from src.utils.paths import MODELS_DIR, PROCESSED_DIR

# ── paths ──────────────────────────────────────────────────────────────────────
DATASET_PATH    = PROCESSED_DIR / "dataset.parquet"
CORPUS_PATH     = PROCESSED_DIR / "retrieval_corpus.parquet"
VEC_PATH        = MODELS_DIR    / "tfidf_retrieval_vectorizer.joblib"
MATRIX_PATH     = MODELS_DIR    / "tfidf_retrieval_matrix.npz"
EMBEDDINGS_PATH = MODELS_DIR    / "dense_doc_embeddings.npy"
DENSE_IDS_PATH  = MODELS_DIR    / "dense_doc_ids.npy"

CANDIDATES_PATH = PROCESSED_DIR / "human_eval_candidates.json"
REVIEW_PATH     = PROCESSED_DIR / "human_eval_review.md"

# ── config ─────────────────────────────────────────────────────────────────────
QUERIES_PER_REPO = 3
TOP_K            = 10   # top-k from each retriever
N_RANDOM_EXTRA   = 5    # random label-matched docs added to pool
MIN_RELEVANT     = 50   # skip if fewer than this many label-matched corpus docs
MAX_RELEVANT     = 4000 # skip if relevant set is too large to be discriminative
RANDOM_SEED      = 7

_GENERIC_LABELS = {
    "bug", "feature-request", "help wanted", "good first issue",
    "enhancement", "proposal",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _build_label_index(corpus: pd.DataFrame) -> dict[str, list[int]]:
    """Map label string → list of corpus row positions."""
    idx: dict[str, list[int]] = {}
    for i, labels in enumerate(corpus["labels_clean"]):
        for label in labels:
            idx.setdefault(label, []).append(i)
    return idx


def _relevant_rows(query_labels: list[str], label_idx: dict) -> set[int]:
    """Corpus row positions that share at least one label with the query."""
    rows: set[int] = set()
    for label in query_labels:
        rows.update(label_idx.get(label, []))
    return rows


def _n_relevant_for_sampling(query_labels: list[str], label_idx: dict) -> int:
    """
    Relevant-set size used for sampling filter only.

    For multi-label queries that include at least one specific label, use the
    smallest specific-label count instead of the union.  This prevents a query
    like ["bug", "crash"] from being excluded just because the "bug" set is 15k;
    the crash-specific docs (769) are what make it discriminative.
    """
    specific = [l for l in query_labels if l not in _GENERIC_LABELS]
    if specific:
        return min(len(label_idx.get(l, [])) for l in specific)
    # All-generic fallback: use union size
    return len(_relevant_rows(query_labels, label_idx))


def _has_specific_label(labels) -> bool:
    return any(l not in _GENERIC_LABELS for l in labels)


# ── query sampling ─────────────────────────────────────────────────────────────

def _sample_queries(
    df: pd.DataFrame,
    label_idx: dict,
    rng: random.Random,
) -> pd.DataFrame:
    """
    Sample up to QUERIES_PER_REPO queries per repo from val/test splits.

    Priority:
      1. Queries with at least one specific label AND moderate relevant-set size
      2. Fall back to any labeled query that passes the size filter
    """
    chosen: list[pd.DataFrame] = []

    labeled = df[
        (df["split"].isin(["val", "test"])) &
        (df["labels_clean"].apply(len) > 0)
    ]

    for repo, group in labeled.groupby("repo"):
        # Prefer specific-label queries
        specific = group[group["labels_clean"].apply(_has_specific_label)].copy()
        pool = specific if len(specific) > 0 else group.copy()

        # Apply relevant-set size filter
        pool["_n_rel"] = pool["labels_clean"].apply(
            lambda ls: _n_relevant_for_sampling(list(ls), label_idx)
        )
        pool = pool[(pool["_n_rel"] >= MIN_RELEVANT) & (pool["_n_rel"] <= MAX_RELEVANT)]

        # If specific-label pool is too small, expand to all labeled
        if len(pool) < QUERIES_PER_REPO and len(specific) != len(group):
            fallback = group.copy()
            fallback["_n_rel"] = fallback["labels_clean"].apply(
                lambda ls: _n_relevant_for_sampling(list(ls), label_idx)
            )
            fallback = fallback[
                (fallback["_n_rel"] >= MIN_RELEVANT) &
                (fallback["_n_rel"] <= MAX_RELEVANT)
            ]
            if len(fallback) > len(pool):
                pool = fallback

        n = min(QUERIES_PER_REPO, len(pool))
        if n == 0:
            print(f"  [warn] no suitable queries for {repo} — skipping")
            continue

        sampled = pool.sample(n=n, random_state=rng.randint(0, 99999))
        chosen.append(sampled)

    return pd.concat(chosen, ignore_index=True) if chosen else pd.DataFrame()


# ── retrieval helpers ──────────────────────────────────────────────────────────

def _lex_retrieve(
    query_text: str,
    vectorizer,
    matrix: sp.csr_matrix,
    k: int,
) -> list[tuple[int, float]]:
    """Return (matrix_row_idx, score) for top-k lexical hits."""
    q_vec  = vectorizer.transform([query_text])
    scores = cosine_similarity(q_vec, matrix).ravel()
    top    = np.argpartition(scores, -k)[-k:]
    top    = top[np.argsort(scores[top])[::-1]]
    return [(int(idx), float(scores[idx])) for idx in top]


def _dense_retrieve(
    query_text: str,
    embeddings: np.ndarray,
    tokenizer,
    model,
    device: torch.device,
    k: int,
) -> list[tuple[int, float]]:
    """Return (embeddings_row_idx, score) for top-k dense hits."""
    q_emb  = encode_texts([query_text], tokenizer, model, device,
                          batch_size=1, show_progress=False)
    scores = (embeddings @ q_emb.T).ravel()
    top    = np.argpartition(scores, -k)[-k:]
    top    = top[np.argsort(scores[top])[::-1]]
    return [(int(idx), float(scores[idx])) for idx in top]


# ── candidate pool builder ─────────────────────────────────────────────────────

def _build_candidates(
    query_labels: list[str],
    lex_hits: list[tuple[int, float]],    # (corpus_row_idx, score)
    dense_hits: list[tuple[int, float]],  # (dense_arr_idx, score)
    corpus: pd.DataFrame,
    dense_ids: np.ndarray,                # dense_arr_idx → corpus_id
    id_to_row: dict[int, int],            # corpus_id → corpus row position
    relevant_rows: set[int],              # label-matched corpus row positions
    rng: random.Random,
) -> list[dict]:
    """
    Merge lexical + dense candidates + random extras into a deduplicated pool.

    Sorted: docs found by both methods first (most likely relevant), then
    lexical-only, then dense-only, then random label-matched.
    """
    pool: dict[int, dict] = {}  # corpus_id → metadata

    for rank, (lex_row, score) in enumerate(lex_hits, 1):
        cid = int(corpus.iloc[lex_row]["id"])
        pool.setdefault(cid, {})
        pool[cid]["lex_rank"]  = rank
        pool[cid]["lex_score"] = round(score, 4)

    for rank, (dense_idx, score) in enumerate(dense_hits, 1):
        cid = int(dense_ids[dense_idx])
        pool.setdefault(cid, {})
        pool[cid]["dense_rank"]  = rank
        pool[cid]["dense_score"] = round(score, 4)

    # Random label-matched extras not already in the pool
    pool_rows = {id_to_row[cid] for cid in pool if cid in id_to_row}
    extras = list(relevant_rows - pool_rows)
    rng.shuffle(extras)
    for row in extras[:N_RANDOM_EXTRA]:
        cid = int(corpus.iloc[row]["id"])
        pool.setdefault(cid, {})["random"] = True

    # Build final candidate records
    candidates: list[dict] = []
    for cid, meta in pool.items():
        row_idx = id_to_row.get(cid)
        if row_idx is None:
            continue

        row  = corpus.iloc[row_idx]
        text = str(row["text"])

        # Parse TITLE / BODY from the stored text format
        lines  = text.split("\n")
        title  = lines[0].replace("TITLE: ", "").strip()[:120]
        body   = " ".join(l.strip() for l in lines[2:] if l.strip())[:280]

        sources = []
        if "lex_rank" in meta:
            sources.append(f"lex:{meta['lex_rank']}")
        if "dense_rank" in meta:
            sources.append(f"dense:{meta['dense_rank']}")
        if meta.get("random") and not sources:
            sources.append("random")

        candidates.append({
            "corpus_id":      cid,
            "corpus_repo":    str(row["repo"]),
            "labels":         sorted(row["labels_clean"]),
            "shared_labels":  sorted(set(row["labels_clean"]) & set(query_labels)),
            "title":          title,
            "body_snippet":   body,
            "lex_rank":       meta.get("lex_rank"),
            "lex_score":      meta.get("lex_score"),
            "dense_rank":     meta.get("dense_rank"),
            "dense_score":    meta.get("dense_score"),
            "random_sample":  meta.get("random", False),
            "sources":        sources,
            "human_relevant": None,   # reviewer fills this in
        })

    # Sort: both-method hits first, then by best rank
    def _sort_key(c: dict) -> tuple:
        has_lex   = c["lex_rank"] is not None
        has_dense = c["dense_rank"] is not None
        both      = has_lex and has_dense
        best      = min(
            c["lex_rank"]   if has_lex   else 999,
            c["dense_rank"] if has_dense else 999,
        )
        return (-int(both), best)

    candidates.sort(key=_sort_key)
    return candidates


# ── markdown review doc ────────────────────────────────────────────────────────

def _write_review_md(eval_data: list[dict]) -> None:
    lines = [
        "# Human Eval Review — Issue Triage Copilot Day 7",
        "",
        "**How to annotate:**",
        "Open `data/processed/human_eval_candidates.json` and set",
        "`human_relevant` to `true` or `false` for each candidate.",
        "",
        "A candidate is **relevant** if it represents a similar issue — same",
        "problem class, same feature area, or would plausibly help triage the",
        "query.  Label overlap is a hint but not the rule: a well-written",
        "duplicate with no labels is still relevant.",
        "",
        "Candidate sources are noted as `[lex:N]`, `[dense:N]`, or `[random]`.",
        "Both-method candidates appear first (they are usually the most relevant).",
        "",
        "---",
        "",
    ]

    for i, entry in enumerate(eval_data, 1):
        label_str = ", ".join(entry["query_labels"])
        lines += [
            f"## Q{i:02d} · {entry['query_repo']} ({entry['query_split']}) "
            f"· `{label_str}`",
            f"*query_id: {entry['query_id']}  "
            f"· n_label_matched: {entry['n_label_matched']}*",
            "",
            "**Query:**",
            "```",
        ]
        text = entry["query_text"]
        lines.append(text[:600].replace("```", "'''"))
        if len(text) > 600:
            lines.append("[…truncated]")
        lines += ["```", ""]

        n_both    = sum(1 for c in entry["candidates"]
                        if c["lex_rank"] and c["dense_rank"])
        n_lex     = sum(1 for c in entry["candidates"]
                        if c["lex_rank"] and not c["dense_rank"])
        n_dense   = sum(1 for c in entry["candidates"]
                        if c["dense_rank"] and not c["lex_rank"])
        n_random  = sum(1 for c in entry["candidates"] if c["random_sample"])
        lines += [
            f"**{len(entry['candidates'])} candidates** "
            f"(both: {n_both} · lex-only: {n_lex} · "
            f"dense-only: {n_dense} · random: {n_random})",
            "",
        ]

        for j, c in enumerate(entry["candidates"], 1):
            src = f"[{', '.join(c['sources'])}]" if c["sources"] else "[?]"
            lines += [
                f"### C{j:02d} · {c['corpus_repo']} · {src}",
                f"**Title:** {c['title']}",
                f"**Snippet:** {c['body_snippet'][:240]}",
                f"**Labels:** `{c['labels']}`  "
                f"**Shared:** `{c['shared_labels']}`",
                f"> `human_relevant: null`  ← set to true / false in the JSON",
                "",
            ]

        lines += ["---", ""]

    REVIEW_PATH.write_text("\n".join(lines))
    print(f"Saved review doc   → {REVIEW_PATH}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = random.Random(RANDOM_SEED)

    # ── load data ──────────────────────────────────────────────────────────────
    print("Loading dataset and corpus …")
    df     = pd.read_parquet(DATASET_PATH)
    corpus = pd.read_parquet(CORPUS_PATH).reset_index(drop=True)

    label_idx  = _build_label_index(corpus)
    id_to_row  = {int(corpus.iloc[i]["id"]): i for i in range(len(corpus))}

    # ── sample queries ─────────────────────────────────────────────────────────
    print("Sampling queries …")
    queries = _sample_queries(df, label_idx, rng)
    print(f"  {len(queries)} queries  "
          f"({(queries['split'] == 'val').sum()} val / "
          f"{(queries['split'] == 'test').sum()} test)")
    print(f"  by repo: {dict(queries.groupby('repo').size())}")

    # ── load lexical index ─────────────────────────────────────────────────────
    print("Loading lexical index …")
    vectorizer = joblib.load(VEC_PATH)
    matrix     = sp.load_npz(str(MATRIX_PATH))

    # ── load dense embeddings ──────────────────────────────────────────────────
    print("Loading dense embeddings …")
    embeddings = np.load(str(EMBEDDINGS_PATH)).astype(np.float32)
    dense_ids  = np.load(str(DENSE_IDS_PATH))   # position → corpus_id

    # ── load BGE model for query encoding ─────────────────────────────────────
    device    = _get_device()
    print(f"Loading BGE model ({MODEL_NAME}) on {device} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    # ── build candidate pools ──────────────────────────────────────────────────
    print(f"\nGenerating candidates for {len(queries)} queries …")
    t0       = time.perf_counter()
    eval_data: list[dict] = []

    for i, (_, row) in enumerate(queries.iterrows(), 1):
        query_labels = list(row["labels_clean"])
        query_text   = str(row["text"])

        rel_rows   = _relevant_rows(query_labels, label_idx)
        lex_hits   = _lex_retrieve(query_text, vectorizer, matrix, TOP_K)
        dense_hits = _dense_retrieve(query_text, embeddings, tokenizer,
                                     model, device, TOP_K)

        candidates = _build_candidates(
            query_labels, lex_hits, dense_hits,
            corpus, dense_ids, id_to_row, rel_rows, rng,
        )

        n_both = sum(1 for c in candidates
                     if c["lex_rank"] is not None and c["dense_rank"] is not None)
        print(f"  [{i:2d}/{len(queries)}]  {row['repo']:<15}  "
              f"labels={query_labels}  "
              f"n_cand={len(candidates)}  both={n_both}")

        eval_data.append({
            "query_id":       int(row["id"]),
            "query_repo":     str(row["repo"]),
            "query_split":    str(row["split"]),
            "query_labels":   query_labels,
            "query_text":     query_text,
            "n_label_matched": len(rel_rows),
            "candidates":     candidates,
        })

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # ── save ───────────────────────────────────────────────────────────────────
    CANDIDATES_PATH.write_text(json.dumps(eval_data, indent=2))
    print(f"Saved candidates   → {CANDIDATES_PATH}")

    _write_review_md(eval_data)

    # ── summary ────────────────────────────────────────────────────────────────
    total      = sum(len(e["candidates"]) for e in eval_data)
    n_both     = sum(sum(1 for c in e["candidates"]
                         if c["lex_rank"] and c["dense_rank"])
                     for e in eval_data)
    n_lex_only = sum(sum(1 for c in e["candidates"]
                         if c["lex_rank"] and not c["dense_rank"])
                     for e in eval_data)
    n_den_only = sum(sum(1 for c in e["candidates"]
                         if c["dense_rank"] and not c["lex_rank"])
                     for e in eval_data)
    n_rand     = sum(sum(1 for c in e["candidates"] if c["random_sample"])
                     for e in eval_data)

    print(f"\nCandidate pool ({total} total across {len(eval_data)} queries):")
    print(f"  Both methods : {n_both}")
    print(f"  Lexical only : {n_lex_only}")
    print(f"  Dense only   : {n_den_only}")
    print(f"  Random extra : {n_rand}")
    print(f"\nNext steps:")
    print(f"  1. Review {REVIEW_PATH}")
    print(f"  2. Set human_relevant: true/false in {CANDIDATES_PATH}")
    print(f"  3. python -m src.retrieval.finalize_human_eval")


if __name__ == "__main__":
    main()
