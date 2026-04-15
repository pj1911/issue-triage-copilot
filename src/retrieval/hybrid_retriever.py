"""
Hybrid retriever: Reciprocal Rank Fusion (RRF) over lexical + dense results.

RRF formula (Cormack et al., 2009):
    score(d) = Σ  1 / (k + rank_m(d))
    for each method m that retrieved document d

where k=60 is the standard smoothing constant.

The hybrid retriever runs both retrievers independently (top-20 each),
then merges by RRF score. Documents retrieved by both methods get boosted;
documents missed by one method still appear if the other ranked them highly.

Usage
-----
    from src.retrieval.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever.load()
    results   = retriever.retrieve(query_text, k=20)

Run from project root (smoke test):
    python -m src.retrieval.hybrid_retriever
"""
import json

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
from src.retrieval.lexical_retriever import LexicalRetriever
from src.utils.paths import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR

# ── RRF constant ───────────────────────────────────────────────────────────────
# k=60: standard value from the original RRF paper. Higher k smooths rank
# differences; 60 is robust across a wide range of fusion tasks.
RRF_K = 60

# How many results to pull from each method before fusion
RETRIEVAL_POOL = 20


class HybridRetriever:
    """
    Combines a LexicalRetriever and a DenseRetriever via Reciprocal Rank Fusion.

    Both retrievers draw from the same corpus (71k train issues). The fused
    ranking is deterministic given the two ranked lists — no model training
    required.

    Parameters
    ----------
    lex     : fitted LexicalRetriever
    dense   : fitted DenseRetriever (pre-computed embeddings)
    tokenizer, model, device : BGE encoder for query-time dense encoding
    pool    : number of results to pull from each method before fusion
    rrf_k   : RRF smoothing constant (default 60)
    """

    def __init__(
        self,
        lex: LexicalRetriever,
        dense: DenseRetriever,
        tokenizer: AutoTokenizer,
        model: AutoModel,
        device: torch.device,
        pool: int = RETRIEVAL_POOL,
        rrf_k: int = RRF_K,
    ) -> None:
        self.lex       = lex
        self.dense     = dense
        self.tokenizer = tokenizer
        self.model     = model
        self.device    = device
        self.pool      = pool
        self.rrf_k     = rrf_k

    # ------------------------------------------------------------------ #
    #  RRF core
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rrf_fuse(
        ranked_lists: list[list[int]],
        rrf_k: int = RRF_K,
    ) -> list[tuple[int, float]]:
        """
        Fuse multiple ranked lists of corpus row indices via RRF.

        Parameters
        ----------
        ranked_lists : one list per method, each containing corpus row indices
                       ordered from rank-1 (best) downward.
        rrf_k        : smoothing constant.

        Returns
        -------
        List of (corpus_row_idx, rrf_score) sorted by rrf_score descending.
        """
        scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, idx in enumerate(ranked, start=1):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------ #
    #  Public retrieve interface
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, k: int = 20) -> pd.DataFrame:
        """
        Return the top-k fused results for *query*.

        Internally fetches `self.pool` results from each method, fuses by RRF,
        and returns the top-k rows from the corpus with an added rrf_score.

        Returns
        -------
        DataFrame with columns:
            rank, rrf_score, lex_rank, dense_rank,
            id, repo, labels_clean, priority_clean, text
        """
        pool = max(k, self.pool)  # always pull at least k from each method

        # ── lexical ───────────────────────────────────────────────────────
        lex_hits   = self.lex.retrieve(query, k=pool)
        lex_ids    = lex_hits["id"].tolist()
        lex_scores = dict(zip(lex_ids, lex_hits["score"].tolist()))
        # corpus row indices for RRF (lex matrix is in parquet/corpus order)
        lex_idxs   = [
            self.lex.corpus.index[self.lex.corpus["id"] == cid].item()
            for cid in lex_ids
        ]

        # ── dense ─────────────────────────────────────────────────────────
        dense_hits   = self.dense.retrieve(
            query, self.tokenizer, self.model, self.device, k=pool
        )
        dense_ids    = dense_hits["id"].tolist()
        dense_scores = dict(zip(dense_ids, dense_hits["score"].tolist()))
        # map corpus_id → embedding row index
        dense_emb_idxs = [self.dense._id_to_idx[int(cid)] for cid in dense_ids]

        # ── RRF fusion ────────────────────────────────────────────────────
        # Use embedding indices as the shared key space (both point into corpus)
        # For lex: corpus parquet index == embedding row order after load()
        fused = self._rrf_fuse([lex_idxs, dense_emb_idxs], self.rrf_k)

        # Build result table (top-k from fused list)
        rows = []
        lex_id_to_rank  = {cid: r for r, cid in enumerate(lex_ids,   start=1)}
        den_id_to_rank  = {cid: r for r, cid in enumerate(dense_ids, start=1)}

        # We need a mapping from embedding row idx → corpus_id
        emb_idx_to_id = {v: k for k, v in self.dense._id_to_idx.items()}

        # Also map lex corpus row idx → corpus_id
        lex_row_to_id  = self.lex.corpus["id"].to_dict()  # {row_idx: corpus_id}

        seen_ids: set[int] = set()
        for emb_idx, rrf_score in fused:
            corpus_id = emb_idx_to_id.get(emb_idx)
            if corpus_id is None:
                continue
            if corpus_id in seen_ids:
                continue
            seen_ids.add(corpus_id)

            # Look up corpus row in the dense retriever's corpus df
            row_df = self.dense.corpus[self.dense.corpus["id"] == corpus_id]
            if row_df.empty:
                continue

            row = row_df.iloc[0]
            rows.append({
                "rrf_score":     round(rrf_score, 6),
                "lex_rank":      lex_id_to_rank.get(corpus_id),
                "dense_rank":    den_id_to_rank.get(corpus_id),
                "id":            int(corpus_id),
                "repo":          row["repo"],
                "labels_clean":  row["labels_clean"],
                "priority_clean": row.get("priority_clean"),
                "text":          row["text"],
            })
            if len(rows) >= k:
                break

        result = pd.DataFrame(rows)
        result.insert(0, "rank", range(1, len(result) + 1))
        return result

    # ------------------------------------------------------------------ #
    #  Factory
    # ------------------------------------------------------------------ #
    @classmethod
    def load(
        cls,
        pool: int = RETRIEVAL_POOL,
        rrf_k: int = RRF_K,
    ) -> "HybridRetriever":
        """Load both retrievers from saved artifacts + initialise BGE encoder."""
        print("Loading lexical retriever …")
        lex = LexicalRetriever.load()

        print("Loading dense retriever …")
        dense = DenseRetriever.load()

        device    = _get_device()
        print(f"Loading BGE encoder ({MODEL_NAME}) on {device} …")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model     = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

        return cls(lex, dense, tokenizer, model, device, pool=pool, rrf_k=rrf_k)


# ── Smoke test ─────────────────────────────────────────────────────────────────

_SMOKE_QUERIES = [
    "App crashes on startup with null pointer exception",
    "Add dark mode support to the settings panel",
    "Documentation for the new API endpoints is missing",
]


def main() -> None:
    retriever = HybridRetriever.load()

    print("\n── Smoke test (top-5 per query) ──")
    for query in _SMOKE_QUERIES:
        hits = retriever.retrieve(query, k=5)
        print(f"\nQuery : {query}")
        for _, row in hits.iterrows():
            lex_r   = f"lex:{row['lex_rank']}"   if row['lex_rank']   else "lex:-"
            den_r   = f"den:{row['dense_rank']}"  if row['dense_rank'] else "den:-"
            print(f"  #{int(row['rank'])}  rrf={row['rrf_score']:.5f}"
                  f"  [{lex_r} {den_r}]"
                  f"  repo={row['repo']}"
                  f"  {row['text'][:80].splitlines()[0]}")


if __name__ == "__main__":
    main()
