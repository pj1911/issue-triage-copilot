"""
Lexical retriever: TF-IDF index over the retrieval corpus + cosine similarity.

Index is built on doc_text (title + body + labels + repo) so label tokens
and repo names contribute to matching.

Queries use the raw issue text (title + body only) — labels/repo are not known
at query time for unseen issues.

Artifacts saved to:
    models/tfidf_retrieval_vectorizer.joblib   fitted TF-IDF vectorizer
    models/tfidf_retrieval_matrix.npz          sparse document matrix (71k × vocab)
    models/tfidf_retrieval_ids.npy             corpus row → issue id mapping

Run from project root:
    python -m src.retrieval.lexical_retriever
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.paths import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR

CORPUS_PATH     = PROCESSED_DIR / "retrieval_corpus.parquet"
VEC_PATH        = MODELS_DIR    / "tfidf_retrieval_vectorizer.joblib"
MATRIX_PATH     = MODELS_DIR    / "tfidf_retrieval_matrix.npz"
IDS_PATH        = MODELS_DIR    / "tfidf_retrieval_ids.npy"
STATS_PATH      = REPORTS_DIR   / "lexical_retriever_stats.json"

# Slightly narrower vocabulary than the classifier — retrieval benefits more
# from common n-grams than rare technical tokens.
VECTORIZER_KWARGS = dict(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=150_000,
    sublinear_tf=True,
)


class LexicalRetriever:
    """TF-IDF + cosine similarity retriever backed by the train corpus."""

    def __init__(self, vectorizer: TfidfVectorizer, matrix: sp.csr_matrix,
                 corpus: pd.DataFrame) -> None:
        self.vectorizer = vectorizer
        self.matrix     = matrix          # (n_docs, vocab)
        self.corpus     = corpus.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Query
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, k: int = 10) -> pd.DataFrame:
        """
        Return the top-k most similar corpus documents for *query*.

        Parameters
        ----------
        query : str
            Raw issue text (title + body).  Do NOT include labels — those
            are unknown at query time.
        k     : int
            Number of results to return.

        Returns
        -------
        DataFrame with columns:
            rank, score, id, repo, labels_clean, priority_clean, text
        """
        q_vec   = self.vectorizer.transform([query])            # (1, vocab)
        scores  = cosine_similarity(q_vec, self.matrix).ravel() # (n_docs,)
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results = self.corpus.iloc[top_idx].copy()
        results.insert(0, "rank",  range(1, k + 1))
        results.insert(1, "score", scores[top_idx].round(4))
        return results[["rank", "score", "id", "repo", "labels_clean",
                         "priority_clean", "text"]]

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #
    def save(self) -> None:
        joblib.dump(self.vectorizer, VEC_PATH)
        sp.save_npz(str(MATRIX_PATH), self.matrix)
        np.save(str(IDS_PATH), self.corpus["id"].values)
        print(f"Saved vectorizer → {VEC_PATH}")
        print(f"Saved matrix     → {MATRIX_PATH}  {self.matrix.shape}")
        print(f"Saved ids        → {IDS_PATH}")

    @classmethod
    def load(cls) -> "LexicalRetriever":
        vectorizer = joblib.load(VEC_PATH)
        matrix     = sp.load_npz(str(MATRIX_PATH))
        corpus     = pd.read_parquet(CORPUS_PATH)
        return cls(vectorizer, matrix, corpus)


# ------------------------------------------------------------------ #
#  Build
# ------------------------------------------------------------------ #
def build(corpus: pd.DataFrame) -> LexicalRetriever:
    print(f"Fitting TF-IDF on {len(corpus):,} documents …")
    t0          = time.perf_counter()
    vectorizer  = TfidfVectorizer(**VECTORIZER_KWARGS)
    matrix      = vectorizer.fit_transform(corpus["doc_text"])
    elapsed     = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s  —  matrix {matrix.shape}  "
          f"({matrix.nnz:,} non-zeros)")
    return LexicalRetriever(vectorizer, matrix.tocsr(), corpus)


# ------------------------------------------------------------------ #
#  Smoke test: top-10 for a handful of example queries
# ------------------------------------------------------------------ #
_SMOKE_QUERIES = [
    "App crashes on startup with null pointer exception",
    "Add dark mode support to the settings panel",
    "Performance regression in the latest release",
    "Documentation for the new API endpoints is missing",
]


def _smoke_test(retriever: LexicalRetriever) -> list[dict]:
    results = []
    for query in _SMOKE_QUERIES:
        hits = retriever.retrieve(query, k=5)
        top  = hits.iloc[0]
        results.append({
            "query":      query,
            "top1_score": float(top["score"]),
            "top1_repo":  top["repo"],
            "top1_labels": top["labels_clean"],
            "top1_text_snippet": top["text"][:120],
        })
        print(f"\nQuery : {query}")
        print(f"  #1  score={top['score']:.4f}  repo={top['repo']}  "
              f"labels={top['labels_clean']}")
        print(f"       {top['text'][:120].splitlines()[0]}")
    return results


def main() -> None:
    corpus    = pd.read_parquet(CORPUS_PATH)
    retriever = build(corpus)
    retriever.save()

    print("\n── Smoke test ──")
    smoke = _smoke_test(retriever)

    stats = {
        "n_documents":  int(len(corpus)),
        "vocab_size":   int(len(retriever.vectorizer.vocabulary_)),
        "matrix_shape": list(retriever.matrix.shape),
        "matrix_nnz":   int(retriever.matrix.nnz),
        "vectorizer_params": VECTORIZER_KWARGS,
        "smoke_test":   smoke,
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2, default=list)
    print(f"\nSaved stats → {STATS_PATH}")


if __name__ == "__main__":
    main()
