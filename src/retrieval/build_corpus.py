"""
Build retrieval corpus from train issues only.

Each document in the corpus encodes:
  - text      : preprocessed title + body (from dataset.parquet)
  - labels    : cleaned label list
  - repo      : source repository
  - doc_text  : retrieval-ready string that includes title, body, labels,
                and repo so a BM25 or dense retriever can match on all fields

Output:
  data/processed/retrieval_corpus.parquet
  reports/corpus_stats.json

Only train-split issues are included — no val/test contamination.
"""
import json

import pandas as pd

from src.utils.paths import PROCESSED_DIR, REPORTS_DIR

DATASET_PATH = PROCESSED_DIR / "dataset.parquet"
CORPUS_PATH  = PROCESSED_DIR / "retrieval_corpus.parquet"
STATS_PATH   = REPORTS_DIR / "corpus_stats.json"


def build_doc_text(row: pd.Series) -> str:
    """
    Concatenate title+body text, labels, and repo into a single string
    for retrieval.  Labels and repo are appended as plain tokens so any
    lexical retriever (BM25, TF-IDF) can match on them; dense retrievers
    will encode the whole string.

    Format:
        <preprocessed title+body text>

        LABELS: label-a label-b label-c
        REPO: owner/name
    """
    labels    = row["labels_clean"]
    label_str = " ".join(labels) if len(labels) > 0 else ""
    label_line = f"LABELS: {label_str}" if label_str else "LABELS:"
    repo_line  = f"REPO: {row['repo']}"
    return f"{row['text']}\n\n{label_line}\n{repo_line}"


def main() -> None:
    df = pd.read_parquet(DATASET_PATH)
    train = df[df["split"] == "train"].copy()
    print(f"Train issues loaded: {len(train):,}")

    train["doc_text"] = train.apply(build_doc_text, axis=1)

    corpus = train[[
        "id",
        "repo",
        "text",
        "labels_clean",
        "priority_clean",
        "doc_text",
    ]].reset_index(drop=True)

    corpus.to_parquet(CORPUS_PATH, index=False)
    print(f"Corpus saved → {CORPUS_PATH}  ({len(corpus):,} documents)")

    # ── stats ──
    corpus["doc_len"]    = corpus["doc_text"].str.len()
    corpus["n_labels"]   = corpus["labels_clean"].apply(len)
    labeled              = corpus[corpus["n_labels"] > 0]

    stats = {
        "n_documents":       int(len(corpus)),
        "n_repos":           int(corpus["repo"].nunique()),
        "repos":             sorted(corpus["repo"].unique().tolist()),
        "labeled_pct":       round(float(len(labeled) / len(corpus) * 100), 1),
        "n_labeled":         int(len(labeled)),
        "n_unlabeled":       int((corpus["n_labels"] == 0).sum()),
        "doc_len": {
            "mean":   round(float(corpus["doc_len"].mean()), 1),
            "median": round(float(corpus["doc_len"].median()), 1),
            "p95":    round(float(corpus["doc_len"].quantile(0.95)), 1),
            "max":    int(corpus["doc_len"].max()),
        },
        "n_labels_per_doc": {
            "mean":   round(float(corpus["n_labels"].mean()), 2),
            "median": round(float(corpus["n_labels"].median()), 1),
            "max":    int(corpus["n_labels"].max()),
        },
        "label_freq": (
            corpus["labels_clean"]
            .explode()
            .dropna()
            .value_counts()
            .head(20)
            .to_dict()
        ),
        "docs_per_repo": (
            corpus.groupby("repo").size().sort_values(ascending=False).to_dict()
        ),
    }

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nCorpus stats:")
    print(f"  documents  : {stats['n_documents']:,}")
    print(f"  repos      : {stats['n_repos']}")
    print(f"  labeled    : {stats['n_labeled']:,}  ({stats['labeled_pct']}%)")
    print(f"  unlabeled  : {stats['n_unlabeled']:,}")
    print(f"  doc_len    : mean={stats['doc_len']['mean']}  "
          f"p95={stats['doc_len']['p95']}  max={stats['doc_len']['max']}")
    print(f"  labels/doc : mean={stats['n_labels_per_doc']['mean']}")
    print(f"\nSaved stats → {STATS_PATH}")


if __name__ == "__main__":
    main()
