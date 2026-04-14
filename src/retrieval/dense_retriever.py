"""
Dense retriever: BAAI/bge-small-en-v1.5 embeddings + cosine similarity.

Encodes the retrieval corpus once and saves:
    models/dense_doc_embeddings.npy   float32  (n_docs, 384)
    models/dense_doc_ids.npy          int64    (n_docs,)
    models/dense_model_config.json    metadata

Queries use the same raw issue text (title + body) as the lexical retriever.
Documents are encoded from doc_text (title + body + labels + repo), matching
the lexical setup exactly so Day 5 vs Day 6 comparisons are clean.

BGE-v1.5 notes
--------------
- Pooling : CLS token (index 0 of last hidden state)
- Normalize: L2 before cosine similarity (equivalent to dot product)
- No instruction prefix needed for v1.5 (unlike earlier BGE versions)
- Max tokens: 512  — long docs are truncated at the tokenizer

Run from project root:
    python -m src.retrieval.dense_retriever
"""
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.utils.paths import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR

# ── paths ──────────────────────────────────────────────────────────────────────
CORPUS_PATH   = PROCESSED_DIR / "retrieval_corpus.parquet"
EMBEDDINGS_PATH = MODELS_DIR  / "dense_doc_embeddings.npy"
IDS_PATH        = MODELS_DIR  / "dense_doc_ids.npy"
CONFIG_PATH     = MODELS_DIR  / "dense_model_config.json"
STATS_PATH      = REPORTS_DIR / "dense_retriever_stats.json"

# ── model ──────────────────────────────────────────────────────────────────────
MODEL_NAME  = "BAAI/bge-small-en-v1.5"
BATCH_SIZE  = 512   # A100 80GB: 512 comfortably fits; MPS/CPU: reduce to 128
MAX_LENGTH  = 512   # model hard limit


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── encoding ──────────────────────────────────────────────────────────────────

def encode_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode a list of strings → float32 numpy array (n, hidden_dim).

    Uses CLS-token pooling + L2 normalization, matching BGE's recommended
    usage with the HuggingFace transformers library.
    """
    all_embeddings = []
    n = len(texts)

    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        # CLS pooling: first token of last hidden state
        cls_embeddings = outputs.last_hidden_state[:, 0, :]          # (B, H)
        cls_embeddings = F.normalize(cls_embeddings, p=2, dim=1)     # L2 norm

        all_embeddings.append(cls_embeddings.cpu().float().numpy())

        if show_progress and (start // batch_size) % 20 == 0:
            pct = min(start + batch_size, n) / n * 100
            print(f"  encoded {min(start + batch_size, n):>6,} / {n:,}  ({pct:.0f}%)",
                  flush=True)

    return np.concatenate(all_embeddings, axis=0)  # (n, H)


# ── DenseRetriever class ───────────────────────────────────────────────────────

class DenseRetriever:
    """
    Cosine-similarity retriever backed by pre-computed corpus embeddings.

    After the index is built once (build() + save()), load it back with
    DenseRetriever.load() for fast per-query retrieval.
    """

    def __init__(
        self,
        embeddings: np.ndarray,   # (n_docs, H)  float32, L2-normalised
        corpus: pd.DataFrame,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.embeddings  = embeddings
        self.corpus      = corpus.reset_index(drop=True)
        self.model_name  = model_name
        self._id_to_idx  = {int(cid): i for i, cid in enumerate(corpus["id"])}

    # ------------------------------------------------------------------ #
    #  Query (tokenizer + model must be passed in; not stored on the obj) #
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        tokenizer: AutoTokenizer,
        model: AutoModel,
        device: torch.device,
        k: int = 20,
    ) -> pd.DataFrame:
        """
        Return the top-k most similar corpus documents for *query*.

        Parameters
        ----------
        query     : raw issue text (title + body).
        k         : number of results.

        Returns
        -------
        DataFrame: rank, score, id, repo, labels_clean, priority_clean, text
        """
        q_emb = encode_texts([query], tokenizer, model, device,
                             batch_size=1, show_progress=False)  # (1, H)
        scores = (self.embeddings @ q_emb.T).ravel()             # dot = cosine (L2-normed)

        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results = self.corpus.iloc[top_idx].copy()
        results.insert(0, "rank",  range(1, k + 1))
        results.insert(1, "score", scores[top_idx].round(4))
        return results[["rank", "score", "id", "repo", "labels_clean",
                         "priority_clean", "text"]]

    def id_to_idx(self, corpus_id: int) -> int | None:
        return self._id_to_idx.get(corpus_id)

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #
    def save(self) -> None:
        np.save(str(EMBEDDINGS_PATH), self.embeddings)
        np.save(str(IDS_PATH),        self.corpus["id"].values.astype(np.int64))
        config = {
            "model_name":  self.model_name,
            "pooling":     "cls",
            "normalize":   "l2",
            "max_length":  MAX_LENGTH,
            "hidden_dim":  int(self.embeddings.shape[1]),
            "n_docs":      int(self.embeddings.shape[0]),
            "dtype":       "float32",
        }
        CONFIG_PATH.write_text(json.dumps(config, indent=2))
        print(f"Saved embeddings → {EMBEDDINGS_PATH}  {self.embeddings.shape}")
        print(f"Saved ids        → {IDS_PATH}")
        print(f"Saved config     → {CONFIG_PATH}")

    @classmethod
    def load(cls) -> "DenseRetriever":
        embeddings = np.load(str(EMBEDDINGS_PATH)).astype(np.float32)
        ids        = np.load(str(IDS_PATH))
        config     = json.loads(CONFIG_PATH.read_text())
        corpus     = pd.read_parquet(CORPUS_PATH)
        # Re-order corpus to match saved id order (should already match)
        id_order   = {int(cid): i for i, cid in enumerate(ids)}
        corpus     = corpus.sort_values("id", key=lambda s: s.map(id_order))
        return cls(embeddings, corpus, model_name=config["model_name"])


# ── Build ──────────────────────────────────────────────────────────────────────

def build(corpus: pd.DataFrame) -> DenseRetriever:
    device    = _get_device()
    print(f"Device : {device}")
    print(f"Model  : {MODEL_NAME}")
    print(f"Loading tokenizer and model …")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f}M")

    texts = corpus["doc_text"].tolist()
    print(f"\nEncoding {len(texts):,} documents (batch_size={BATCH_SIZE}) …")
    t0 = time.perf_counter()
    embeddings = encode_texts(texts, tokenizer, model, device,
                              batch_size=BATCH_SIZE, show_progress=True)
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s  —  {len(texts)/elapsed:.0f} docs/s  "
          f"shape={embeddings.shape}  dtype={embeddings.dtype}")

    return DenseRetriever(embeddings, corpus, model_name=MODEL_NAME)


# ── Smoke test ─────────────────────────────────────────────────────────────────

_SMOKE_QUERIES = [
    "App crashes on startup with null pointer exception",
    "Add dark mode support to the settings panel",
    "Performance regression in the latest release",
    "Documentation for the new API endpoints is missing",
]


def _smoke_test(
    retriever: DenseRetriever,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
) -> list[dict]:
    results = []
    for query in _SMOKE_QUERIES:
        hits = retriever.retrieve(query, tokenizer, model, device, k=5)
        top  = hits.iloc[0]
        results.append({
            "query":             query,
            "top1_score":        float(top["score"]),
            "top1_repo":         top["repo"],
            "top1_labels":       list(top["labels_clean"]),
            "top1_text_snippet": top["text"][:120],
        })
        print(f"\nQuery : {query}")
        print(f"  #1  score={top['score']:.4f}  repo={top['repo']}  "
              f"labels={list(top['labels_clean'])}")
        print(f"       {top['text'][:120].splitlines()[0]}")
    return results


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    corpus = pd.read_parquet(CORPUS_PATH)
    print(f"Corpus loaded: {len(corpus):,} documents")

    retriever = build(corpus)
    retriever.save()

    # reload tokenizer/model for smoke test (already in memory; reuse)
    device    = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    print("\n── Smoke test ──")
    smoke = _smoke_test(retriever, tokenizer, model, device)

    stats = {
        "model_name":   MODEL_NAME,
        "n_documents":  int(len(corpus)),
        "embedding_shape": list(retriever.embeddings.shape),
        "device":       str(device),
        "batch_size":   BATCH_SIZE,
        "max_length":   MAX_LENGTH,
        "smoke_test":   smoke,
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2, default=list))
    print(f"\nSaved stats → {STATS_PATH}")


if __name__ == "__main__":
    main()
