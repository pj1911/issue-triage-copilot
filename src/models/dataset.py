"""
IssueDataset: lazily tokenizes issue text for encoder fine-tuning.

Reused by train_encoder.py and evaluate_encoder.py.
"""
import torch
from torch.utils.data import Dataset


class IssueDataset(Dataset):
    """
    Wraps a list of raw text strings and binary multi-label targets.

    Tokenization is lazy (per __getitem__) so the dataset itself is
    cheap to construct even for 70k+ examples.
    """

    def __init__(
        self,
        texts: list[str],
        labels_bin,          # np.ndarray shape (n, n_classes), dtype float32
        tokenizer,
        max_length: int = 256,
    ) -> None:
        self.texts = texts
        self.labels = torch.tensor(labels_bin, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         self.labels[idx],
        }
