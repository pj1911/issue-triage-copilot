"""
Ablation: title-only vs title+body.

Reuses the saved encoder checkpoint — no retraining needed.
Runs inference twice on val:
  1. Full text (TITLE: ... \n\nBODY: ...) — already in dataset.parquet
  2. Title-only text (TITLE: ... only, body stripped)

Compares val micro-F1, macro-F1, precision, recall at the saved threshold.

Run from project root:
    python -m src.models.ablation_title_only

Saves: reports/ablation_title_only.json
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.models.dataset import IssueDataset
from src.models.metrics import eval_multilabel, sweep_thresholds, select_threshold
from src.models.train_encoder import EncoderClassifier, predict_probs
from src.utils.paths import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR


def main() -> None:
    with open(MODELS_DIR / "encoder_config.json") as f:
        config = json.load(f)

    mlb      = joblib.load(MODELS_DIR / "labels_mlb.joblib")
    classes  = list(mlb.classes_)
    device   = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    model = EncoderClassifier(config["model_name"], config["num_labels"]).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "encoder_best.pt", map_location=device))
    model.eval()

    tokenizer  = AutoTokenizer.from_pretrained(config["model_name"])
    max_length = config["max_length"]
    pin        = device.type == "cuda"

    print(f"Device: {device}")

    ds_full = pd.read_parquet(PROCESSED_DIR / "dataset.parquet")
    val_df  = ds_full[ds_full["split"] == "val"].reset_index(drop=True)
    y_val   = mlb.transform(val_df["labels_clean"]).astype(np.float32)

    def run(texts: list[str], label: str) -> dict:
        ds     = IssueDataset(texts, y_val, tokenizer, max_length)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=pin)
        probs, y_true = predict_probs(model, loader, device)
        thresh, micro = select_threshold(sweep_thresholds(probs, y_true))
        y_pred = (probs >= thresh).astype(int)
        m = eval_multilabel(y_true, y_pred, classes)
        print(f"  {label:<20} thr={thresh:.2f}  micro_f1={m['micro_f1']:.4f}  "
              f"macro_f1={m['macro_f1']:.4f}  prec={m.get('precision', '?')}  "
              f"recall={m.get('recall', '?')}")
        return {"threshold": thresh, **m}

    # Title + body (standard)
    print("\nRunning title+body …")
    full_results = run(val_df["text"].tolist(), "title+body")

    # Title only — extract from formatted text (format: "TITLE: ...\n\nBODY:\n...")
    print("Running title-only …")
    title_only_texts = [t.split("\n\nBODY:")[0] for t in val_df["text"]]
    title_results = run(title_only_texts, "title-only")

    # Summary
    delta = round(full_results["micro_f1"] - title_results["micro_f1"], 4)
    print(f"\n  Body contribution: micro-F1 delta = {delta:+.4f}")
    print(f"  {'title+body':<20} micro_f1={full_results['micro_f1']:.4f}")
    print(f"  {'title-only':<20} micro_f1={title_results['micro_f1']:.4f}")

    report = {
        "model":          config["model_name"],
        "checkpoint":     "models/encoder_best.pt",
        "split":          "val",
        "title_plus_body": full_results,
        "title_only":     title_results,
        "body_delta_micro_f1": delta,
    }
    out = REPORTS_DIR / "ablation_title_only.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
