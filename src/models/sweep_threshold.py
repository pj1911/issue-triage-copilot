"""
Threshold sweep on the current saved encoder checkpoint.

Evaluates thresholds 0.10 – 0.40 on both val and test, logging:
  micro-F1, macro-F1, precision, recall,
  predicted-positive rate, mean labels per issue

Selects threshold on val only (never test), then reports test at that threshold.

Run from project root:
    python -m src.models.sweep_threshold

Artifacts saved:
    reports/threshold_sweep.json
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
from src.models.metrics import THRESHOLD_GRID, select_threshold, sweep_thresholds
from src.models.train_encoder import EncoderClassifier, predict_probs
from src.utils.paths import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR


def main() -> None:
    # ── Load artifacts ────────────────────────────────────────────────────────
    print("Loading artifacts …")
    with open(MODELS_DIR / "encoder_config.json") as f:
        config = json.load(f)

    mlb       = joblib.load(MODELS_DIR / "labels_mlb.joblib")
    classes   = list(mlb.classes_)
    device    = torch.device(
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

    print(f"  model={config['model_name']}  device={device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data …")
    ds_full = pd.read_parquet(PROCESSED_DIR / "dataset.parquet")
    val_df  = ds_full[ds_full["split"] == "val"].reset_index(drop=True)
    test_df = ds_full[ds_full["split"] == "test"].reset_index(drop=True)

    def make_loader(df):
        y  = mlb.transform(df["labels_clean"]).astype(np.float32)
        ds = IssueDataset(df["text"].tolist(), y, tokenizer, max_length)
        return DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=pin), y

    val_loader,  y_val  = make_loader(val_df)
    test_loader, y_test = make_loader(test_df)

    # ── Run inference once ────────────────────────────────────────────────────
    print("Running inference on val …")
    val_probs,  _ = predict_probs(model, val_loader,  device)
    print("Running inference on test …")
    test_probs, _ = predict_probs(model, test_loader, device)

    # ── Sweep val ─────────────────────────────────────────────────────────────
    print("\nVal threshold sweep:")
    print(f"  {'thr':>5}  {'micro_f1':>8}  {'macro_f1':>8}  {'prec':>6}  {'rec':>6}  {'pos_rate':>8}  {'mean_lbl':>8}")
    val_rows = sweep_thresholds(val_probs, y_val, THRESHOLD_GRID)
    for r in val_rows:
        print(f"  {r['threshold']:>5.2f}  {r['micro_f1']:>8.4f}  {r['macro_f1']:>8.4f}  "
              f"{r['precision']:>6.4f}  {r['recall']:>6.4f}  "
              f"{r['pred_pos_rate']:>8.4f}  {r['mean_labels']:>8.4f}")

    # ── Select threshold on val ───────────────────────────────────────────────
    best_thresh, best_val_micro = select_threshold(val_rows)
    print(f"\nSelected threshold: {best_thresh:.2f}  val micro-F1={best_val_micro:.4f}")

    # ── Apply to test ─────────────────────────────────────────────────────────
    # Also compute the full test sweep for inspection (NOT for selection)
    print("\nTest threshold sweep (for inspection only — not used for selection):")
    print(f"  {'thr':>5}  {'micro_f1':>8}  {'macro_f1':>8}  {'prec':>6}  {'rec':>6}  {'pos_rate':>8}  {'mean_lbl':>8}")
    test_rows = sweep_thresholds(test_probs, y_test, THRESHOLD_GRID)
    for r in test_rows:
        marker = " ←" if abs(r["threshold"] - best_thresh) < 0.001 else ""
        print(f"  {r['threshold']:>5.2f}  {r['micro_f1']:>8.4f}  {r['macro_f1']:>8.4f}  "
              f"{r['precision']:>6.4f}  {r['recall']:>6.4f}  "
              f"{r['pred_pos_rate']:>8.4f}  {r['mean_labels']:>8.4f}{marker}")

    # ── Compare vs TF-IDF ─────────────────────────────────────────────────────
    test_at_best = next(r for r in test_rows if abs(r["threshold"] - best_thresh) < 0.001)
    tfidf_path   = REPORTS_DIR / "tfidf_metrics.json"
    if tfidf_path.exists():
        with open(tfidf_path) as f:
            tfidf = json.load(f)
        tfidf_val  = tfidf["labels"]["val"]["micro_f1"]
        tfidf_test = tfidf["labels"]["test"]["micro_f1"]
        print(f"\n── Comparison at thr={best_thresh:.2f} ──────────────────")
        print(f"             val       test")
        print(f"  TF-IDF   {tfidf_val:.4f}    {tfidf_test:.4f}")
        print(f"  Encoder  {best_val_micro:.4f}    {test_at_best['micro_f1']:.4f}")
        print(f"  Delta   {best_val_micro - tfidf_val:+.4f}   {test_at_best['micro_f1'] - tfidf_test:+.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    report = {
        "model":             config["model_name"],
        "checkpoint":        "models/encoder_best.pt",
        "selected_threshold": best_thresh,
        "val_sweep":         val_rows,
        "test_sweep":        test_rows,
        "val_at_selected":   next(r for r in val_rows  if abs(r["threshold"] - best_thresh) < 0.001),
        "test_at_selected":  test_at_best,
    }
    out = REPORTS_DIR / "threshold_sweep.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
