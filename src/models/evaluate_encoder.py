"""
Final evaluation of the best encoder checkpoint on val and test.

Call this exactly once after training is complete and the threshold
has been tuned on val. Produces the comparison report against TF-IDF.

Run from project root:
    python -m src.models.evaluate_encoder

Artifacts saved:
    reports/encoder_metrics.json          same schema as tfidf_metrics.json
    reports/encoder_predictions_val.csv   id, true_labels, pred_labels, correct
    reports/encoder_predictions_test.csv  same for test
    reports/encoder_error_analysis.csv    test rows where prediction was wrong
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.models.dataset import IssueDataset
from src.models.metrics import eval_multilabel
from src.models.train_encoder import EncoderClassifier, predict_probs
from src.utils.paths import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR


def load_artifacts() -> tuple:
    """Load checkpoint, config, threshold, and MLB."""
    with open(MODELS_DIR / "encoder_config.json") as f:
        config = json.load(f)

    mlb       = joblib.load(MODELS_DIR / "labels_mlb.joblib")
    threshold = joblib.load(MODELS_DIR / "encoder_threshold.joblib")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    model = EncoderClassifier(config["model_name"], config["num_labels"]).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "encoder_best.pt", map_location=device))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    return model, tokenizer, mlb, threshold, config, device


def build_loader(df: pd.DataFrame, mlb, tokenizer, max_length: int, batch_size: int = 64, device: torch.device | None = None) -> tuple:
    y = mlb.transform(df["labels_clean"]).astype(np.float32)
    ds = IssueDataset(df["text"].tolist(), y, tokenizer, max_length)
    pin = device is not None and device.type == "cuda"
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)
    return loader, y


def predictions_df(
    df: pd.DataFrame,
    probs: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    classes: list[str],
) -> pd.DataFrame:
    """Build a per-issue predictions DataFrame."""
    y_pred = (probs >= threshold).astype(int)

    rows = []
    for i, (true_bin, pred_bin) in enumerate(zip(y_true, y_pred)):
        true_labels = [classes[j] for j, v in enumerate(true_bin) if v == 1]
        pred_labels = [classes[j] for j, v in enumerate(pred_bin) if v == 1]
        correct = set(true_labels) == set(pred_labels)
        rows.append({
            "id":          df.iloc[i]["id"],
            "repo":        df.iloc[i]["repo"],
            "true_labels": "|".join(sorted(true_labels)),
            "pred_labels": "|".join(sorted(pred_labels)),
            "correct":     correct,
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("Loading artifacts …")
    model, tokenizer, mlb, threshold, config, device = load_artifacts()
    classes: list[str] = list(mlb.classes_)
    max_length = config["max_length"]
    print(f"  model={config['model_name']}  threshold={threshold:.2f}  device={device}")

    # ── Load splits ───────────────────────────────────────────────────────────
    print("\nLoading data …")
    ds_full  = pd.read_parquet(PROCESSED_DIR / "dataset.parquet")
    val_df   = ds_full[ds_full["split"] == "val"].reset_index(drop=True)
    test_df  = ds_full[ds_full["split"] == "test"].reset_index(drop=True)
    print(f"  val={len(val_df):,}  test={len(test_df):,}")

    # ── Evaluate val ──────────────────────────────────────────────────────────
    print("\nEvaluating val …")
    val_loader, y_val = build_loader(val_df, mlb, tokenizer, max_length, device=device)
    val_probs, _      = predict_probs(model, val_loader, device)
    val_pred          = (val_probs >= threshold).astype(int)
    val_metrics       = eval_multilabel(y_val, val_pred, classes)
    val_metrics["threshold"] = threshold
    print(f"  micro_f1={val_metrics['micro_f1']:.4f}  macro_f1={val_metrics['macro_f1']:.4f}  "
          f"exact_match={val_metrics['exact_match_ratio']:.4f}")

    # ── Evaluate test ─────────────────────────────────────────────────────────
    print("\nEvaluating test …")
    test_loader, y_test = build_loader(test_df, mlb, tokenizer, max_length, device=device)
    test_probs, _       = predict_probs(model, test_loader, device)
    test_pred           = (test_probs >= threshold).astype(int)
    test_metrics        = eval_multilabel(y_test, test_pred, classes)
    test_metrics["threshold"] = threshold
    print(f"  micro_f1={test_metrics['micro_f1']:.4f}  macro_f1={test_metrics['macro_f1']:.4f}  "
          f"exact_match={test_metrics['exact_match_ratio']:.4f}")

    # ── Metrics report ────────────────────────────────────────────────────────
    report = {
        "model":      config["model_name"],
        "checkpoint": "models/encoder_best.pt",
        "config":     config,
        "labels": {
            "n_classes":        len(classes),
            "classes":          classes,
            "threshold_source": "tuned on val set only",
            "val":              val_metrics,
            "test":             test_metrics,
        },
    }
    metrics_path = REPORTS_DIR / "encoder_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved → {metrics_path}")

    # ── Predictions + error analysis ──────────────────────────────────────────
    val_preds_df  = predictions_df(val_df,  val_probs,  y_val,  threshold, classes)
    test_preds_df = predictions_df(test_df, test_probs, y_test, threshold, classes)

    val_preds_df.to_csv(REPORTS_DIR / "encoder_predictions_val.csv",  index=False)
    test_preds_df.to_csv(REPORTS_DIR / "encoder_predictions_test.csv", index=False)

    errors = test_preds_df[~test_preds_df["correct"]].copy()
    errors.to_csv(REPORTS_DIR / "encoder_error_analysis.csv", index=False)

    print(f"Saved → reports/encoder_predictions_val.csv   ({len(val_preds_df):,} rows)")
    print(f"Saved → reports/encoder_predictions_test.csv  ({len(test_preds_df):,} rows)")
    print(f"Saved → reports/encoder_error_analysis.csv    ({len(errors):,} errors)")

    # ── Quick comparison vs TF-IDF ────────────────────────────────────────────
    tfidf_path = REPORTS_DIR / "tfidf_metrics.json"
    if tfidf_path.exists():
        with open(tfidf_path) as f:
            tfidf = json.load(f)
        tfidf_val  = tfidf["labels"]["val"]["micro_f1"]
        tfidf_test = tfidf["labels"]["test"]["micro_f1"]
        print(f"\n── Comparison: micro-F1 ──────────────────")
        print(f"             val       test")
        print(f"  TF-IDF   {tfidf_val:.4f}    {tfidf_test:.4f}")
        print(f"  Encoder  {val_metrics['micro_f1']:.4f}    {test_metrics['micro_f1']:.4f}")
        delta_val  = val_metrics["micro_f1"]  - tfidf_val
        delta_test = test_metrics["micro_f1"] - tfidf_test
        print(f"  Delta   {delta_val:+.4f}   {delta_test:+.4f}")


if __name__ == "__main__":
    main()
