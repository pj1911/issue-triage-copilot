"""
Fine-tune a compact encoder (DistilBERT) for multi-label issue labeling.

Evaluation protocol is frozen from Day 3:
  - same train/val split (repo-level, from data/processed/dataset.parquet)
  - same label vocab (models/labels_mlb.joblib)
  - same primary metric: val micro-F1
  - threshold tuned on val only, never on test

Run from project root:
    python -m src.models.train_encoder

Key flags:
    --model_name    HuggingFace model ID  (default: distilbert-base-uncased)
    --epochs        number of training epochs  (default: 4)
    --batch_size    per-device batch size      (default: 32)
    --lr            peak learning rate         (default: 2e-5)
    --max_length    tokenizer max tokens       (default: 256)
    --max_train     cap training rows for a dry run (default: 0 = all)

Artifacts saved:
    models/encoder_best.pt         model state dict (best val micro-F1)
    models/encoder_config.json     hyperparams + val metric at checkpoint
    models/encoder_threshold.joblib  threshold tuned on val
    reports/encoder_train_log.json   per-epoch metrics
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.models.dataset import IssueDataset
from src.models.metrics import eval_multilabel, tune_threshold
from src.utils.paths import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR


# ── Model ─────────────────────────────────────────────────────────────────────

class EncoderClassifier(nn.Module):
    """Pretrained encoder + single linear classification head."""

    def __init__(self, model_name: str, num_labels: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # [CLS] token
        return self.classifier(cls)             # raw logits


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs, y_true) for all batches in loader."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch["labels"].numpy())
    return np.vstack(all_probs), np.vstack(all_labels)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune encoder for multi-label issue labeling")
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--epochs",     type=int,   default=6)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=5e-6)
    parser.add_argument("--max_length", type=int,   default=256)
    parser.add_argument("--warmup_pct", type=float, default=0.10,
                        help="Fraction of total steps used for LR warmup")
    parser.add_argument("--max_train",  type=int,   default=0,
                        help="Cap training rows (0 = use all; set small for a dry run)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data …")
    ds = pd.read_parquet(PROCESSED_DIR / "dataset.parquet")
    train_df = ds[ds["split"] == "train"].reset_index(drop=True)
    val_df   = ds[ds["split"] == "val"].reset_index(drop=True)

    if args.max_train > 0:
        train_df = train_df.sample(n=min(args.max_train, len(train_df)),
                                   random_state=42).reset_index(drop=True)
        print(f"  [dry run] capped train to {len(train_df):,} rows")

    print(f"  train={len(train_df):,}  val={len(val_df):,}")

    # ── Labels — reuse Day-3 MLB for identical vocab / ordering ───────────────
    mlb = joblib.load(MODELS_DIR / "labels_mlb.joblib")
    classes: list[str] = list(mlb.classes_)
    num_labels = len(classes)
    print(f"  Labels: {num_labels} classes")

    y_train = mlb.transform(train_df["labels_clean"]).astype(np.float32)
    y_val   = mlb.transform(val_df["labels_clean"]).astype(np.float32)

    # ── Tokenizer + datasets ──────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {args.model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = IssueDataset(train_df["text"].tolist(), y_train, tokenizer, args.max_length)
    val_ds   = IssueDataset(val_df["text"].tolist(),   y_val,   tokenizer, args.max_length)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nBuilding model: {args.model_name} + linear head ({num_labels} classes) …")
    model = EncoderClassifier(args.model_name, num_labels).to(device)

    total_params   = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_pct)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.BCEWithLogitsLoss()

    # AMP only on CUDA
    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"  AMP: {use_amp}")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_micro  = -1.0
    best_epoch      = 0
    epoch_log: list[dict] = []

    print(f"\nTraining {args.epochs} epochs …")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(input_ids, attention_mask)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids, attention_mask)
                loss   = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        elapsed  = time.time() - t0

        # ── Val eval: tune threshold per epoch for accurate checkpoint selection ──
        val_probs, val_true = predict_probs(model, val_loader, device)
        epoch_thresh, epoch_micro = tune_threshold(val_probs, val_true)
        val_pred = (val_probs >= epoch_thresh).astype(int)
        val_metrics = eval_multilabel(val_true, val_pred, classes)

        print(
            f"Epoch {epoch}  loss={avg_loss:.4f}  "
            f"val_micro_f1={epoch_micro:.4f}  "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}  "
            f"thr={epoch_thresh:.2f}  [{elapsed:.0f}s]"
        )

        entry = {"epoch": epoch, "train_loss": round(avg_loss, 4),
                 "elapsed_s": round(elapsed, 1),
                 "val_threshold": epoch_thresh, **val_metrics}
        epoch_log.append(entry)

        # Save per-epoch val predictions (probs only — apply any threshold later)
        np.save(MODELS_DIR / f"val_probs_epoch{epoch}.npy", val_probs)

        # Save best checkpoint by tuned val micro-F1
        if epoch_micro > best_val_micro:
            best_val_micro = epoch_micro
            best_epoch     = epoch
            torch.save(model.state_dict(), MODELS_DIR / "encoder_best.pt")
            print(f"  ✓ New best checkpoint (val micro-F1={best_val_micro:.4f})")

    print(f"\nBest checkpoint: epoch {best_epoch}  val micro-F1={best_val_micro:.4f}")

    # ── Tune threshold on val using best checkpoint ───────────────────────────
    print("\nTuning threshold on val (best checkpoint) …")
    model.load_state_dict(torch.load(MODELS_DIR / "encoder_best.pt", map_location=device))
    val_probs, val_true = predict_probs(model, val_loader, device)

    best_thresh, tuned_micro = tune_threshold(val_probs, val_true)
    print(f"  Best threshold: {best_thresh:.2f}  val micro-F1={tuned_micro:.4f}")
    joblib.dump(best_thresh, MODELS_DIR / "encoder_threshold.joblib")

    # ── Cost tracking ─────────────────────────────────────────────────────────
    checkpoint_size_mb = round(
        (MODELS_DIR / "encoder_best.pt").stat().st_size / 1e6, 1
    )
    total_train_time_s = sum(e.get("elapsed_s", 0) for e in epoch_log)

    gpu_mem_gb = None
    if device.type == "cuda":
        gpu_mem_gb = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)

    # ── Save config ───────────────────────────────────────────────────────────
    config = {
        "model_name":              args.model_name,
        "num_labels":              num_labels,
        "max_length":              args.max_length,
        "epochs":                  args.epochs,
        "batch_size":              args.batch_size,
        "lr":                      args.lr,
        "best_epoch":              best_epoch,
        "best_val_micro_f1_thr05": round(best_val_micro, 4),
        "best_val_micro_f1_tuned": round(tuned_micro, 4),
        "best_threshold":          best_thresh,
        "cost": {
            "checkpoint_size_mb":    checkpoint_size_mb,
            "total_train_time_s":    total_train_time_s,
            "gpu_peak_memory_gb":    gpu_mem_gb,
            "device":                str(device),
        },
    }
    with open(MODELS_DIR / "encoder_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Save training log ─────────────────────────────────────────────────────
    with open(REPORTS_DIR / "encoder_train_log.json", "w") as f:
        json.dump({"args": vars(args), "epochs": epoch_log}, f, indent=2)

    print(f"\nArtifacts saved:")
    print(f"  models/encoder_best.pt")
    print(f"  models/encoder_config.json")
    print(f"  models/encoder_threshold.joblib")
    print(f"  reports/encoder_train_log.json")


if __name__ == "__main__":
    main()
