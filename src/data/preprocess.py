"""
Preprocessing pipeline: raw parquet → model-ready text.

Reads train/val/test from data/processed/, formats each issue as:

    TITLE: <title>

    BODY:
    <body>

Light cleaning only — no aggressive normalization.
Outputs data/processed/dataset.parquet with all splits combined,
plus a text_stats report.
"""
import json
import re
import pandas as pd

from src.utils.paths import PROCESSED_DIR, REPORTS_DIR

SPLITS = {
    "train": PROCESSED_DIR / "train.parquet",
    "val":   PROCESSED_DIR / "val.parquet",
    "test":  PROCESSED_DIR / "test.parquet",
}
OUTPUT_PATH = PROCESSED_DIR / "dataset.parquet"
STATS_PATH  = REPORTS_DIR / "text_stats.json"

# HTML tags that add pure noise with no semantic content
_HTML_NOISE = re.compile(
    r"<(script|style|head)[^>]*>.*?</\1>",
    flags=re.DOTALL | re.IGNORECASE,
)
# Collapse 4+ consecutive blank lines → 2 (preserve paragraph breaks)
_EXCESS_BLANK = re.compile(r"\n{4,}")
# Collapse 200+ repeated dashes/equals (horizontal rules in issue templates)
_HR = re.compile(r"[-=*]{200,}")


def light_clean(text: str) -> str:
    """Remove only severe noise. Do not strip markdown or code blocks."""
    text = _HTML_NOISE.sub(" ", text)
    text = _HR.sub("---", text)
    text = _EXCESS_BLANK.sub("\n\n", text)
    return text.strip()


def format_text(title: str, body: str) -> str:
    title = (title or "").strip()
    body  = light_clean(body or "")
    return f"TITLE: {title}\n\nBODY:\n{body}"


def normalize_priority(p: str) -> str:
    return p.strip().lower() if isinstance(p, str) else "unknown"


def main() -> None:
    frames = []
    for split_name, path in SPLITS.items():
        df = pd.read_parquet(path)
        df["split"] = split_name
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # ── format text ──
    df["body"] = df["body"].fillna("").astype(str)
    df["text"] = df.apply(
        lambda row: format_text(row["title"], row["body"]), axis=1
    )

    # ── normalize priority ──
    df["priority_clean"] = df["priority"].apply(normalize_priority)

    # ── select output columns ──
    out = df[[
        "id",
        "repo",
        "text",
        "labels",        # already cleaned list from split.py
        "priority_clean",
        "split",
    ]].rename(columns={"labels": "labels_clean"})

    out.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(out):,} rows → {OUTPUT_PATH}")

    # ── text length stats ──
    out["text_len"] = out["text"].str.len()
    out["title_len"] = df["title"].fillna("").str.len()
    out["body_len"]  = df["body"].fillna("").str.len()
    out["n_labels"]  = out["labels_clean"].apply(len)

    def stat_block(series: pd.Series) -> dict:
        return {
            "mean":   round(float(series.mean()), 1),
            "median": round(float(series.median()), 1),
            "p95":    round(float(series.quantile(0.95)), 1),
            "max":    int(series.max()),
        }

    stats: dict = {}
    for split_name in ["train", "val", "test"]:
        mask = out["split"] == split_name
        sub  = out[mask]
        stats[split_name] = {
            "n_rows":    int(len(sub)),
            "text_len":  stat_block(sub["text_len"]),
            "title_len": stat_block(sub["title_len"]),
            "body_len":  stat_block(sub["body_len"]),
            "n_labels":  stat_block(sub["n_labels"]),
            "empty_labels_pct": round(
                float((sub["n_labels"] == 0).mean() * 100), 1
            ),
            "priority_dist": sub["priority_clean"].value_counts().to_dict(),
        }

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    # ── print summary ──
    for split_name, s in stats.items():
        print(f"\n── {split_name} ({s['n_rows']:,} rows) ──")
        print(f"  text_len   mean={s['text_len']['mean']}  "
              f"median={s['text_len']['median']}  "
              f"p95={s['text_len']['p95']}  "
              f"max={s['text_len']['max']}")
        print(f"  n_labels   mean={s['n_labels']['mean']}  "
              f"empty={s['empty_labels_pct']}%")
        print(f"  priority   {s['priority_dist']}")

    print(f"\nSaved stats → {STATS_PATH}")


if __name__ == "__main__":
    main()
