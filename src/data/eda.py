import json
import pandas as pd

from src.utils.paths import RAW_DIR, REPORTS_DIR

INPUT_PATH = RAW_DIR / "issues.parquet"
OUTPUT_PATH = REPORTS_DIR / "day1_eda.json"

def safe_top_values(series: pd.Series, n: int = 10):
    try:
        return series.astype(str).value_counts(dropna=False).head(n).to_dict()
    except Exception:
        return {}

def main() -> None:
    df = pd.read_parquet(INPUT_PATH)

    summary = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "null_fraction": {
            col: float(df[col].isna().mean()) for col in df.columns
        },
        "sample_top_values": {},
    }

    for col in df.columns[:10]:
        summary["sample_top_values"][col] = safe_top_values(df[col], n=5)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved EDA summary to: {OUTPUT_PATH}")
    print(json.dumps({
        "n_rows": summary["n_rows"],
        "n_columns": summary["n_columns"],
        "columns": summary["columns"][:15]
    }, indent=2))

if __name__ == "__main__":
    main()
