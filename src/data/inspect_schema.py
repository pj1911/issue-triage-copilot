import pandas as pd
import ast

from src.utils.paths import RAW_DIR

INPUT_PATH = RAW_DIR / "issues.parquet"


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)

    print("=" * 60)
    print(f"ROW COUNT:    {len(df):,}")
    print(f"COLUMN COUNT: {len(df.columns)}")
    print(f"COLUMNS:      {list(df.columns)}")

    print("\n--- MISSINGNESS BY COLUMN ---")
    for col in df.columns:
        null_pct = df[col].isna().mean() * 100
        empty_pct = (df[col].astype(str).str.strip() == "").mean() * 100
        print(f"  {col:<12} null={null_pct:.2f}%  empty_str={empty_pct:.2f}%")

    print("\n--- DUPLICATES ---")
    print(f"  Duplicate rows:  {df.duplicated().sum():,}")
    if "id" in df.columns:
        print(f"  Duplicate IDs:   {df['id'].duplicated().sum():,}")

    print("\n--- UNIQUE REPOS ---")
    if "repo" in df.columns:
        print(f"  Count: {df['repo'].nunique():,}")
        print(f"  Top 10: {df['repo'].value_counts().head(10).to_dict()}")

    print("\n--- LABELS (top 20 raw values) ---")
    if "labels" in df.columns:
        sample = df["labels"].dropna().head(5).tolist()
        print(f"  Sample raw values: {sample}")
        print(f"  Dtype: {df['labels'].dtype}")

        # Try to detect format
        first = df["labels"].dropna().iloc[0]
        if isinstance(first, list):
            print("  Format: list")
            all_labels = df["labels"].dropna().explode()
        elif isinstance(first, str) and first.startswith("["):
            print("  Format: stringified list")
            all_labels = (
                df["labels"]
                .dropna()
                .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                .explode()
            )
        else:
            print("  Format: comma-separated string")
            all_labels = df["labels"].dropna().str.split(",").explode().str.strip()

        top20 = all_labels.value_counts().head(20)
        print(f"  Top 20 labels:\n{top20.to_string()}")

    print("\n--- PRIORITY CLASS COUNTS ---")
    if "priority" in df.columns:
        print(df["priority"].value_counts(dropna=False).to_string())

    print("\n--- SEVERITY CLASS COUNTS ---")
    if "severity" in df.columns:
        print(df["severity"].value_counts(dropna=False).to_string())

    print("\n--- TEXT LENGTH STATS ---")
    if "title" in df.columns:
        title_len = df["title"].fillna("").str.len()
        print(f"  Title  — mean={title_len.mean():.1f}  median={title_len.median():.1f}  max={title_len.max()}")
    if "body" in df.columns:
        body_len = df["body"].fillna("").str.len()
        print(f"  Body   — mean={body_len.mean():.1f}  median={body_len.median():.1f}  max={body_len.max()}")

    print("\n--- TIMESTAMP COLUMNS ---")
    ts_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "created", "closed", "at"])]
    if ts_cols:
        for col in ts_cols:
            print(f"  {col}: {df[col].dtype}  sample={df[col].dropna().iloc[0] if df[col].notna().any() else 'N/A'}")
    else:
        print("  None found")

    print("=" * 60)


if __name__ == "__main__":
    main()
