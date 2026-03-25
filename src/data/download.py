from datasets import load_dataset
import pandas as pd

from src.utils.paths import RAW_DIR

DATASET_NAME = "sharjeelyunus/github-issues-dataset"
OUTPUT_PATH = RAW_DIR / "issues.parquet"

def main() -> None:
    print(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME, split="train")

    print("Available columns:")
    print(ds.column_names)

    df = ds.to_pandas()

    # Save everything for now; we'll prune tomorrow if needed
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved dataset to: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    main()
