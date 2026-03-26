"""
Clean and normalize the labels column.
Produces reports/label_stats.json and prints the kept label set.
"""
import json
import re
import pandas as pd

from src.utils.paths import RAW_DIR, REPORTS_DIR

INPUT_PATH = RAW_DIR / "issues.parquet"
OUTPUT_PATH = REPORTS_DIR / "label_stats.json"

# Labels assigned during/after triage — would cause leakage
LEAKAGE_LABELS = {
    "triaged", "needs-triage", "needsinvestigation", "needs investigation",
    "has reproducible steps", "needs more info", "needs repro",
    "waiting for author", "waiting-for-author", "wontfix", "duplicate",
    "invalid", "closed",
}

# Regex patterns for structural labels that encode metadata, not issue type
LEAKAGE_PATTERNS = [
    r"^triaged",        # triaged, triaged-framework, etc.
    r"^team-",          # team-framework, team-engine, etc.
    r"^t-",             # T-compiler, T-lang, etc.
    r"^c-",             # c-bug, c-enhancement (rust-style)
    r"^a-",             # a-diagnostics (rust-style)
    r"^p\d$",           # p0, p1, p2, p3, p4 (priority encoded in label)
    r"^priority",       # priority-high, priority-medium
    r"^severity",
    r"^s\d$",           # s1, s2, s3
    r"^topic:",         # topic:editor
    r"^area:",          # area:something
    r"^issue-",         # issue-bug, issue-enhancement (redundant)
    r"^\d+$",           # pure numeric labels
]

MIN_OCCURRENCES = 50  # drop labels rarer than this


def is_leakage(label: str) -> bool:
    if label in LEAKAGE_LABELS:
        return True
    return any(re.match(p, label) for p in LEAKAGE_PATTERNS)


def parse_labels(raw: str) -> list[str]:
    """Split comma-separated label string, normalize each token."""
    tokens = [t.strip().lower() for t in raw.split(",")]
    return [t for t in tokens if t]


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)

    # --- parse all labels ---
    all_parsed = df["labels"].fillna("").apply(parse_labels)

    # raw counts before any filtering
    from collections import Counter
    raw_counter: Counter = Counter()
    for label_list in all_parsed:
        raw_counter.update(label_list)

    total_raw_unique = len(raw_counter)
    total_raw_occurrences = sum(raw_counter.values())

    # --- filter leakage labels ---
    def clean(label_list: list[str]) -> list[str]:
        return list(dict.fromkeys(  # deduplicate, preserve order
            lbl for lbl in label_list if not is_leakage(lbl)
        ))

    cleaned = all_parsed.apply(clean)

    cleaned_counter: Counter = Counter()
    for label_list in cleaned:
        cleaned_counter.update(label_list)

    # --- apply frequency threshold ---
    kept_labels = sorted(
        [lbl for lbl, cnt in cleaned_counter.items() if cnt >= MIN_OCCURRENCES],
        key=lambda x: -cleaned_counter[x],
    )
    dropped_labels = sorted(
        [lbl for lbl, cnt in cleaned_counter.items() if cnt < MIN_OCCURRENCES],
        key=lambda x: -cleaned_counter[x],
    )

    leakage_labels_found = sorted(
        [lbl for lbl in raw_counter if is_leakage(lbl)],
        key=lambda x: -raw_counter[x],
    )

    stats = {
        "threshold": MIN_OCCURRENCES,
        "total_raw_unique_labels": total_raw_unique,
        "total_raw_occurrences": total_raw_occurrences,
        "leakage_labels_removed": len(leakage_labels_found),
        "leakage_labels": {lbl: raw_counter[lbl] for lbl in leakage_labels_found[:30]},
        "total_cleaned_unique_labels": len(cleaned_counter),
        "labels_kept_count": len(kept_labels),
        "labels_dropped_count": len(dropped_labels),
        "kept_labels": {lbl: cleaned_counter[lbl] for lbl in kept_labels},
        "dropped_labels_sample": {lbl: cleaned_counter[lbl] for lbl in dropped_labels[:50]},
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Raw unique labels:     {total_raw_unique:,}")
    print(f"Leakage labels removed:{len(leakage_labels_found):,}")
    print(f"Cleaned unique labels: {len(cleaned_counter):,}")
    print(f"Labels kept (>={MIN_OCCURRENCES}):   {len(kept_labels):,}")
    print(f"Labels dropped:        {len(dropped_labels):,}")
    print(f"\nTop 30 kept labels:")
    for lbl, cnt in list(stats["kept_labels"].items())[:30]:
        print(f"  {cnt:>7,}  {lbl}")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
