"""
Repo-level train/val/test split.

Repos are assigned wholesale to one partition so that no repo appears
in more than one split.  This tests generalization across codebases,
not memorization of per-repo label conventions.

Split targets (by issue count, not repo count):
  train ~70%  val ~15%  test ~15%
"""
import json
import yaml
import re
import pandas as pd
from pathlib import Path
from collections import Counter

from src.utils.paths import RAW_DIR, PROCESSED_DIR, CONFIGS, REPORTS_DIR

INPUT_PATH  = RAW_DIR / "issues.parquet"
RULES_PATH  = CONFIGS / "label_cleaning.yaml"

TRAIN_PATH  = PROCESSED_DIR / "train.parquet"
VAL_PATH    = PROCESSED_DIR / "val.parquet"
TEST_PATH   = PROCESSED_DIR / "test.parquet"
SPLIT_REPORT = REPORTS_DIR / "split_report.json"

# ── Repo assignments ─────────────────────────────────────────────────────────
# Assigned manually to control label coverage in val/test.
# Test repos = large, diverse, unseen at training time.
# Val repos  = medium-sized for hyperparameter tuning.
# Everything else → train.

TEST_REPOS = {
    "flutter",      # 13,130  — large, mobile/UI domain
    "rust",         # 9,925   — large, systems/compiler domain
    "typescript",   # 5,451   — large, language tooling
    "neovim",       # 1,534   — medium, editor tooling
    "ollama",       # 1,099   — medium, LLM tooling (domain shift)
}

VAL_REPOS = {
    "vscode",       # 7,580   — large, editor
    "deno",         # 1,705   — medium, runtime
    "tauri",        # 892     — medium, desktop apps
    "svelte",       # 428     — small-medium, frontend framework
    "transformers", # 978     — ML library (useful domain for this project)
}


# ── Label cleaning ───────────────────────────────────────────────────────────

def load_rules() -> dict:
    with open(RULES_PATH) as f:
        return yaml.safe_load(f)


def build_leakage_set(rules: dict) -> set[str]:
    return set(rules.get("leakage_exact", []))


def build_leakage_matchers(rules: dict):
    """Return (prefix_list, compiled_regex_list) for leakage detection."""
    prefixes = [p.lower() for p in rules.get("leakage_prefixes", [])]
    patterns = [re.compile(p) for p in rules.get("leakage_patterns_regex", [])]
    return prefixes, patterns


def normalize_label(raw: str, synonyms: dict, strip_prefixes: list[str]) -> str:
    lbl = raw.strip().lower()
    for prefix in strip_prefixes:
        if lbl.startswith(prefix):
            lbl = lbl[len(prefix):]
    return synonyms.get(lbl, lbl)


def is_leakage(tok: str, leakage_exact: set, prefixes: list, patterns: list) -> bool:
    if tok in leakage_exact:
        return True
    if any(tok.startswith(p) for p in prefixes):
        return True
    if any(p.match(tok) for p in patterns):
        return True
    return False


def clean_labels(
    raw: str,
    leakage_exact: set[str],
    prefixes: list[str],
    patterns: list[re.Pattern],
    synonyms: dict,
    strip_prefixes: list[str],
) -> list[str]:
    tokens = [t.strip().lower() for t in raw.split(",") if t.strip()]
    result = []
    seen = set()
    for tok in tokens:
        if is_leakage(tok, leakage_exact, prefixes, patterns):
            continue
        normalized = normalize_label(tok, synonyms, strip_prefixes)
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    rules = load_rules()
    leakage_exact       = build_leakage_set(rules)
    prefixes, patterns  = build_leakage_matchers(rules)
    synonyms            = {k.lower(): v.lower() for k, v in rules.get("synonyms", {}).items()}
    strip_prefixes      = [p.lower() for p in rules.get("strip_prefixes", [])]
    min_occ             = rules["min_occurrences"]
    top_n               = rules["top_n"]

    df = pd.read_parquet(INPUT_PATH)

    # normalize repo name for matching (dataset uses mixed case)
    df["repo_lower"] = df["repo"].str.lower()

    # ── assign split ──
    def assign_split(repo: str) -> str:
        if repo in TEST_REPOS:
            return "test"
        if repo in VAL_REPOS:
            return "val"
        return "train"

    df["split"] = df["repo_lower"].apply(assign_split)

    # ── clean labels on TRAIN only, then apply vocab to all splits ──
    df["labels_clean"] = df["labels"].fillna("").apply(
        lambda x: clean_labels(x, leakage_exact, prefixes, patterns, synonyms, strip_prefixes)
    )

    train_mask = df["split"] == "train"
    counter: Counter = Counter()
    for lbl_list in df.loc[train_mask, "labels_clean"]:
        counter.update(lbl_list)

    # frequency filter + top-N — computed from train only
    frequent = {lbl for lbl, cnt in counter.items() if cnt >= min_occ}
    top_labels: list[str] = [lbl for lbl, _ in counter.most_common() if lbl in frequent][:top_n]
    vocab = set(top_labels)

    # restrict each row's labels to vocab
    df["labels_final"] = df["labels_clean"].apply(
        lambda lst: [l for l in lst if l in vocab]
    )

    # ── select final columns ──
    keep = ["id", "repo", "title", "body", "labels_final", "priority", "split"]
    df_out = df[keep].copy()
    df_out["body"] = df_out["body"].fillna("")
    df_out = df_out.rename(columns={"labels_final": "labels"})

    # ── save splits ──
    for split_name, path in [("train", TRAIN_PATH), ("val", VAL_PATH), ("test", TEST_PATH)]:
        subset = df_out[df_out["split"] == split_name].drop(columns=["split"])
        subset.to_parquet(path, index=False)

    # ── report ──
    split_counts = df_out["split"].value_counts().to_dict()
    total = sum(split_counts.values())

    report = {
        "split_type": "repo-level",
        "total_issues": total,
        "splits": {
            k: {"count": v, "pct": round(v / total * 100, 1)}
            for k, v in split_counts.items()
        },
        "train_repos": sorted(df.loc[train_mask, "repo"].unique().tolist()),
        "val_repos": sorted(VAL_REPOS),
        "test_repos": sorted(TEST_REPOS),
        "label_vocab_size": len(vocab),
        "label_vocab": top_labels,
        "min_occurrences_threshold": min_occ,
        "top_n": top_n,
    }

    with open(SPLIT_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Split complete.")
    for split_name, info in report["splits"].items():
        print(f"  {split_name:<6} {info['count']:>7,} issues  ({info['pct']}%)")
    print(f"\nLabel vocab ({len(vocab)} labels):")
    for i, lbl in enumerate(top_labels, 1):
        print(f"  {i:>3}. {lbl}  ({counter[lbl]:,})")
    print(f"\nSaved: train/val/test parquets + {SPLIT_REPORT}")


if __name__ == "__main__":
    main()
