"""
Finalize the human-annotated eval set for Day 7 hybrid retrieval evaluation.

Reads the annotated `human_eval_candidates.json` (where a human has set
`human_relevant: true/false` for each candidate) and produces a clean eval
set in the same format as the existing `retrieval_eval.json`.

Input
-----
  data/processed/human_eval_candidates.json   (annotated by human)

Output
------
  data/processed/retrieval_eval_human.json    eval set for Day 7 evaluation
  reports/human_eval_stats.json               annotation stats

Run from project root:
    python -m src.retrieval.finalize_human_eval
"""
import json
import sys

import numpy as np

from src.utils.paths import PROCESSED_DIR, REPORTS_DIR

CANDIDATES_PATH = PROCESSED_DIR / "human_eval_candidates.json"
EVAL_OUT_PATH   = PROCESSED_DIR / "retrieval_eval_human.json"
STATS_PATH      = REPORTS_DIR   / "human_eval_stats.json"


def main() -> None:
    if not CANDIDATES_PATH.exists():
        print(f"ERROR: {CANDIDATES_PATH} not found.")
        print("Run `python -m src.retrieval.build_human_eval` first.")
        sys.exit(1)

    with open(CANDIDATES_PATH) as f:
        annotated = json.load(f)

    # ── validate: check annotation coverage ───────────────────────────────────
    total_candidates    = sum(len(e["candidates"]) for e in annotated)
    unannotated         = sum(
        sum(1 for c in e["candidates"] if c["human_relevant"] is None)
        for e in annotated
    )

    if unannotated > 0:
        pct = unannotated / total_candidates * 100
        print(f"WARNING: {unannotated}/{total_candidates} candidates "
              f"({pct:.0f}%) still have human_relevant=null.")
        print("These will be treated as NOT relevant.")
        print("Annotate the JSON and re-run for a complete eval set.\n")

    # ── build eval records ─────────────────────────────────────────────────────
    eval_set:  list[dict] = []
    skipped:   list[dict] = []

    for entry in annotated:
        gold = [
            c for c in entry["candidates"]
            if c.get("human_relevant") is True
        ]

        record = {
            "query_id":          entry["query_id"],
            "query_repo":        entry["query_repo"],
            "query_split":       entry["query_split"],
            "query_labels":      entry["query_labels"],
            "query_text":        entry["query_text"],
            "n_relevant_corpus": entry["n_label_matched"],
            "gold_relevant": [
                {
                    "corpus_id":    c["corpus_id"],
                    "corpus_repo":  c["corpus_repo"],
                    "shared_labels": c["shared_labels"],
                    # Keep retrieval scores for reference; evaluation ignores them
                    "lex_score":    c.get("lex_score"),
                    "dense_score":  c.get("dense_score"),
                    "text_snippet": f"TITLE: {c['title']}\n{c['body_snippet'][:180]}",
                }
                for c in gold
            ],
        }

        if not gold:
            skipped.append(record)
            print(f"  [skip] {entry['query_repo']} (query_id={entry['query_id']}) "
                  f"— no gold docs annotated")
        else:
            eval_set.append(record)

    # ── save ───────────────────────────────────────────────────────────────────
    EVAL_OUT_PATH.write_text(json.dumps(eval_set, indent=2))
    print(f"\nSaved eval set ({len(eval_set)} queries) → {EVAL_OUT_PATH}")

    # ── stats ──────────────────────────────────────────────────────────────────
    if not eval_set:
        print("No queries with gold docs — nothing to evaluate.")
        sys.exit(0)

    gold_counts   = [len(e["gold_relevant"]) for e in eval_set]
    rel_sizes     = [e["n_relevant_corpus"]  for e in eval_set]
    by_repo: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for e in eval_set:
        by_repo[e["query_repo"]]    = by_repo.get(e["query_repo"], 0) + 1
        by_split[e["query_split"]]  = by_split.get(e["query_split"], 0) + 1

    # Count how gold was sourced (both/lex-only/dense-only/random)
    annotated_map = {e["query_id"]: e["candidates"] for e in annotated}
    source_counts: dict[str, int] = {
        "both": 0, "lex_only": 0, "dense_only": 0, "random": 0
    }
    for e in eval_set:
        cand_map = {c["corpus_id"]: c for c in annotated_map[e["query_id"]]}
        for g in e["gold_relevant"]:
            c = cand_map.get(g["corpus_id"], {})
            has_lex   = c.get("lex_rank") is not None
            has_dense = c.get("dense_rank") is not None
            rand      = c.get("random_sample", False)
            if has_lex and has_dense:
                source_counts["both"] += 1
            elif has_lex:
                source_counts["lex_only"] += 1
            elif has_dense:
                source_counts["dense_only"] += 1
            elif rand:
                source_counts["random"] += 1

    total_gold = sum(gold_counts)

    stats = {
        "n_queries":          len(eval_set),
        "n_skipped":          len(skipped),
        "n_unannotated":      unannotated,
        "n_gold_total":       total_gold,
        "gold_per_query": {
            "mean":   round(float(np.mean(gold_counts)), 2),
            "min":    int(min(gold_counts)),
            "max":    int(max(gold_counts)),
        },
        "relevant_set_size": {
            "min":    int(min(rel_sizes)),
            "median": int(np.median(rel_sizes)),
            "max":    int(max(rel_sizes)),
        },
        "queries_per_repo":  dict(sorted(by_repo.items())),
        "queries_per_split": dict(sorted(by_split.items())),
        "gold_source": source_counts,
        "note": (
            "gold_source counts how each gold doc entered the candidate pool. "
            "'both' = ranked by lexical and dense; 'lex_only'/'dense_only' = "
            "only one method found it; 'random' = neither method, added randomly. "
            "A healthy eval set has gold spread across all sources."
        ),
    }

    STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(f"Saved stats        → {STATS_PATH}")

    # ── print summary ──────────────────────────────────────────────────────────
    print(f"\n── Eval set summary ──")
    print(f"  queries         : {stats['n_queries']}  "
          f"(skipped: {stats['n_skipped']})")
    print(f"  gold total      : {total_gold}  "
          f"(mean {stats['gold_per_query']['mean']:.1f} / query)")
    print(f"  relevant range  : "
          f"{stats['relevant_set_size']['min']} – "
          f"{stats['relevant_set_size']['max']}  "
          f"(median {stats['relevant_set_size']['median']})")
    print(f"\n  By repo  : {stats['queries_per_repo']}")
    print(f"  By split : {stats['queries_per_split']}")

    print(f"\n── Gold source breakdown ──")
    print(f"  (total {total_gold} gold docs)")
    for src, count in source_counts.items():
        pct = count / total_gold * 100 if total_gold else 0
        print(f"  {src:<12} : {count:3d}  ({pct:.0f}%)")

    if source_counts.get("lex_only", 0) > source_counts.get("dense_only", 0) * 3:
        print("\n  [warn] Gold set skews heavily toward lexical candidates.")
        print("         Consider reviewing dense-only candidates more carefully.")
    elif source_counts.get("dense_only", 0) > source_counts.get("lex_only", 0) * 3:
        print("\n  [warn] Gold set skews heavily toward dense candidates.")
        print("         Consider reviewing lexical-only candidates more carefully.")

    print(f"\nEval set ready → {EVAL_OUT_PATH}")
    print("Run Day 7 evaluation:")
    print("  python -m src.retrieval.evaluate_hybrid")


if __name__ == "__main__":
    main()
