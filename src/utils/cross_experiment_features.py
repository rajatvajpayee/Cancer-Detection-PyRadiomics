"""
cross_experiment_features.py
──────────────────────────────────────────────────────────────────────────────
Standalone analysis script — run AFTER experiments complete.

Scans all `*_feature_frequency.csv` files across every experiment's output
folder and reports which features are consistently selected.

Two views are produced:

1. STRICT intersection — features present in EVERY experiment's
   feature_frequency.csv (any-fold) and features that were 5/5-stable
   in EVERY experiment. This is often EMPTY in practice — different
   selectors operate on very different candidate pools (e.g. covariance-only
   keeps 200-400 features, HSIC top-12 keeps only 12), so a feature surviving
   ALL experiments is a very high bar.

2. THRESHOLD view (the more useful one in practice) — for each feature,
   the fraction of experiments in which it was selected (any-fold) and the
   fraction in which it was 5/5-stable. Features are ranked by this fraction,
   and a configurable threshold (default 0.5) highlights features selected
   in at least that fraction of experiments.

Outputs under outputs/cross_experiment/:
  cross_experiment_summary.csv
      Per-experiment counts (n features any-fold, n features 5/5-stable).

  cross_experiment_feature_ranking.csv
      Every feature that appears in >=1 experiment, with:
        - n_experiments_any        : # experiments where selected (any fold)
        - frac_experiments_any      : as a fraction of total experiments
        - n_experiments_stable      : # experiments where 5/5-stable
        - frac_experiments_stable   : as a fraction of total experiments
      Sorted by frac_experiments_any descending.

  cross_experiment_intersection_any.csv
      STRICT: features present in every experiment (any-fold). May be empty.

  cross_experiment_intersection_stable.csv
      STRICT: features 5/5-stable in every experiment. May be empty.

  cross_experiment_above_threshold.csv
      Features with frac_experiments_any >= --threshold (default 0.5).

Usage:
  python -m src.utils.cross_experiment_features
  python -m src.utils.cross_experiment_features --outputs-dir outputs --threshold 0.5
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def find_frequency_files(outputs_dir: Path) -> list[Path]:
    """Find all *_feature_frequency.csv files under outputs_dir."""
    return sorted(outputs_dir.rglob("*_feature_frequency.csv"))


def experiment_name(freq_path: Path, outputs_dir: Path) -> str:
    """
    Build a readable experiment identifier from the file path.
    e.g. outputs/phase2/covar_mi/xgboost/xgboost_70covar_top20mi_feature_frequency.csv
      -> phase2/covar_mi/xgboost_70covar_top20mi
    """
    rel = freq_path.relative_to(outputs_dir)
    tag = freq_path.stem.replace("_feature_frequency", "")
    parent_name = rel.parent.parent.name if len(rel.parts) >= 2 else rel.parent.name
    return f"{parent_name}/{tag}"


def main():
    parser = argparse.ArgumentParser(
        description="Find features selected consistently across all experiments."
    )
    parser.add_argument(
        "--outputs-dir", type=str, default="outputs",
        help="Root outputs directory to scan (default: outputs)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Fraction of experiments a feature must appear in to be "
             "listed in cross_experiment_above_threshold.csv (default: 0.5)"
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    freq_files = find_frequency_files(outputs_dir)

    if not freq_files:
        print(f"[cross_experiment_features] No *_feature_frequency.csv files "
              f"found under {outputs_dir}. Run experiments first.")
        return

    print(f"[cross_experiment_features] Found {len(freq_files)} experiment "
          f"feature_frequency files:")
    for f in freq_files:
        print(f"  - {experiment_name(f, outputs_dir)}")

    # ── Load each experiment's selected features ───────────────────────
    any_fold_sets: dict[str, set[str]] = {}     # experiment -> {features selected in >=1 fold}
    stable_sets: dict[str, set[str]] = {}       # experiment -> {features selected in ALL folds}
    summary_rows = []

    for f in freq_files:
        exp = experiment_name(f, outputs_dir)
        df = pd.read_csv(f)

        any_set    = set(df["feature"])
        stable_set = set(df.loc[df["fraction_of_folds"] >= 1.0, "feature"])

        any_fold_sets[exp]   = any_set
        stable_sets[exp]     = stable_set

        summary_rows.append({
            "experiment": exp,
            "n_features_any_fold": len(any_set),
            "n_features_stable_5of5": len(stable_set),
        })

    n_experiments = len(freq_files)

    # ── STRICT intersections across ALL experiments ─────────────────────
    intersection_any    = set.intersection(*any_fold_sets.values()) if any_fold_sets else set()
    intersection_stable = set.intersection(*stable_sets.values()) if stable_sets else set()

    # ── THRESHOLD view: per-feature ranking across experiments ──────────
    from collections import Counter
    any_counter    = Counter()
    stable_counter = Counter()
    for s in any_fold_sets.values():
        any_counter.update(s)
    for s in stable_sets.values():
        stable_counter.update(s)

    all_features = set(any_counter.keys())
    ranking_rows = []
    for feat in all_features:
        n_any    = any_counter.get(feat, 0)
        n_stable = stable_counter.get(feat, 0)
        ranking_rows.append({
            "feature": feat,
            "n_experiments_any": n_any,
            "frac_experiments_any": n_any / n_experiments,
            "n_experiments_stable": n_stable,
            "frac_experiments_stable": n_stable / n_experiments,
        })

    ranking_df = pd.DataFrame(ranking_rows).sort_values(
        ["frac_experiments_any", "frac_experiments_stable"],
        ascending=False
    ).reset_index(drop=True)

    above_threshold_df = ranking_df[
        ranking_df["frac_experiments_any"] >= args.threshold
    ].copy()

    # ── Save results ─────────────────────────────────────────────────────
    out_dir = outputs_dir / "cross_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "cross_experiment_summary.csv", index=False)

    ranking_df.to_csv(out_dir / "cross_experiment_feature_ranking.csv", index=False)

    pd.DataFrame({"feature": sorted(intersection_any), "n_experiments": n_experiments}) \
        .to_csv(out_dir / "cross_experiment_intersection_any.csv", index=False)

    pd.DataFrame({"feature": sorted(intersection_stable), "n_experiments": n_experiments}) \
        .to_csv(out_dir / "cross_experiment_intersection_stable.csv", index=False)

    above_threshold_df.to_csv(out_dir / "cross_experiment_above_threshold.csv", index=False)

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CROSS-EXPERIMENT FEATURE STABILITY  ({n_experiments} experiments)")
    print(f"{'='*70}")
    print(f"\nPer-experiment counts:")
    print(summary_df.to_string(index=False))

    print(f"\n--- STRICT intersection (all {n_experiments} experiments) ---")
    print(f"Features selected in >=1 fold of EVERY experiment: {len(intersection_any)}")
    if intersection_any:
        for feat in sorted(intersection_any):
            print(f"    - {feat}")
    else:
        print("    (none — expected, since selectors use very different "
              "candidate pool sizes)")

    print(f"\nFeatures 5/5-stable in EVERY experiment: {len(intersection_stable)}")
    if intersection_stable:
        for feat in sorted(intersection_stable):
            print(f"    - {feat}")
    else:
        print("    (none)")

    print(f"\n--- THRESHOLD view (>= {args.threshold:.0%} of experiments) ---")
    print(f"Features meeting threshold: {len(above_threshold_df)}")
    if len(above_threshold_df):
        print(above_threshold_df.head(20).to_string(index=False))
        if len(above_threshold_df) > 20:
            print(f"    ... and {len(above_threshold_df) - 20} more "
                  f"(see cross_experiment_above_threshold.csv)")
    else:
        print(f"    (none reached {args.threshold:.0%} — try a lower "
              f"--threshold, e.g. 0.3)")

    print(f"\nSaved to {out_dir}/:")
    print(f"  cross_experiment_summary.csv")
    print(f"  cross_experiment_feature_ranking.csv")
    print(f"  cross_experiment_intersection_any.csv")
    print(f"  cross_experiment_intersection_stable.csv")
    print(f"  cross_experiment_above_threshold.csv")


if __name__ == "__main__":
    main()