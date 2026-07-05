"""
run_experiment.py
─────────────────────────────────────────────────────────────────────────────
Unified entry point for all experiments.

Usage examples:
  # Single experiment
  python run_experiment.py --config configs/phase1_mi.yaml --model tabicl

  # Full-feature TabICL baseline (no feature selection)
  python run_experiment.py --config configs/phase1_full.yaml --model tabicl

  # Run ALL configs × ALL models
  python run_experiment.py --all

  # Run only TabICL across all configs
  python run_experiment.py --all --model tabicl
─────────────────────────────────────────────────────────────────────────────
"""
import argparse
import yaml
from pathlib import Path

from src.utils.data_loader import load_data
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.evaluation.cv_runner import run_cv

# ── Experiment registry ────────────────────────────────────────────────────
ALL_CONFIGS = [
    # ── Phase 1: fixed selectors ────────────────────────────────────────
    "configs/phase1_full.yaml",           # no selection — full ~1086 features baseline
    "configs/phase1_variance.yaml",       # variance filter only
    "configs/phase1_covariance.yaml",     # covariance filter only
    "configs/phase1_mi.yaml",             # MI fixed top-20
    "configs/phase1_mi_tabicl.yaml",      # MI fixed top-15 (TabICL-tuned)
    "configs/phase1_hsic.yaml",           # HSIC fixed top-12
    # ── Phase 1: dynamic selectors (nested CV k-search) ─────────────────
    # "configs/phase1_dynamic_mi.yaml",     # MI   — k searched over [5..50]
    # "configs/phase1_dynamic_hsic.yaml",   # HSIC — k searched over [5..20]
    # ── Phase 2: fixed sequential ───────────────────────────────────────
    "configs/phase2_var_mi.yaml",         # variance → MI fixed
    "configs/phase2_covar_mi.yaml",       # covariance → MI fixed
    "configs/phase2_covar_hsic.yaml",     # covariance → HSIC fixed
    # ── Phase 2: dynamic sequential ─────────────────────────────────────
    "configs/phase2_var_dynamic_mi.yaml",    # variance → MI dynamic
    "configs/phase2_covar_dynamic_mi.yaml",  # covariance → MI dynamic
]
ALL_MODELS = ["xgboost", "tabicl","tabfm"]

# Configs where XGBoost is skipped (1086 features + N=63 = degenerate overfit)
_SKIP_XGBOOST = {"configs/phase1_full.yaml"}


def load_config(config_path: str) -> dict:
    """Merge base_config.yaml with the experiment-specific override."""
    base_path = Path("configs/base_config.yaml")
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    with open(config_path) as f:
        override = yaml.safe_load(f)
    # Deep merge: override top-level keys (feature_selection merged key-by-key)
    for key, val in override.items():
        if key == "extends":
            continue
        if key in cfg and isinstance(cfg[key], dict) and isinstance(val, dict):
            cfg[key] = {**cfg[key], **val}
        else:
            cfg[key] = val
    return cfg


def run_single(config_path: str, model_name: str) -> dict:
    cfg = load_config(config_path)
    exp = cfg.get("experiment", {})
    output_dir = f"{exp.get('output_dir', 'outputs/unknown')}/{model_name}"

    logger = get_logger(name=f"{exp.get('name','exp')}_{model_name}")
    logger.info(f"Config: {config_path}  |  Model: {model_name}")
    logger.info(f"Selector: {exp.get('selector','?')}  |  Output: {output_dir}")

    set_seed(cfg["data"]["random_state"])

    X, y = load_data(cfg["data"]["path"])
    summary = run_cv(X, y, cfg, model_name=model_name, output_dir=output_dir)

    logger.info(
        f"DONE — AUC={summary['auc']:.3f}  F1={summary['f1']:.3f}  "
        f"Prec={summary['precision']:.3f}  Rec={summary['recall']:.3f}"
    )
    return summary


def main(ALL_CONFIGS=ALL_CONFIGS, ALL_MODELS=ALL_MODELS, _SKIP_XGBOOST=_SKIP_XGBOOST):
    parser = argparse.ArgumentParser(description="GBC Radiomics Experiment Runner")
    parser.add_argument("--config", type=str, help="Path to experiment YAML config")
    parser.add_argument("--model",  type=str, choices=["xgboost", "tabicl","tabfm"],
                        help="Model to use (required with --config; optional filter with --all)")
    parser.add_argument("--all",   action="store_true",
                        help="Run all registered configs × models")
    args = parser.parse_args()

    if args.all :
        models_to_run = [args.model] if args.model else ALL_MODELS
        results = {}
        if args.config :
            ALL_CONFIGS = [args.config]
        for cfg_path in ALL_CONFIGS:
            for model in models_to_run:
                # Skip XGBoost on full-feature config
                if model == "xgboost" and cfg_path in _SKIP_XGBOOST:
                    print(f"\n[SKIP] {cfg_path} × xgboost "
                          f"(full features + N=63 → degenerate)")
                    continue

                key = f"{Path(cfg_path).stem}__{model}"
                print(f"\n{'='*70}")
                print(f"  {key}")
                print(f"{'='*70}")
                summary = run_single(cfg_path, model)
                results[key] = summary

        # ── Final summary table ────────────────────────────────────────
        print("\n\n" + "="*90)
        print(f"  {'EXPERIMENT':<48} {'AUC':>5}  {'F1':>5}  {'PREC':>5}  {'REC':>5}  RESP")
        print("="*90)
        for k, v in results.items():
            print(f"  {k:<48} {v['auc']:>5.3f}  {v['f1']:>5.3f}  "
                  f"{v['precision']:>5.3f}  {v['recall']:>5.3f}  "
                  f"{v['n_responders_detected']}/{v['total_responders']}")
        print("="*90)

    elif args.config and args.model:
        run_single(args.config, args.model)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
