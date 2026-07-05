"""
cv_runner.py
Stratified K-Fold CV loop with support for dynamic feature selection.

Key design points:
  - Feature selection fitted inside each fold (no leakage)
  - For dynamic selectors: model_fn passed into pipeline so inner CV
    can score each candidate k without touching the outer val fold
  - Imputation and scaling inside each fold
  - OOF probabilities collected for global threshold sweep
  - best_k logged per fold when dynamic selection is used
  - tqdm progress bar on outer fold loop
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

from src.features.feature_pipeline import build_feature_pipeline,build_experiment_tag
from src.models.model_factory import get_model
from src.evaluation.metrics import (
    compute_metrics, save_metrics_csv,
    save_roc_curve, save_confusion_matrix,
)
from src.evaluation.threshold_sweep import find_best_threshold

# def _save_selected_features(fold_features: list[list[str]], all_columns: list[str],
#                              output_dir: str, tag: str):
#     """
#     Saves two files:
#       {tag}_selected_features_per_fold.csv  — which features each fold selected
#       {tag}_feature_frequency.csv           — how often each feature was selected
#                                                across folds (stability indicator)
#     """
#     import pandas as pd

#     # Per-fold list
#     max_len = max(len(f) for f in fold_features)
#     rows = {}
#     for i, feats in enumerate(fold_features):
#         col = feats + [""] * (max_len - len(feats))
#         rows[f"fold_{i+1}"] = col
#     pd.DataFrame(rows).to_csv(f"{output_dir}/{tag}_selected_features_per_fold.csv", index=False)

#     # Frequency across folds — stability check
#     from collections import Counter
#     counter = Counter()
#     for feats in fold_features:
#         counter.update(feats)
#     freq_df = pd.DataFrame(
#         [(feat, count, count/len(fold_features)) for feat, count in counter.most_common()],
#         columns=["feature", "n_folds_selected", "fraction_of_folds"]
#     )
#     freq_df.to_csv(f"{output_dir}/{tag}_feature_frequency.csv", index=False)

#     print(f"  [features] saved → {output_dir}/{tag}_selected_features_per_fold.csv")
#     print(f"  [features] saved → {output_dir}/{tag}_feature_frequency.csv")


def _save_selected_features(fold_features: list[list[str]], all_columns: list[str],
                             output_dir: str, tag: str):
    """
    Saves:
      {output_dir}/{tag}_selected_features_per_fold.csv
      {output_dir}/{tag}_feature_frequency.csv
      outputs/features_nums/{tag}_feature_counts.csv   <- NEW
    """
    import pandas as pd
    from pathlib import Path
    from collections import Counter

    # ── existing: per-fold feature list ────────────────────────────────
    max_len = max(len(f) for f in fold_features)
    rows = {}
    for i, feats in enumerate(fold_features):
        col = feats + [""] * (max_len - len(feats))
        rows[f"fold_{i+1}"] = col
    pd.DataFrame(rows).to_csv(f"{output_dir}/{tag}_selected_features_per_fold.csv", index=False)

    # ── existing: frequency across folds ───────────────────────────────
    counter = Counter()
    for feats in fold_features:
        counter.update(feats)
    freq_df = pd.DataFrame(
        [(feat, count, count/len(fold_features)) for feat, count in counter.most_common()],
        columns=["feature", "n_folds_selected", "fraction_of_folds"]
    )
    freq_df.to_csv(f"{output_dir}/{tag}_feature_frequency.csv", index=False)

    # ── NEW: feature counts per fold ────────────────────────────────────
    root = Path(output_dir)
    while root.name != "outputs" and root.parent != root:
        root = root.parent
    feat_dir = root / "features_nums"
    feat_dir.mkdir(parents=True, exist_ok=True)

    counts_df = pd.DataFrame({
        "fold": [f"fold_{i+1}" for i in range(len(fold_features))],
        "n_features_selected": [len(f) for f in fold_features],
    })
    mean_row = pd.DataFrame([{
        "fold": "mean",
        "n_features_selected": counts_df["n_features_selected"].mean(),
    }])
    counts_df = pd.concat([counts_df, mean_row], ignore_index=True)
    counts_df.to_csv(feat_dir / f"{tag}_feature_counts.csv", index=False)

    print(f"  [features] saved → {output_dir}/{tag}_selected_features_per_fold.csv")
    print(f"  [features] saved → {output_dir}/{tag}_feature_frequency.csv")
    print(f"  [features_nums] saved → {feat_dir}/{tag}_feature_counts.csv")


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    model_name: str,
    output_dir: str,
) -> dict:
    """
    Full stratified CV run for one (feature_selection_config × model) combination.
    Returns OOF summary metrics dict.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_splits     = cfg["cross_validation"]["n_splits"]
    random_state = cfg["cross_validation"]["random_state"]
    scaler_type  = cfg["preprocessing"]["scaler"]
    imputer_strat = cfg["preprocessing"]["imputer"]

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    oof_probs    = np.zeros(len(y))
    oof_true     = np.zeros(len(y))
    fold_metrics = []
    fold_best_ks = []   # track dynamic k per fold
    fold_selected_features = []
    
    fold_bar = tqdm(
        enumerate(skf.split(X, y)),
        total=n_splits,
        desc=f"[{model_name}] CV folds",
        unit="fold",
    )

    for fold_idx, (train_idx, val_idx) in fold_bar:
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # ── 1. Impute (fit on train only) ──────────────────────────────
        if imputer_strat != "none":
            imputer = SimpleImputer(strategy=imputer_strat)
            X_tr  = pd.DataFrame(
                imputer.fit_transform(X_tr), columns=X_tr.columns
            )
            X_val = pd.DataFrame(
                imputer.transform(X_val), columns=X_val.columns
            )

        # ── 2. Feature selection (fit on train only) ───────────────────
        fp = build_feature_pipeline(cfg)

        # For dynamic selectors: build a model_fn that returns a fresh
        # lightweight model for inner CV scoring.
        # We use a fixed lightweight XGBoost for inner CV regardless of
        # the outer model — it's fast and reliable for AUC scoring.
        # This avoids TabICL's GPU cost during the k-search inner loop.
        model_fn = _make_inner_model_fn(cfg)
        X_tr_sel  = fp.fit_transform(
            X_tr, y_tr,
            model_fn=model_fn,
            fold_idx=fold_idx,
        )
        fold_selected_features.append(fp.selected_features)
        X_val_sel = fp.transform(X_val)

        # Log best_k for dynamic steps
        best_ks = fp.best_k_per_step()
        if best_ks:
            fold_best_ks.append(best_ks)

        # ── 3. Scale (fit on train only) ───────────────────────────────
        Scaler = StandardScaler if scaler_type == "standard" else RobustScaler
        scaler = Scaler()
        X_tr_sc  = scaler.fit_transform(X_tr_sel)
        X_val_sc = scaler.transform(X_val_sel)

        # ── 4. Model ───────────────────────────────────────────────────
        model = get_model(model_name, cfg)
        model.fit(X_tr_sc, y_tr.values, X_val_sc, y_val.values)
        probs = model.predict_proba(X_val_sc)

        # VRAM cleanup for TabICL
        if hasattr(model, "cleanup"):
            model.cleanup()

        # ── 5. Store OOF ───────────────────────────────────────────────
        oof_probs[val_idx] = probs
        oof_true[val_idx]  = y_val.values

        fold_m = compute_metrics(y_val.values, probs)
        # Attach best_k info if dynamic
        if best_ks:
            fold_m["best_k"] = str(best_ks)
        fold_metrics.append(fold_m)

        postfix = dict(
            auc=f"{fold_m['auc']:.3f}",
            f1=f"{fold_m['f1']:.3f}",
            prec=f"{fold_m['precision']:.3f}",
            rec=f"{fold_m['recall']:.3f}",
        )
        if best_ks:
            postfix["k"] = str(list(best_ks.values()))
        fold_bar.set_postfix(**postfix)

    # ── 6. Global OOF threshold sweep ─────────────────────────────────
    best_t, _ = find_best_threshold(oof_true, oof_probs)
    summary = compute_metrics(oof_true, oof_probs, threshold=best_t)
    summary["n_responders_detected"] = int(
        sum((oof_probs >= best_t) & (oof_true == 1))
    )
    summary["total_responders"] = int(sum(oof_true == 1))

    # Summarise dynamic k selection across folds
    if fold_best_ks:
        for step_name in fold_best_ks[0]:
            ks = [d[step_name] for d in fold_best_ks if step_name in d]
            summary[f"best_k_{step_name}"] = str(ks)
            print(f"  [{step_name}] k per fold: {ks}  (mean={np.mean(ks):.1f})")

    print(f"\n[{model_name}] OOF Summary →")
    print(f"  AUC       : {summary['auc']:.3f}")
    print(f"  F1        : {summary['f1']:.3f}")
    print(f"  Precision : {summary['precision']:.3f}")
    print(f"  Recall    : {summary['recall']:.3f}")
    print(f"  Threshold : {best_t:.3f}")
    print(f"  Responders: {summary['n_responders_detected']}/{summary['total_responders']}")

    # ── 7. Save outputs ────────────────────────────────────────────────
    # tag = model_name
    tag = build_experiment_tag(cfg, model_name)
    _save_selected_features(fold_selected_features, X.columns.tolist(), output_dir, tag)

    save_metrics_csv(
        fold_metrics, summary,
        f"{output_dir}/{tag}_metrics.csv",
    )
    save_roc_curve(
        oof_true, oof_probs,
        f"{output_dir}/{tag}_roc.png",
        title=f"{model_name} OOF ROC",
    )
    save_confusion_matrix(
        oof_true, (oof_probs >= best_t).astype(int),
        f"{output_dir}/{tag}_cm.png",
        title=f"{model_name} Confusion Matrix (t={best_t:.2f})",
    )

    # Save k-search curves if any dynamic step was used
    if fold_best_ks:
        _save_k_search_summary(fold_metrics, output_dir, tag)

    return summary


def _make_inner_model_fn(cfg: dict):
    """
    Returns a callable () → fresh model for use inside dynamic selector
    inner CV. Always uses a lightweight XGBoost regardless of outer model —
    fast, no GPU needed, reliable AUC scorer for k selection.
    """
    from xgboost import XGBClassifier

    def model_fn():
        return XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            scale_pos_weight=cfg.get("models",{}).get("xgboost",{}).get("scale_pos_weight", 2),
            tree_method="hist",
            n_jobs=1,
            random_state=cfg["cross_validation"]["random_state"],
            eval_metric="logloss",
            verbosity=0,
        )

    return model_fn


def _save_k_search_summary(fold_metrics: list[dict], output_dir: str, tag: str):
    """Save a CSV showing which k was selected in each fold."""
    rows = []
    for i, fm in enumerate(fold_metrics):
        if "best_k" in fm:
            rows.append({"fold": i + 1, "best_k": fm["best_k"],
                         "auc": fm["auc"], "f1": fm["f1"]})
    if rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(
            f"{output_dir}/{tag}_k_search.csv", index=False
        )
        print(f"  [k-search] saved → {output_dir}/{tag}_k_search.csv")
