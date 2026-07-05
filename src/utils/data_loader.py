"""
data_loader.py
──────────────────────────────────────────────────────────────────────────────
Loads the pyradiomics-extracted CSV and returns clean (X, y).

CSV column layout (confirmed from actual data):
  ┌─────────────────────────┬────────────────────────────────────────────────┐
  │ Column group            │ Action                                         │
  ├─────────────────────────┼────────────────────────────────────────────────┤
  │ case_id                 │ DROP — row identifier, not a feature           │
  │ ct_path, mask_path      │ DROP — file paths, not features                │
  │ elapsed_s               │ DROP — pyradiomics extraction runtime          │
  │ label                   │ TARGET (0 = non-responder, 1 = responder)      │
  │ diagnostics_*  (37 cols)│ DROP — pyradiomics metadata / version info     │
  │ original_*              │ KEEP — shape + 6 texture families (~107 feats) │
  │ log-sigma-*             │ KEEP — LoG filtered (sigmas 1,2,3 × ~89 feats) │
  │ wavelet-*               │ KEEP — 8 wavelet bands × ~89 feats             │
  └─────────────────────────┴────────────────────────────────────────────────┘

Estimated total radiomic features: ~1,086
(original ~107 + LoG ×3 ~267 + wavelet ×8 ~712)
"""
import pandas as pd
import numpy as np
from pathlib import Path


# Columns that are metadata / identifiers — always drop
_META_COLS = {"case_id", "ct_path", "mask_path", "elapsed_s"}

# Valid radiomic feature prefixes — anything else that isn't label or meta is dropped
_FEATURE_PREFIXES = ("original_", "log-sigma-", "wavelet-")


def load_data(
    path: str,
    target_col: str = "label",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the pyradiomics CSV and return (X, y).

    Parameters
    ----------
    path        : path to data.csv
    target_col  : name of the binary label column (default: 'label')
    verbose     : print column audit summary

    Returns
    -------
    X : pd.DataFrame  — radiomic features only, float64
    y : pd.Series     — binary labels (0 / 1)
    """
    df = pd.read_csv(path, low_memory=False)

    # ── Validate target ────────────────────────────────────────────────
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns (first 10): {list(df.columns[:10])}"
        )
    y = df[target_col].astype(int)
    assert set(y.unique()).issubset({0, 1}), \
        f"Target '{target_col}' must be binary (0/1). Found: {sorted(y.unique())}"

    # ── Identify and drop non-feature columns ─────────────────────────
    diag_cols   = [c for c in df.columns if c.startswith("diagnostics_")]
    meta_cols   = [c for c in df.columns if c in _META_COLS]
    drop_cols   = set(diag_cols + meta_cols + [target_col])

    # Keep only confirmed radiomic feature columns
    feature_cols = [
        c for c in df.columns
        if c not in drop_cols and c.startswith(_FEATURE_PREFIXES)
    ]

    # Catch-all: warn about any column not accounted for
    unaccounted = [
        c for c in df.columns
        if c not in drop_cols and c not in feature_cols
    ]
    if unaccounted and verbose:
        print(f"[data_loader] ⚠ Unaccounted columns (not feature/meta/diag) — "
              f"dropping: {unaccounted}")

    X = df[feature_cols].copy()

    # ── Type enforcement ───────────────────────────────────────────────
    # Some columns may have been read as object if they contain strings like
    # "(np.float64(x), ...)" from pyradiomics CenterOfMass — these should
    # not be in the feature set, but guard anyway.
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        if verbose:
            print(f"[data_loader] ⚠ Dropping {len(non_numeric)} non-numeric "
                  f"feature columns: {non_numeric[:5]}{'...' if len(non_numeric)>5 else ''}")
        X = X.drop(columns=non_numeric)

    X = X.astype(np.float64)

    # ── Missing value audit ────────────────────────────────────────────
    n_missing_cols = (X.isna().sum() > 0).sum()
    total_missing  = X.isna().sum().sum()

    # ── Summary ───────────────────────────────────────────────────────
    if verbose:
        n_resp     = int(y.sum())
        n_nonresp  = int((y == 0).sum())
        n_orig     = sum(1 for c in X.columns if c.startswith("original_"))
        n_log      = sum(1 for c in X.columns if c.startswith("log-sigma-"))
        n_wav      = sum(1 for c in X.columns if c.startswith("wavelet-"))

        print(f"\n[data_loader] ── Data loaded ──────────────────────────")
        print(f"  Path              : {path}")
        print(f"  Rows              : {len(df)}")
        print(f"  Target            : {n_resp} responders (1) / {n_nonresp} non-responders (0)")
        print(f"  Dropped (meta)    : {len(meta_cols)} cols  {sorted(meta_cols)}")
        print(f"  Dropped (diag)    : {len(diag_cols)} cols  [diagnostics_*]")
        print(f"  ── Features kept  : {X.shape[1]} total ──")
        print(f"     original_*     : {n_orig}")
        print(f"     log-sigma-*    : {n_log}")
        print(f"     wavelet-*      : {n_wav}")
        if total_missing:
            print(f"  ⚠ Missing values  : {total_missing} total across {n_missing_cols} columns")
        else:
            print(f"  Missing values    : 0 ✓")
        print(f"────────────────────────────────────────────────────────\n")

    return X, y


def get_feature_groups(X: pd.DataFrame) -> dict[str, list[str]]:
    """
    Returns a dict grouping feature column names by image type.
    Useful for analysis and reporting.

    Example keys:
        'original_shape', 'original_firstorder', 'original_glcm', ...
        'log-sigma-1-0-mm-3D_firstorder', ...,
        'wavelet-LLH_glcm', ..., 'wavelet-HHH_ngtdm'
    """
    groups: dict[str, list[str]] = {}
    for col in X.columns:
        # Split on last texture family: e.g. "wavelet-LLH_glcm_Autocorrelation"
        # Group key = everything before the last underscore-separated metric name
        parts = col.split("_")
        if col.startswith("original_shape"):
            key = "original_shape"
        elif col.startswith("original_"):
            key = f"original_{parts[1]}"          # e.g. original_glcm
        elif col.startswith("log-sigma-"):
            key = f"{parts[0]}_{parts[1]}"         # e.g. log-sigma-1-0-mm-3D_firstorder
        elif col.startswith("wavelet-"):
            key = f"{parts[0]}_{parts[1]}"         # e.g. wavelet-LLH_glcm
        else:
            key = "other"
        groups.setdefault(key, []).append(col)
    return groups
