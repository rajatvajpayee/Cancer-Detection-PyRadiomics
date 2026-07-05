"""
xgboost_wrapper.py
Thin wrapper around XGBClassifier with project defaults.
CPU-only: tree_method=hist, n_jobs=1, no CUDA.

Changes from v1:
  - scale_pos_weight reduced from 4 → 2 (was too aggressive, caused
    high recall / low precision pattern seen in results)
  - min_child_weight=3 added (prevents splits on tiny minority-class
    leaf nodes at N=63)
  - subsample=0.8, colsample_bytree=0.8 added (light regularisation
    for small dataset)
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


class XGBoostWrapper:
    def __init__(self, cfg: dict):
        model_cfg = cfg.get("models", {}).get("xgboost", {})
        self.model = XGBClassifier(
            n_estimators      = model_cfg.get("n_estimators", 300),
            max_depth         = model_cfg.get("max_depth", 4),
            learning_rate     = model_cfg.get("learning_rate", 0.05),
            scale_pos_weight  = model_cfg.get("scale_pos_weight", 2),   # ← was 4
            min_child_weight  = model_cfg.get("min_child_weight", 3),   # ← new
            subsample         = model_cfg.get("subsample", 0.8),        # ← new
            colsample_bytree  = model_cfg.get("colsample_bytree", 0.8), # ← new
            tree_method       = "hist",    # CPU — never CUDA
            n_jobs            = 1,
            random_state      = model_cfg.get("random_state", 42),
            eval_metric       = "logloss",
            verbosity         = 0,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
