"""
dynamic_selector.py
──────────────────────────────────────────────────────────────────────────────
Dynamic top_k selection via nested (inner) cross-validation.

Instead of fixing top_k to a hardcoded number, this module searches the
optimal k inside each outer CV fold using a 3-fold inner CV — so the outer
val fold is NEVER touched during k selection (no leakage).

Architecture:
  Outer fold (5-fold)                ← evaluation; outer val never seen here
    Inner fold (3-fold on outer train) ← k selection only
      For each candidate k:
        fit selector + model on inner-train
        score AUC on inner-val
      k* = argmax mean(inner-val AUC) across inner folds
    Refit selector with k* on full outer train
    Evaluate model on outer val → OOF prob stored

Two selectors provided:
  DynamicMISelector   — MI-based, search grid [5,8,10,12,15,20,25,30,40,50]
  DynamicHSICSelector — HSIC-based, search grid [5,8,10,12,15,20]
                        (smaller grid; HSIC is O(n²) per feature)

Both expose:
  .fit(X_train, y_train, model_fn)  →  finds k*, refits on full outer train
  .transform(X)                     →  applies selected features
  .best_k_                          →  optimal k found
  .search_curve_                    →  dict {k: mean_inner_auc}
  .selected_features_               →  list of selected column names
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.features.mutual_information import MutualInformationSelector
from src.features.hsic import HSICSelector


# ── Candidate grids ───────────────────────────────────────────────────────
MI_K_GRID   = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50]
HSIC_K_GRID = [5, 8, 10, 12, 15, 20]


def _inner_auc_for_k(
    X_outer_train: pd.DataFrame,
    y_outer_train: pd.Series,
    selector_cls,
    selector_kwargs: dict,
    k: int,
    model_fn,
    n_inner: int = 3,
    random_state: int = 42,
) -> float:
    """
    Run n_inner-fold CV on X_outer_train to estimate AUC for a given k.
    Returns mean AUC across inner folds.
    If any fold fails (e.g. only one class in inner val), returns 0.0.
    """
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=random_state)
    aucs = []

    for tr_idx, val_idx in skf.split(X_outer_train, y_outer_train):
        X_itr = X_outer_train.iloc[tr_idx]
        X_ival = X_outer_train.iloc[val_idx]
        y_itr = y_outer_train.iloc[tr_idx]
        y_ival = y_outer_train.iloc[val_idx]

        # Skip fold if val has only one class (can't compute AUC)
        if len(y_ival.unique()) < 2:
            continue

        # Fit selector on inner train
        sel = selector_cls(**{**selector_kwargs, "top_k": k})
        try:
            X_itr_sel  = sel.fit_transform(X_itr, y_itr)
            X_ival_sel = sel.transform(X_ival)
        except Exception:
            continue

        # Scale
        scaler = StandardScaler()
        X_itr_sc  = scaler.fit_transform(X_itr_sel)
        X_ival_sc = scaler.transform(X_ival_sel)

        # Fit model and score
        model = model_fn()
        try:
            model.fit(X_itr_sc, y_itr.values)
            probs = model.predict_proba(X_ival_sc)
            # Handle both wrapper objects and raw sklearn estimators
            if hasattr(probs, '__len__') and len(np.array(probs).shape) == 2:
                probs = np.array(probs)[:, 1]
            auc = roc_auc_score(y_ival.values, probs)
            aucs.append(auc)
        except Exception:
            continue

    return float(np.mean(aucs)) if aucs else 0.0


class DynamicMISelector:
    """
    Finds optimal top_k for MI selection via inner 3-fold CV,
    then refits on the full outer-train with that k.
    """

    def __init__(
        self,
        k_grid: list[int] = None,
        n_inner: int = 3,
        random_state: int = 42,
    ):
        self.k_grid       = k_grid or MI_K_GRID
        self.n_inner      = n_inner
        self.random_state = random_state

        # Set after fit()
        self.best_k_: int | None           = None
        self.search_curve_: dict[int,float] = {}
        self.selected_features_: list[str]  = []
        self._fitted_selector: MutualInformationSelector | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_fn,
        desc: str = "MI k-search",
    ) -> "DynamicMISelector":
        """
        Parameters
        ----------
        model_fn : callable → returns a fresh unfitted model with
                   a sklearn-compatible .fit() and .predict_proba() interface.
                   Must be callable with no arguments.
        """
        self.search_curve_ = {}

        for k in tqdm(self.k_grid, desc=f"  [{desc}]", leave=False):
            # k can't exceed available features
            if k > X_train.shape[1]:
                continue
            mean_auc = _inner_auc_for_k(
                X_train, y_train,
                selector_cls=MutualInformationSelector,
                selector_kwargs={"random_state": self.random_state},
                k=k,
                model_fn=model_fn,
                n_inner=self.n_inner,
                random_state=self.random_state,
            )
            self.search_curve_[k] = mean_auc

        if not self.search_curve_:
            # Fallback if all inner folds failed
            self.best_k_ = min(20, X_train.shape[1])
        else:
            self.best_k_ = max(self.search_curve_, key=self.search_curve_.get)

        # Refit on full outer train with best k
        self._fitted_selector = MutualInformationSelector(
            top_k=self.best_k_, random_state=self.random_state
        )
        self._fitted_selector.fit(X_train, y_train)
        self.selected_features_ = self._fitted_selector.selected_features_

        best_auc = self.search_curve_.get(self.best_k_, 0.0)
        print(f"  [DynamicMI] best_k={self.best_k_} "
              f"(inner AUC={best_auc:.3f}) | "
              f"curve: { {k: round(v,3) for k,v in sorted(self.search_curve_.items())} }")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._fitted_selector.transform(X)

    def fit_transform(self, X_train, y_train, model_fn, desc="MI k-search"):
        return self.fit(X_train, y_train, model_fn, desc).transform(X_train)


class DynamicHSICSelector:
    """
    Finds optimal top_k for HSIC selection via inner 3-fold CV,
    then refits on the full outer-train with that k.
    Uses a smaller grid than MI (HSIC is O(n²) per feature).
    """

    def __init__(
        self,
        k_grid: list[int] = None,
        n_inner: int = 3,
        random_state: int = 42,
    ):
        self.k_grid       = k_grid or HSIC_K_GRID
        self.n_inner      = n_inner
        self.random_state = random_state

        self.best_k_: int | None           = None
        self.search_curve_: dict[int,float] = {}
        self.selected_features_: list[str]  = []
        self._fitted_selector: HSICSelector | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_fn,
        desc: str = "HSIC k-search",
    ) -> "DynamicHSICSelector":
        self.search_curve_ = {}

        for k in tqdm(self.k_grid, desc=f"  [{desc}]", leave=False):
            if k > X_train.shape[1]:
                continue
            mean_auc = _inner_auc_for_k(
                X_train, y_train,
                selector_cls=HSICSelector,
                selector_kwargs={},
                k=k,
                model_fn=model_fn,
                n_inner=self.n_inner,
                random_state=self.random_state,
            )
            self.search_curve_[k] = mean_auc

        if not self.search_curve_:
            self.best_k_ = min(12, X_train.shape[1])
        else:
            self.best_k_ = max(self.search_curve_, key=self.search_curve_.get)

        self._fitted_selector = HSICSelector(top_k=self.best_k_)
        self._fitted_selector.fit(X_train, y_train)
        self.selected_features_ = self._fitted_selector.selected_features_

        best_auc = self.search_curve_.get(self.best_k_, 0.0)
        print(f"  [DynamicHSIC] best_k={self.best_k_} "
              f"(inner AUC={best_auc:.3f}) | "
              f"curve: { {k: round(v,3) for k,v in sorted(self.search_curve_.items())} }")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._fitted_selector.transform(X)

    def fit_transform(self, X_train, y_train, model_fn, desc="HSIC k-search"):
        return self.fit(X_train, y_train, model_fn, desc).transform(X_train)
