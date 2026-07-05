"""
feature_pipeline.py
Orchestrates sequential feature selection based on config's active selectors list.

Supported selectors (applied in listed order):
  none          →  passthrough, all features forwarded
  variance      →  drop near-zero variance features
  covariance    →  drop highly correlated pairs
  mi            →  fixed top-k by Mutual Information
  hsic          →  fixed top-k by HSIC
  dynamic_mi    →  optimal top-k found via inner 3-fold CV search (MI)
  dynamic_hsic  →  optimal top-k found via inner 3-fold CV search (HSIC)

Phase 1 examples:
  active: [none]                  →  full features, no selection
  active: [variance]              →  variance filter only
  active: [covariance]            →  covariance filter only
  active: [mi]                    →  MI fixed top-k
  active: [dynamic_mi]            →  MI with nested CV k-search
  active: [dynamic_hsic]          →  HSIC with nested CV k-search

Phase 2 examples:
  active: [covariance, mi]        →  covariance → MI fixed
  active: [covariance, dynamic_mi]→  covariance → MI dynamic
  active: [variance, dynamic_mi]  →  variance → MI dynamic

All selectors are fitted on X_train only; transforms applied to both
train and val to prevent leakage.

For dynamic selectors, a model_fn must be passed to fit_transform()
so the inner CV can score each candidate k.
"""
import pandas as pd
from src.features.variance_filter import VarianceFilter
from src.features.covariance_filter import CovarianceFilter
from src.features.mutual_information import MutualInformationSelector
from src.features.hsic import HSICSelector
from src.features.dynamic_selector import DynamicMISelector, DynamicHSICSelector

# Selectors that do not use the label y
_UNSUPERVISED = {"variance", "covariance", "none"}

# Selectors that need a model_fn for inner CV
_DYNAMIC = {"dynamic_mi", "dynamic_hsic"}

def build_experiment_tag(cfg: dict, model_name: str) -> str:
    """
    Builds a descriptive filename tag from active selectors + their params.
    Examples:
      xgboost_70covar
      tabicl_70covar_top15mi
      xgboost_var_top20mi
      tabicl_top12hsic
      xgboost_70covar_dynamicMI
      tabicl_full
    """
    active = cfg["feature_selection"]["active"]
    fs_cfg = cfg["feature_selection"]
    parts = []

    for name in active:
        if name == "none":
            parts.append("full")
        elif name == "variance":
            thr = fs_cfg.get("variance", {}).get("threshold", 0.01)
            parts.append(f"var{thr}")
        elif name == "covariance":
            thr = fs_cfg.get("covariance", {}).get("threshold", 0.85)
            parts.append(f"{int(thr*100)}covar")
        elif name == "mi":
            k = fs_cfg.get("mi", {}).get("top_k", 20)
            parts.append(f"top{k}mi")
        elif name == "hsic":
            k = fs_cfg.get("hsic", {}).get("top_k", 12)
            parts.append(f"top{k}hsic")
        elif name == "dynamic_mi":
            parts.append("dynamicMI")
        elif name == "dynamic_hsic":
            parts.append("dynamicHSIC")

    tag = "_".join(parts) if parts else "nosel"
    return f"{model_name}_{tag}"

def build_feature_pipeline(cfg: dict) -> "FeaturePipeline":
    active = cfg["feature_selection"]["active"]
    fs_cfg = cfg["feature_selection"]
    steps  = []

    for name in active:
        if name == "none":
            steps.append(("none", None))

        elif name == "variance":
            threshold = fs_cfg.get("variance", {}).get("threshold", 0.01)
            steps.append(("variance", VarianceFilter(threshold=threshold)))

        elif name == "covariance":
            threshold = fs_cfg.get("covariance", {}).get("threshold", 0.85)
            steps.append(("covariance", CovarianceFilter(threshold=threshold)))

        elif name == "mi":
            top_k = fs_cfg.get("mi", {}).get("top_k", 20)
            steps.append(("mi", MutualInformationSelector(top_k=top_k)))

        elif name == "hsic":
            top_k = fs_cfg.get("hsic", {}).get("top_k", 12)
            steps.append(("hsic", HSICSelector(top_k=top_k)))

        elif name == "dynamic_mi":
            k_grid   = fs_cfg.get("dynamic_mi", {}).get("k_grid", None)
            n_inner  = fs_cfg.get("dynamic_mi", {}).get("n_inner", 3)
            rs       = fs_cfg.get("dynamic_mi", {}).get("random_state", 42)
            steps.append(("dynamic_mi", DynamicMISelector(
                k_grid=k_grid, n_inner=n_inner, random_state=rs
            )))

        elif name == "dynamic_hsic":
            k_grid   = fs_cfg.get("dynamic_hsic", {}).get("k_grid", None)
            n_inner  = fs_cfg.get("dynamic_hsic", {}).get("n_inner", 3)
            rs       = fs_cfg.get("dynamic_hsic", {}).get("random_state", 42)
            steps.append(("dynamic_hsic", DynamicHSICSelector(
                k_grid=k_grid, n_inner=n_inner, random_state=rs
            )))

        else:
            raise ValueError(
                f"Unknown selector: '{name}'. "
                f"Choose from: none, variance, covariance, mi, hsic, "
                f"dynamic_mi, dynamic_hsic"
            )

    return FeaturePipeline(steps)


class FeaturePipeline:
    def __init__(self, steps: list):
        self.steps = steps

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_fn=None,       # required when any step is dynamic_mi / dynamic_hsic
        fold_idx: int = 0,   # for logging only
    ) -> pd.DataFrame:
        X = X_train.copy()
        for name, selector in self.steps:
            if name == "none" or selector is None:
                pass
            elif name in _UNSUPERVISED:
                X = selector.fit_transform(X)
            elif name in _DYNAMIC:
                if model_fn is None:
                    raise ValueError(
                        f"Selector '{name}' requires model_fn to be passed "
                        f"to fit_transform() for inner CV k-search."
                    )
                desc = f"fold {fold_idx+1} {name}"
                X = selector.fit_transform(X, y_train, model_fn, desc=desc)
            else:
                X = selector.fit_transform(X, y_train)
        return X

    def transform(self, X_val: pd.DataFrame) -> pd.DataFrame:
        X = X_val.copy()
        for name, selector in self.steps:
            if name == "none" or selector is None:
                pass
            else:
                X = selector.transform(X)
        return X

    def has_dynamic_step(self) -> bool:
        return any(name in _DYNAMIC for name, _ in self.steps)

    @property
    def selected_features(self) -> list[str]:
        for name, selector in reversed(self.steps):
            if selector is not None and hasattr(selector, "selected_features_"):
                return selector.selected_features_
        return []

    def best_k_per_step(self) -> dict[str, int]:
        """Returns {step_name: best_k} for any dynamic steps."""
        result = {}
        for name, selector in self.steps:
            if name in _DYNAMIC and hasattr(selector, "best_k_"):
                result[name] = selector.best_k_
        return result
