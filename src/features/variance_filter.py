"""
variance_filter.py
Removes near-zero variance features before any selector runs.
Especially important for TabICL — constant/near-constant features
add noise without signal and inflate feature count.

Fitted on train fold only to prevent leakage.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


class VarianceFilter:
    """
    Drops features whose variance falls below `threshold`.
    Default threshold=0.01 removes near-constant features while
    keeping genuinely low-variance but discriminative ones.
    """

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self._selector = VarianceThreshold(threshold=threshold)
        self.selected_features_: list[str] = []
        self.dropped_features_: list[str] = []

    def fit(self, X: pd.DataFrame) -> "VarianceFilter":
        self._selector.fit(X)
        mask = self._selector.get_support()
        self.selected_features_ = X.columns[mask].tolist()
        self.dropped_features_  = X.columns[~mask].tolist()
        print(f"[VarianceFilter] threshold={self.threshold} | "
              f"kept={len(self.selected_features_)}, "
              f"dropped={len(self.dropped_features_)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
