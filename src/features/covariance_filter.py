"""
covariance_filter.py
Removes one feature from every highly correlated pair.
Must be fitted on train fold only — never on full data — to prevent leakage.
"""
import numpy as np
import pandas as pd


class CovarianceFilter:
    """
    Drops columns where |Pearson correlation| > threshold.
    Keeps the first feature of each correlated pair (deterministic).
    """

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.selected_features_: list[str] = []
        self.dropped_features_: list[str] = []

    def fit(self, X: pd.DataFrame) -> "CovarianceFilter":
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper.columns if any(upper[col] > self.threshold)]
        self.dropped_features_ = to_drop
        self.selected_features_ = [c for c in X.columns if c not in to_drop]
        print(f"[CovarianceFilter] threshold={self.threshold} | "
              f"kept={len(self.selected_features_)}, dropped={len(self.dropped_features_)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
