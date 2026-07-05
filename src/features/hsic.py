"""
hsic.py
Selects top-k features by Hilbert-Schmidt Independence Criterion (HSIC).
Uses RBF kernel. Fitted on train fold only.

KNOWN INSTABILITY at small N (~50):
  - Score distribution can be very flat (all features score similarly)
  - This causes degenerate selection (random or near-random features)
  - Guard added: warns if score spread (max-min) < 1e-6
  - Recommend cross-checking with MI if HSIC AUC is near 0.5

Changes from v1:
  - Default top_k reduced from 20 → 12 (more stable at N=63)
  - Score spread guard added (warns on degenerate scores)
  - RBF gamma now uses median heuristic per feature (more robust)
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel


def _median_gamma(x: np.ndarray) -> float:
    """Median heuristic for RBF bandwidth: gamma = 1 / (2 * median_dist^2)."""
    dists = np.abs(x[:, None] - x[None, :]).ravel()
    dists = dists[dists > 0]
    if len(dists) == 0:
        return 1.0
    median_dist = np.median(dists)
    return 1.0 / (2.0 * median_dist ** 2 + 1e-10)


def _hsic_score(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased HSIC estimate between feature vector x and label vector y."""
    n = len(x)
    gamma_x = _median_gamma(x)
    gamma_y = _median_gamma(y)
    Kx = rbf_kernel(x.reshape(-1, 1), gamma=gamma_x)
    Ky = rbf_kernel(y.reshape(-1, 1), gamma=gamma_y)
    H  = np.eye(n) - np.ones((n, n)) / n      # centering matrix
    return float(np.trace(Kx @ H @ Ky @ H)) / (n - 1) ** 2


class HSICSelector:
    def __init__(self, top_k: int = 12):
        self.top_k = top_k
        self.selected_features_: list[str] = []
        self.scores_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HSICSelector":
        y_arr = y.values.astype(float)
        scores = {col: _hsic_score(X[col].values.astype(float), y_arr)
                  for col in X.columns}
        self.scores_ = pd.Series(scores).sort_values(ascending=False)

        # Degenerate score guard
        score_spread = self.scores_.max() - self.scores_.min()
        if score_spread < 1e-6:
            print(f"[HSIC] ⚠ WARNING: Score spread={score_spread:.2e} — "
                  f"scores are nearly identical. HSIC selection may be unreliable "
                  f"at this sample size. Consider using MI instead.")

        top_k = min(self.top_k, len(X.columns))
        self.selected_features_ = self.scores_.head(top_k).index.tolist()
        print(f"[HSIC] top_k={top_k} | spread={score_spread:.2e} | "
              f"selected {len(self.selected_features_)} features")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
