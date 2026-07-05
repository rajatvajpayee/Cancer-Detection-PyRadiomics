"""
mutual_information.py
Selects top-k features by Mutual Information score.
Fitted on train fold only to prevent leakage.
"""
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


class MutualInformationSelector:
    """
    Ranks features by MI with the binary target, keeps top_k.
    """

    def __init__(self, top_k: int = 20, random_state: int = 42):
        self.top_k = top_k
        self.random_state = random_state
        self.selected_features_: list[str] = []
        self.scores_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MutualInformationSelector":
        scores = mutual_info_classif(X, y, random_state=self.random_state)
        self.scores_ = pd.Series(scores, index=X.columns).sort_values(ascending=False)
        self.selected_features_ = self.scores_.head(self.top_k).index.tolist()
        print(f"[MI] top_k={self.top_k} | selected {len(self.selected_features_)} features")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
