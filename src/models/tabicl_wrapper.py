"""
tabicl_wrapper.py
Wrapper around TabICL with GPU memory management.
Clears VRAM after each fold: del clf + torch.cuda.empty_cache().
"""
import numpy as np
import torch


class TabICLWrapper:
    def __init__(self, cfg: dict):
        model_cfg = cfg.get("models", {}).get("tabicl", {})
        self.device = model_cfg.get("device", "cuda")
        self.random_state = model_cfg.get("random_state", 42)
        self.clf = None

        if self.device == "cuda" and not torch.cuda.is_available():
            print("[TabICL] CUDA not available — falling back to CPU.")
            self.device = "cpu"

    def fit(self, X_train, y_train,random,r2):
        from tabicl import TabICLClassifier  # lazy import
        self.clf = TabICLClassifier(device=self.device,
                                    n_estimators=8,
                                    norm_methods="robust",
                                    class_shuffle_method='latin',
                                    softmax_temperature=0.9,
                                    checkpoint_version="tabicl-classifier-v1-20250208.ckpt",
                                    random_state=self.random_state)
        self.clf.fit(X_train, y_train)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def cleanup(self):
        """Call after each fold to release VRAM."""
        del self.clf
        self.clf = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
