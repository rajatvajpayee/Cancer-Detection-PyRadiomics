"""
tabfm_wrapper.py
Wrapper around TabFM with GPU memory management.
Clears VRAM after each fold: del clf + torch.cuda.empty_cache().
"""
import numpy as np
import torch


class TabfmWrapper:
    def __init__(self, cfg: dict):
        model_cfg = cfg.get("models", {}).get("tabicl", {})
        self.device = model_cfg.get("device", "cuda")
        self.random_state = model_cfg.get("random_state", 42)
        self.clf = None

        if self.device == "cuda" and not torch.cuda.is_available():
            print("[TabICL] CUDA not available — falling back to CPU.")
            self.device = "cpu"

    def fit(self, X_train, y_train,random,r2):
        import tabfm
        from tabfm import TabFMClassifier
        from tabfm.src.pytorch import model as pytorch_model
        from tabfm.src.classifier_and_regressor import TabFMClassifier
        
        clf = tabfm.tabfm_v1_0_0_pytorch.load(model_type="classification",
                                        device=self.device)
        ## Use below to set the model configuration from scratch
        # clf = pytorch_model.TabFM(
        #             embed_dim=128,
        #             max_classes=2,
        #             col_num_blocks=2,
        #             col_nhead=4,
        #             col_num_inds=32,#
        #             row_num_blocks=2,
        #             row_nhead=4,
        #             row_num_cls=2,
        #             icl_num_blocks=12,#
        #             icl_nhead=4,
        #             ff_factor=2,
        #             feature_group_size=4,
        #             is_classifier=True,
        #         )
        self.clf = TabFMClassifier(
            max_num_features=200,
            model=clf,
        )
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
