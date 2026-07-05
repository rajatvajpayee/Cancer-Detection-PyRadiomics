"""
model_factory.py
Returns the correct wrapper instance based on model name string.
Supported: 'xgboost', 'tabicl'
"""
from src.models.xgboost_wrapper import XGBoostWrapper
from src.models.tabicl_wrapper import TabICLWrapper
from src.models.tabfm_wrapper import TabfmWrapper


def get_model(model_name: str, cfg: dict):
    name = model_name.lower()
    if name == "xgboost":
        return XGBoostWrapper(cfg)
    elif name == "tabicl":
        return TabICLWrapper(cfg)
    elif name == "tabfm":
        return TabfmWrapper(cfg)
    else:
        raise ValueError(f"Unknown model: '{model_name}'. "
                         f"Choose from: xgboost, tabicl")
