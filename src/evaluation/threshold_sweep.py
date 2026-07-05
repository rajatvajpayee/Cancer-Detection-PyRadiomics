"""
threshold_sweep.py
Global OOF threshold sweep — preferred over per-fold tuning.
Sweeps thresholds on the full OOF probability vector to maximise F1.
"""
import numpy as np
from sklearn.metrics import f1_score


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                        n_steps: int = 100) -> tuple[float, float]:
    """
    Returns (best_threshold, best_f1) by sweeping over [0.01, 0.99].
    Uses the full OOF set — not per-fold — to avoid noise at small N.
    """
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_t, best_f1 = 0.5, 0.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print(f"[threshold_sweep] best_threshold={best_t:.3f}, best_f1={best_f1:.3f}")
    return best_t, best_f1
