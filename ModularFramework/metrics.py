'''
Splitting out metric computation to streamline main_eval logic.
'''

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class MetricResult:
    auc: float
    f1: float
    sens: float
    spec: float
    brier: float
    n_pos: int
    n_neg: int


def compute_metrics(y_true: np.ndarray, p: np.ndarray, thr: float) -> MetricResult:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    # AUC guard
    if len(np.unique(y_true)) < 2:
        auc = np.nan
    else:
        auc = float(roc_auc_score(y_true, p))

    y_pred = (p >= thr).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = sens if np.isfinite(sens) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    brier = float(np.mean((p - y_true) ** 2))

    return MetricResult(
        auc=auc,
        f1=float(f1),
        sens=float(sens),
        spec=float(spec),
        brier=brier,
        n_pos=n_pos,
        n_neg=n_neg,
    )


def select_threshold_for_target_sensitivity(
    y_true: np.ndarray,
    p: np.ndarray,
    target_sens: float = 0.80,
    fallback: float = 0.5,
) -> float:
    """
    Choose threshold from positive-class scores only:
    thr = quantile(p[y==1], 1 - target_sens).

    With decision rule y_pred = (p >= thr), this targets sensitivity ~= target_sens
    on the calibration set and avoids selecting overly high thresholds.
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)

    if not np.isfinite(target_sens):
        return float(fallback)
    target_sens = float(np.clip(target_sens, 0.0, 1.0))

    pos_mask = (y_true == 1) & np.isfinite(p)
    if not np.any(pos_mask):
        return float(fallback)

    q = 1.0 - target_sens
    p_pos = p[pos_mask]
    thr = float(np.quantile(p_pos, q))
    return thr
