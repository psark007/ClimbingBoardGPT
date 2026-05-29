"""Metrics used to evaluate continuous grade predictions."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .grades import to_grouped_v


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute difficulty-scale and grouped-V-grade prediction metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_v = np.asarray([to_grouped_v(x) for x in y_true])
    pred_v = np.asarray([to_grouped_v(x) for x in y_pred])

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "within_1_difficulty": float(np.mean(np.abs(y_true - y_pred) <= 1) * 100),
        "within_2_difficulty": float(np.mean(np.abs(y_true - y_pred) <= 2) * 100),
        "exact_grouped_v": float(np.mean(true_v == pred_v) * 100),
        "within_1_vgrade": float(np.mean(np.abs(true_v - pred_v) <= 1) * 100),
        "within_2_vgrades": float(np.mean(np.abs(true_v - pred_v) <= 2) * 100),
    }


def metrics_by_board(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Compute regression metrics separately for each board in a prediction table."""
    rows = []
    for board_key, frame in pred_df.groupby("board_key"):
        metrics = regression_metrics(frame["y_true"].values, frame["y_pred"].values)
        rows.append({"board_key": board_key, **metrics})
    return pd.DataFrame(rows)


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    """Pretty-print a metric dictionary in the training scripts."""
    print(name)
    print("-" * len(name))
    for key, value in metrics.items():
        suffix = "%" if "within" in key or "exact" in key else ""
        if suffix:
            print(f"{key:24s}: {value:8.2f}{suffix}")
        else:
            print(f"{key:24s}: {value:8.4f}")
