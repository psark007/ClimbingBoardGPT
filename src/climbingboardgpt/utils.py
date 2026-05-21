from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def safe_train_test_split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    stratify_col: str | None = None,
):
    stratify = None
    if stratify_col is not None and stratify_col in df.columns:
        counts = df[stratify_col].value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            stratify = df[stratify_col]

    try:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )
