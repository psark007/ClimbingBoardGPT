"""Small shared utilities for reproducibility, JSON output, and data splits."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when PyTorch is installed."""
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
    """Convert NumPy/pandas values into JSON-serializable Python objects."""
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
    """Write an object as indented UTF-8 JSON after ``json_safe`` cleanup."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def safe_train_test_split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    stratify_col: str | None = None,
):
    """Split a DataFrame with optional stratification and graceful fallback.

    scikit-learn raises when a requested stratum is too small. The tokenization
    pipeline prefers stratified splits when possible, but falls back to an
    unstratified split rather than failing on tiny smoke-test subsets.
    """
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



def assign_group_splits(
    df: pd.DataFrame,
    group_cols: list[str],
    test_size: float,
    val_size_within_temp: float,
    random_state: int,
    stratify_col: str | None = None,
) -> pd.Series:
    """Assign train/val/test splits at group level.

    This prevents multiple rows for the same logical climb, for example the
    same UUID at several angles, from being distributed across different
    splits. The returned Series is indexed like ``df`` and contains
    ``train``, ``val``, or ``test``.
    """
    group_df = df[group_cols + ([stratify_col] if stratify_col else [])].copy()
    group_df = group_df.drop_duplicates(group_cols).reset_index(drop=True)

    train_groups, temp_groups = safe_train_test_split(
        group_df,
        test_size=test_size,
        random_state=random_state,
        stratify_col=stratify_col,
    )
    val_groups, test_groups = safe_train_test_split(
        temp_groups,
        test_size=val_size_within_temp,
        random_state=random_state,
        stratify_col=stratify_col,
    )

    def key_frame(frame: pd.DataFrame) -> set[tuple]:
        """Return stringified group keys so pandas dtypes cannot affect joins."""
        return set(map(tuple, frame[group_cols].astype(str).values.tolist()))

    train_keys = key_frame(train_groups)
    val_keys = key_frame(val_groups)
    test_keys = key_frame(test_groups)

    def split_for_row(row) -> str:
        """Map one original row back to its group-level split assignment."""
        key = tuple(str(row[col]) for col in group_cols)
        if key in train_keys:
            return "train"
        if key in val_keys:
            return "val"
        if key in test_keys:
            return "test"
        raise KeyError(f"Could not assign split for group key {key}")

    return df.apply(split_for_row, axis=1)
