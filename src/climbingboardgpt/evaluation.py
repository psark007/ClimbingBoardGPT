"""Evaluation utilities for generated climbing-board routes.

The helpers in this module are intentionally model-agnostic: they work from
tokens, frames strings, and token metadata so notebooks, scripts, and tests can
reuse the same route validity, novelty, and geometry calculations.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from .tokenization import parse_tokens, tokens_to_hold_records


def parse_token_list(value) -> list[str]:
    """Compatibility wrapper around the shared token parser."""
    return parse_tokens(value)


def validity_from_records(records: list[dict[str, object]], requested_board_prefix: str | None = None) -> dict[str, object]:
    """Compute evaluation-specific route-validity flags from hold records."""
    placements = [int(record["placement_id"]) for record in records]
    roles = [str(record["role"]) for record in records]
    prefixes = [str(record["board_token_prefix"]) for record in records]
    one_board_only = len(set(prefixes)) <= 1
    matches_requested_board = requested_board_prefix is None or all(prefix == requested_board_prefix for prefix in prefixes)

    out = {
        "n_holds_eval": len(records),
        "n_unique_placements_eval": len(set(placements)),
        "has_duplicate_placements_eval": len(records) != len(set(placements)),
        "one_board_only_eval": one_board_only,
        "matches_requested_board_eval": matches_requested_board,
        "n_start_eval": roles.count("start"),
        "n_middle_eval": roles.count("middle"),
        "n_foot_eval": roles.count("foot"),
        "n_finish_eval": roles.count("finish"),
        "has_start_eval": "start" in roles,
        "has_middle_eval": "middle" in roles,
        "has_finish_eval": "finish" in roles,
    }
    out["basic_valid_eval"] = (
        one_board_only
        and out["n_holds_eval"] >= 3
        and out["n_holds_eval"] == out["n_unique_placements_eval"]
        and out["has_start_eval"]
        and out["has_finish_eval"]
    )
    out["strict_valid_eval"] = (
        out["basic_valid_eval"]
        and out["has_middle_eval"]
        and out["n_holds_eval"] >= 4
    )
    return out


def frames_to_holds(frames: str | None) -> list[tuple[int, int]]:
    """Parse a frames string into ``(placement_id, role_id)`` pairs."""
    if not isinstance(frames, str):
        return []
    return [(int(p), int(r)) for p, r in re.findall(r"p(\d+)r(\d+)", frames)]


def holds_to_placement_set(holds: Iterable[tuple[int, int]]) -> frozenset[int]:
    """Drop role IDs and represent a route by its unique placement IDs."""
    return frozenset(int(placement_id) for placement_id, _ in holds)


def jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    """Return Jaccard similarity between two placement sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def nearest_real_route_same_board(
    generated_set: frozenset[int],
    generated_board_key: str,
    real_df: pd.DataFrame,
) -> dict[str, object]:
    """Find the most similar real route on the same board by Jaccard score."""
    board_frame = real_df[real_df["board_key"] == generated_board_key]
    if board_frame.empty:
        return {
            "nearest_real_jaccard": np.nan,
            "nearest_real_uuid": None,
            "nearest_real_name": None,
            "nearest_real_grouped_v": None,
            "nearest_real_angle": None,
            "novelty_distance": np.nan,
        }

    similarities = board_frame["hold_set"].map(lambda hold_set: jaccard(generated_set, hold_set))
    best_idx = similarities.idxmax()
    row = board_frame.loc[best_idx]

    nearest_real_jaccard = float(similarities.loc[best_idx])
    return {
        "nearest_real_jaccard": nearest_real_jaccard,
        "nearest_real_uuid": row["uuid"],
        "nearest_real_name": row["climb_name"],
        "nearest_real_grouped_v": row["grouped_v"],
        "nearest_real_angle": row["angle"],
        "novelty_distance": 1.0 - nearest_real_jaccard,
    }


def build_placement_coords(df_token_meta: pd.DataFrame) -> dict[tuple[str, int], dict[str, float]]:
    """Build a placement-coordinate lookup from token metadata."""
    hold_meta = df_token_meta[df_token_meta["kind"] == "hold"].dropna(subset=["placement_id"]).copy()
    coords = {}
    for _, row in hold_meta.drop_duplicates(["board_key", "placement_id"]).iterrows():
        key = (str(row["board_key"]), int(row["placement_id"]))
        coords[key] = {
            "x": float(row["x"]),
            "y": float(row["y"]),
        }
    return coords


def simple_route_features(
    board_key: str,
    records: list[dict[str, object]],
    placement_coords: dict[tuple[str, int], dict[str, float]],
) -> dict[str, float]:
    """Compute simple geometric route features from hold coordinates.

    These features are descriptive rather than a full climbing-physics model:
    height/width describe route spread, and hand-reach distances summarize the
    pairwise spacing among start/middle/finish holds.
    """
    rows = []
    for record in records:
        key = (str(board_key), int(record["placement_id"]))
        coord = placement_coords.get(key)
        if coord is None:
            continue
        x = float(coord["x"])
        y = float(coord["y"])
        if np.isnan(x) or np.isnan(y):
            continue
        role = str(record["role"])
        rows.append(
            {
                "x": x,
                "y": y,
                "role": role,
                "is_hand": role in {"start", "middle", "finish"},
                "is_foot": role == "foot",
            }
        )

    if not rows:
        return {
            "geom_n_holds": 0.0,
            "geom_height": np.nan,
            "geom_width": np.nan,
            "geom_mean_y": np.nan,
            "geom_mean_x_abs": np.nan,
            "geom_mean_hand_reach": np.nan,
            "geom_max_hand_reach": np.nan,
        }

    d = pd.DataFrame(rows)
    out = {
        "geom_n_holds": float(len(d)),
        "geom_height": float(d["y"].max() - d["y"].min()),
        "geom_width": float(d["x"].max() - d["x"].min()),
        "geom_mean_y": float(d["y"].mean()),
        "geom_mean_x_abs": float(d["x"].abs().mean()),
    }

    hands = d[d["is_hand"]].sort_values(["y", "x"])
    if len(hands) >= 2:
        distances = pdist(hands[["x", "y"]].values)
        out["geom_mean_hand_reach"] = float(distances.mean())
        out["geom_max_hand_reach"] = float(distances.max())
    else:
        out["geom_mean_hand_reach"] = np.nan
        out["geom_max_hand_reach"] = np.nan

    return out
