from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from .config import BoardConfig
from .grades import GRADE_TO_V, grade_token, to_grouped_v

SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<CLS>",
    "<MASK>",
]

ANGLE_TOKEN_PATTERN = re.compile(r"^<ANGLE_(-?\d+)>$")
GRADE_TOKEN_PATTERN = re.compile(r"^<GRADE_V(\d+)>$")
BOARD_TOKEN_PATTERN = re.compile(r"^<BOARD_([A-Z0-9_]+)>$")
HOLD_TOKEN_PATTERN = re.compile(r"^<([A-Z0-9_]+)_p(\d+)_(start|middle|finish|foot|unknown)>$")

ROLE_SORT_ORDER = {
    "start": 0,
    "middle": 1,
    "foot": 2,
    "finish": 3,
    "unknown": 9,
}


def parse_frames(frames_str: str | None) -> list[tuple[int, int]]:
    if not isinstance(frames_str, str):
        return []
    matches = re.findall(r"p(\d+)r(\d+)", frames_str)
    return [(int(placement_id), int(role_id)) for placement_id, role_id in matches]


def make_placement_lookup(df_placements: pd.DataFrame) -> dict[tuple[str, int], dict]:
    rows = {}
    for _, row in df_placements.iterrows():
        key = (str(row["board_key"]), int(row["placement_id"]))
        rows[key] = row.to_dict()
    return rows


def role_name(role_id: int, config: BoardConfig) -> str:
    return config.role_id_to_name.get(int(role_id), "unknown")


def placement_xy(
    board_key: str,
    placement_id: int,
    placement_lookup: dict[tuple[str, int], dict],
) -> tuple[float, float]:
    row = placement_lookup.get((str(board_key), int(placement_id)))
    if row is None:
        return (float("nan"), float("nan"))
    return (float(row["x"]), float(row["y"]))


def canonicalize_holds(
    holds: Iterable[tuple[int, int]],
    config: BoardConfig,
    placement_lookup: dict[tuple[str, int], dict],
) -> list[tuple[int, int]]:
    def key(pair: tuple[int, int]):
        placement_id, role_id = pair
        x, y = placement_xy(config.board_key, placement_id, placement_lookup)
        name = role_name(role_id, config)
        return (
            ROLE_SORT_ORDER.get(name, 9),
            y if not np.isnan(y) else 9999.0,
            x if not np.isnan(x) else 9999.0,
            placement_id,
        )

    return sorted(list(holds), key=key)


def board_token(config: BoardConfig) -> str:
    return f"<BOARD_{config.token_prefix}>"


def angle_token(angle: float) -> str:
    return f"<ANGLE_{int(round(float(angle)))}>"


def hold_token(
    placement_id: int,
    role_id: int,
    config: BoardConfig,
) -> str:
    semantic_role = role_name(role_id, config)
    return f"<{config.token_prefix}_p{int(placement_id)}_{semantic_role}>"


def tokenize_route(
    row,
    config: BoardConfig,
    placement_lookup: dict[tuple[str, int], dict],
    include_grade: bool = True,
    canonical: bool = True,
) -> list[str]:
    holds = parse_frames(row["frames"])
    if canonical:
        holds = canonicalize_holds(holds, config, placement_lookup)

    tokens = [
        "<BOS>",
        board_token(config),
        angle_token(row["angle"]),
    ]
    if include_grade:
        tokens.append(grade_token(row["display_difficulty"]))

    tokens.extend(hold_token(placement_id, role_id, config) for placement_id, role_id in holds)
    tokens.append("<EOS>")
    return tokens


def build_route_records(
    df_climbs: pd.DataFrame,
    configs_by_key: dict[str, BoardConfig],
    placement_lookup: dict[tuple[str, int], dict],
) -> pd.DataFrame:
    records: list[dict] = []

    for _, row in df_climbs.iterrows():
        board_key = str(row["board_key"])
        config = configs_by_key[board_key]
        holds = canonicalize_holds(parse_frames(row["frames"]), config, placement_lookup)
        if not holds:
            continue

        hold_tokens = [hold_token(p, r, config) for p, r in holds]
        semantic_roles = [role_name(r, config) for _, r in holds]

        tokens_with_grade = tokenize_route(
            row,
            config=config,
            placement_lookup=placement_lookup,
            include_grade=True,
            canonical=True,
        )
        tokens_no_grade = tokenize_route(
            row,
            config=config,
            placement_lookup=placement_lookup,
            include_grade=False,
            canonical=True,
        )

        records.append(
            {
                "uuid": row["uuid"],
                "board_key": board_key,
                "board_display_name": row["board_display_name"],
                "board_token_prefix": row["board_token_prefix"],
                "board_token": board_token(config),
                "climb_name": row["climb_name"],
                "setter_username": row.get("setter_username"),
                "layout_id": int(row["layout_id"]),
                "layout_name": row.get("layout_name"),
                "board_name": row.get("board_name"),
                "frames": row["frames"],
                "angle": float(row["angle"]),
                "display_difficulty": float(row["display_difficulty"]),
                "grouped_v": int(to_grouped_v(row["display_difficulty"])),
                "boulder_grade": row.get("boulder_grade"),
                "ascensionist_count": row.get("ascensionist_count"),
                "quality_average": row.get("quality_average"),
                "fa_at": row.get("fa_at"),
                "n_holds": len(holds),
                "n_start": semantic_roles.count("start"),
                "n_middle": semantic_roles.count("middle"),
                "n_foot": semantic_roles.count("foot"),
                "n_finish": semantic_roles.count("finish"),
                "holds": holds,
                "hold_tokens": hold_tokens,
                "tokens_with_grade": tokens_with_grade,
                "tokens_no_grade": tokens_no_grade,
                "sequence_with_grade": " ".join(tokens_with_grade),
                "sequence_no_grade": " ".join(tokens_no_grade),
            }
        )

    return pd.DataFrame(records)


def build_vocab(df_routes: pd.DataFrame) -> tuple[list[str], dict[str, int], dict[int, str]]:
    all_tokens: list[str] = []
    for tokens in df_routes["tokens_with_grade"]:
        all_tokens.extend(tokens)

    vocab_tokens = list(SPECIAL_TOKENS)
    for token in sorted(set(all_tokens)):
        if token not in vocab_tokens:
            vocab_tokens.append(token)

    stoi = {token: idx for idx, token in enumerate(vocab_tokens)}
    itos = {idx: token for token, idx in stoi.items()}
    return vocab_tokens, stoi, itos


def encode(tokens: Iterable[str], stoi: dict[str, int]) -> list[int]:
    unk_id = stoi["<UNK>"]
    return [stoi.get(token, unk_id) for token in tokens]


def decode(ids: Iterable[int], itos: dict[int, str]) -> list[str]:
    return [itos.get(int(idx), "<UNK>") for idx in ids]


def build_token_metadata(
    vocab_tokens: list[str],
    stoi: dict[str, int],
    df_placements: pd.DataFrame,
    placement_lookup: dict[tuple[str, int], dict],
    configs_by_prefix: dict[str, BoardConfig],
) -> pd.DataFrame:
    bounds = {}
    for board_key, frame in df_placements.groupby("board_key"):
        xs = frame["x"].astype(float)
        ys = frame["y"].astype(float)
        bounds[str(board_key)] = {
            "x_min": float(xs.min()),
            "x_max": float(xs.max()),
            "y_min": float(ys.min()),
            "y_max": float(ys.max()),
        }

    def normalize(value: float, lo: float, hi: float) -> float:
        if pd.isna(value) or hi == lo:
            return 0.0
        return 2 * ((float(value) - lo) / (hi - lo)) - 1

    rows: list[dict] = []

    for token in vocab_tokens:
        meta = {
            "token": token,
            "token_id": stoi[token],
            "kind": "special",
            "board_key": None,
            "board_token_prefix": None,
            "placement_id": np.nan,
            "role": None,
            "x": np.nan,
            "y": np.nan,
            "x_norm": 0.0,
            "y_norm": 0.0,
            "is_hold": 0,
            "angle": np.nan,
            "grouped_v": np.nan,
        }

        hold_match = HOLD_TOKEN_PATTERN.match(token)
        if hold_match:
            prefix = hold_match.group(1)
            placement_id = int(hold_match.group(2))
            role = hold_match.group(3)
            config = configs_by_prefix[prefix]
            board_key = config.board_key
            row = placement_lookup.get((board_key, placement_id), {})
            x = float(row.get("x", np.nan))
            y = float(row.get("y", np.nan))
            board_bounds = bounds.get(board_key, {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1})

            meta.update(
                {
                    "kind": "hold",
                    "board_key": board_key,
                    "board_token_prefix": prefix,
                    "placement_id": placement_id,
                    "role": role,
                    "x": x,
                    "y": y,
                    "x_norm": normalize(x, board_bounds["x_min"], board_bounds["x_max"]),
                    "y_norm": normalize(y, board_bounds["y_min"], board_bounds["y_max"]),
                    "is_hold": 1,
                }
            )

        angle_match = ANGLE_TOKEN_PATTERN.match(token)
        if angle_match:
            meta.update({"kind": "angle", "angle": int(angle_match.group(1))})

        grade_match = GRADE_TOKEN_PATTERN.match(token)
        if grade_match:
            meta.update({"kind": "grade", "grouped_v": int(grade_match.group(1))})

        board_match = BOARD_TOKEN_PATTERN.match(token)
        if board_match:
            prefix = board_match.group(1)
            config = configs_by_prefix.get(prefix)
            meta.update(
                {
                    "kind": "board",
                    "board_key": None if config is None else config.board_key,
                    "board_token_prefix": prefix,
                }
            )

        rows.append(meta)

    return pd.DataFrame(rows)


def vocab_payload(
    stoi: dict[str, int],
    itos: dict[int, str],
    configs_by_key: dict[str, BoardConfig],
) -> dict:
    return {
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
        "special_tokens": SPECIAL_TOKENS,
        "boards": {
            board_key: {
                "token_prefix": config.token_prefix,
                "board_token": board_token(config),
                "role_definitions": config.role_definitions,
            }
            for board_key, config in configs_by_key.items()
        },
        "grade_to_v": {str(k): v for k, v in GRADE_TO_V.items()},
    }
