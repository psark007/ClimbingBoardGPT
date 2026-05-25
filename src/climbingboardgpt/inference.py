"""Inference helpers for ClimbingBoardGPT demos.

This module is intentionally small and webapp-friendly: it loads trained
checkpoints once, keeps them in memory, and exposes route generation helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoints import load_checkpoint
from .config import BoardConfig, load_board_config
from .generation import generate_one
from .grades import to_grouped_v
from .models import JointRouteGPT, JointRouteTransformerRegressor
from .tokenization import (
    angle_token,
    board_token,
    canonicalize_holds,
    hold_token,
    parse_frames,
)


@dataclass
class LoadedGenerator:
    """A loaded GPT-style route generator plus vocabulary metadata."""

    model: JointRouteGPT
    stoi: dict[str, int]
    itos: dict[int, str]
    device: torch.device
    checkpoint_path: Path


@dataclass
class LoadedGradePredictor:
    """A loaded transformer grade predictor plus vocabulary metadata."""

    model: JointRouteTransformerRegressor
    stoi: dict[str, int]
    itos: dict[int, str]
    device: torch.device
    checkpoint_path: Path
    max_len: int
    pad_id: int
    unk_id: int


def load_grade_predictor(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    torch_threads: int | None = None,
) -> LoadedGradePredictor:
    """Load a trained joint grade-prediction checkpoint.

    Args:
        checkpoint_path:
            Path to ``models/joint_transformer_grade_predictor.pth``.
        device:
            ``"cpu"``, ``"cuda"``, or None for auto-detection.
        torch_threads:
            Optional CPU thread cap for small VPS demos.

    Returns:
        LoadedGradePredictor containing the PyTorch model and tokenizer maps.
    """
    if torch_threads is not None:
        torch.set_num_threads(int(torch_threads))

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find grade predictor checkpoint: {checkpoint_path}")

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint = load_checkpoint(checkpoint_path, map_location=resolved_device, trusted=True)

    cfg = checkpoint["config"]
    stoi = {str(k): int(v) for k, v in checkpoint["stoi"].items()}
    itos = {int(k): str(v) for k, v in checkpoint["itos"].items()}
    coord_features = checkpoint["coord_features"]
    if not isinstance(coord_features, torch.Tensor):
        coord_features = torch.tensor(coord_features, dtype=torch.float32)

    model = JointRouteTransformerRegressor(
        vocab_size=cfg["vocab_size"],
        max_len=cfg["max_len"],
        coord_features=coord_features,
        d_model=cfg.get("d_model", 128),
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 4),
        dim_feedforward=cfg.get("dim_feedforward", 256),
        dropout=cfg.get("dropout", 0.10),
        pad_id=cfg.get("pad_id", stoi["<PAD>"]),
    ).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return LoadedGradePredictor(
        model=model,
        stoi=stoi,
        itos=itos,
        device=resolved_device,
        checkpoint_path=checkpoint_path,
        max_len=int(cfg["max_len"]),
        pad_id=int(cfg.get("pad_id", stoi["<PAD>"])),
        unk_id=int(stoi["<UNK>"]),
    )


def predict_route_grade(
    grade_predictor: LoadedGradePredictor,
    tokens: list[str],
) -> dict[str, object]:
    """Predict the grade of a route-token sequence.

    The grade token is removed before scoring, because the predictor should
    infer the grade from the board, angle, and hold-role tokens rather than
    reading the requested grade.
    """
    model_tokens = [token for token in tokens if not str(token).startswith("<GRADE_")]
    if model_tokens and model_tokens[0] == "<BOS>":
        model_tokens = ["<CLS>"] + model_tokens[1:]
    else:
        model_tokens = ["<CLS>"] + model_tokens

    ids = [grade_predictor.stoi.get(token, grade_predictor.unk_id) for token in model_tokens]
    ids = ids[: grade_predictor.max_len]
    mask = [1] * len(ids)

    if len(ids) < grade_predictor.max_len:
        pad_n = grade_predictor.max_len - len(ids)
        ids += [grade_predictor.pad_id] * pad_n
        mask += [0] * pad_n

    with torch.no_grad():
        input_ids = torch.tensor([ids], dtype=torch.long, device=grade_predictor.device)
        attention_mask = torch.tensor([mask], dtype=torch.bool, device=grade_predictor.device)
        pred_display_difficulty = float(grade_predictor.model(input_ids, attention_mask).cpu().item())

    return {
        "predicted_display_difficulty": pred_display_difficulty,
        "predicted_grouped_v": int(to_grouped_v(pred_display_difficulty)),
    }


def load_route_generator(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    torch_threads: int | None = None,
) -> LoadedGenerator:
    """Load a trained joint route generator checkpoint.

    Args:
        checkpoint_path:
            Path to ``models/joint_route_gpt_generator.pth``.
        device:
            ``"cpu"``, ``"cuda"``, or None for auto-detection.
        torch_threads:
            Optional CPU thread cap for small VPS demos.

    Returns:
        LoadedGenerator containing the PyTorch model and tokenizer maps.
    """
    if torch_threads is not None:
        torch.set_num_threads(int(torch_threads))

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find generator checkpoint: {checkpoint_path}")

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint = load_checkpoint(checkpoint_path, map_location=resolved_device, trusted=True)

    cfg = checkpoint["config"]
    stoi = {str(k): int(v) for k, v in checkpoint["stoi"].items()}
    itos = {int(k): str(v) for k, v in checkpoint["itos"].items()}

    model = JointRouteGPT(
        vocab_size=cfg["vocab_size"],
        block_size=cfg["block_size"],
        n_embd=cfg.get("n_embd", 128),
        n_head=cfg.get("n_head", 4),
        n_layer=cfg.get("n_layer", 4),
        dropout=cfg.get("dropout", 0.10),
        pad_id=cfg.get("pad_id", stoi["<PAD>"]),
    ).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return LoadedGenerator(
        model=model,
        stoi=stoi,
        itos=itos,
        device=resolved_device,
        checkpoint_path=checkpoint_path,
    )


def generate_route(
    generator: LoadedGenerator,
    board_config: BoardConfig,
    angle: int,
    grade: int,
    temperature: float = 0.9,
    top_k: int | None = 50,
    max_new_tokens: int = 40,
) -> dict[str, object]:
    """Generate a single route for a board/angle/grade condition."""
    return {
        "board_key": board_config.board_key,
        "board_display_name": board_config.display_name,
        **generate_one(
            model=generator.model,
            stoi=generator.stoi,
            itos=generator.itos,
            device=generator.device,
            board_prefix=board_config.token_prefix,
            angle=int(angle),
            grouped_v=int(grade),
            role_name_to_id=board_config.role_definitions,
            temperature=float(temperature),
            top_k=top_k,
            max_new_tokens=int(max_new_tokens),
        ),
    }


def load_board_for_demo(board: str, config_dir: str | Path | None = None) -> BoardConfig:
    """Load a board config by key, with a clearer demo error message."""
    try:
        return load_board_config(board, config_dir=config_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Unknown board '{board}'. Expected one of the JSON configs in configs/."
        ) from exc


def build_placement_lookup_from_token_metadata(df_token_meta) -> dict[tuple[str, int], dict]:
    """Build the placement lookup expected by tokenization helpers.

    The training-time tokenization code canonicalizes holds using a lookup keyed
    by ``(board_key, placement_id)``. At inference/demo time, we usually have
    ``token_metadata.csv`` rather than the raw database, so this reconstructs
    the necessary coordinate lookup from token metadata.
    """
    hold_meta = df_token_meta[df_token_meta["kind"] == "hold"].dropna(subset=["placement_id"]).copy()
    lookup: dict[tuple[str, int], dict] = {}

    for _, row in hold_meta.drop_duplicates(["board_key", "placement_id"]).iterrows():
        key = (str(row["board_key"]), int(row["placement_id"]))
        lookup[key] = {
            "board_key": str(row["board_key"]),
            "board_token_prefix": str(row["board_token_prefix"]),
            "placement_id": int(row["placement_id"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "x_norm": float(row.get("x_norm", 0.0)),
            "y_norm": float(row.get("y_norm", 0.0)),
        }

    return lookup


def frames_to_grade_model_tokens(
    frames: str,
    angle: int,
    board_config: BoardConfig,
    df_token_meta,
) -> list[str]:
    """Convert a user-provided frames string into grade-predictor tokens.

    Output format matches training for the grade predictor:

    ``<CLS> <BOARD_...> <ANGLE_...> <BOARDPREFIX_p..._role> ... <EOS>``

    The route is canonicalized using the same role/y/x ordering used during
    tokenization. No grade token is included.
    """
    placement_lookup = build_placement_lookup_from_token_metadata(df_token_meta)
    holds = parse_frames(frames)
    holds = canonicalize_holds(holds, board_config, placement_lookup)

    tokens = [
        "<CLS>",
        board_token(board_config),
        angle_token(angle),
    ]
    tokens.extend(
        hold_token(placement_id, role_id, board_config)
        for placement_id, role_id in holds
    )
    tokens.append("<EOS>")
    return tokens


def predict_frames_grade(
    grade_predictor: LoadedGradePredictor,
    frames: str,
    angle: int,
    board_config: BoardConfig,
    df_token_meta,
) -> dict[str, object]:
    """Predict grade from board, angle, and a frames string."""
    tokens = frames_to_grade_model_tokens(
        frames=frames,
        angle=angle,
        board_config=board_config,
        df_token_meta=df_token_meta,
    )

    # predict_route_grade accepts either <BOS>-style generated tokens or
    # already-prepared <CLS>-style model tokens. It will leave the leading
    # <CLS> intact through the fallback branch.
    pred = predict_route_grade(grade_predictor, tokens)

    return {
        **pred,
        "tokens": tokens,
        "sequence": " ".join(tokens),
        "board_key": board_config.board_key,
        "board_display_name": board_config.display_name,
        "requested_angle": int(angle),
        "frames": frames,
    }
