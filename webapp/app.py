"""FastAPI web demo for ClimbingBoardGPT.

Inference-only design:
  - model checkpoints and token metadata are loaded once at startup;
  - board background images are served as static files;
  - each request returns route hold coordinates/roles as JSON;
  - the browser draws the overlay as SVG on top of the already-loaded image.

Local run:

    uvicorn webapp.app:app --host 127.0.0.1 --port 8055
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from climbingboardgpt.config import load_board_config
from climbingboardgpt.generation import validity_summary
from climbingboardgpt.inference import (
    generate_route,
    load_grade_predictor,
    load_route_generator,
    predict_frames_grade,
    predict_route_grade,
)
from climbingboardgpt.tokenization import tokens_to_hold_records
from climbingboardgpt.utils import json_safe
from climbingboardgpt.visualization import BOARD_CANVAS, load_token_metadata, tokens_to_route_records


# Environment variables keep deployment-specific paths and resource limits out
# of code, while the defaults make a local checkout runnable without config.
DEVICE = os.getenv("CBGPT_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
TORCH_THREADS = os.getenv("CBGPT_TORCH_THREADS")
TORCH_THREADS_INT = int(TORCH_THREADS) if TORCH_THREADS else None

MODEL_DIR = Path(os.getenv("CBGPT_MODEL_DIR", REPO_ROOT / "models"))
TOKENIZED_DIR = Path(
    os.getenv("CBGPT_TOKENIZED_DIR", REPO_ROOT / "data" / "processed" / "tokenized")
)
KNOWN_ROUTES_PATH = Path(
    os.getenv("CBGPT_KNOWN_ROUTES_PATH", TOKENIZED_DIR / "route_sequences.csv")
)

STATIC_DIR = REPO_ROOT / "webapp" / "static"
GENERATOR_PATH = MODEL_DIR / "joint_route_gpt_generator.pth"
GRADE_MODEL_PATH = MODEL_DIR / "joint_transformer_grade_predictor.pth"

BOARD_IMAGE_PATHS = {
    "tb2": REPO_ROOT / "images" / "tb2_board_12x12_composite.png",
    "kilter": REPO_ROOT / "images" / "kilter-original-16x12_composite.png",
}


class GenerateRequest(BaseModel):
    """JSON body for ``POST /api/generate``."""

    board: str = Field(..., pattern="^(tb2|kilter)$")
    angle: int = Field(40, ge=0, le=80)
    grade: int = Field(6, ge=0, le=16)
    temperature: float = Field(0.9, ge=0.1, le=2.0)
    top_k: int = Field(50, ge=1, le=500)
    max_new_tokens: int = Field(40, ge=4, le=80)
    valid_only: bool = Field(True, description="Retry generation until a basic-valid climb is sampled.")
    max_attempts: int = Field(8, ge=1, le=25, description="Maximum attempts when valid_only is true.")


class PredictRequest(BaseModel):
    """JSON body for ``POST /api/predict``."""

    board: str = Field(..., pattern="^(tb2|kilter)$")
    angle: int = Field(..., ge=0, le=80)
    frames: str = Field(..., min_length=1, max_length=500)


def _board_config(board: str):
    """Return the loaded board config or translate unknown boards to HTTP 400."""
    try:
        return app.state.board_configs[board]
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown board: {board}") from exc


def _require_generator():
    """Return the loaded generator or raise HTTP 503 with a useful path hint."""
    if app.state.generator is None:
        raise HTTPException(
            status_code=503,
            detail=f"Route generator checkpoint not loaded. Expected {GENERATOR_PATH}.",
        )
    return app.state.generator


def _require_grade_predictor():
    """Return the loaded grade predictor or raise HTTP 503 with a path hint."""
    if app.state.grade_predictor is None:
        raise HTTPException(
            status_code=503,
            detail=f"Grade predictor checkpoint not loaded. Expected {GRADE_MODEL_PATH}.",
        )
    return app.state.grade_predictor


def _tokens_to_holds(board: str, tokens: list[str]) -> list[dict[str, Any]]:
    """Join route tokens to board coordinates for browser-side SVG drawing."""
    route_records = tokens_to_route_records(tokens)
    if route_records.empty:
        return []

    token_meta = app.state.token_meta
    coords = token_meta[
        (token_meta["kind"] == "hold")
        & (token_meta["board_key"].astype(str) == str(board))
    ][["board_key", "board_token_prefix", "placement_id", "x", "y"]].drop_duplicates(
        ["board_key", "placement_id"]
    )

    merged = route_records.merge(
        coords,
        on=["board_token_prefix", "placement_id"],
        how="left",
    ).dropna(subset=["x", "y"])

    holds = []
    for _, row in merged.iterrows():
        holds.append(
            {
                "token": str(row["token"]),
                "placement_id": int(row["placement_id"]),
                "role": str(row["role"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
            }
        )
    return holds



def _file_version(path: Path) -> str:
    """Small cache-busting version string based on mtime and file size."""
    try:
        stat = path.stat()
        return f"{int(stat.st_mtime)}-{stat.st_size}"
    except FileNotFoundError:
        return "missing"


def _static_image_url(board: str) -> str:
    """Return the static board image URL with a cache-busting query string."""
    image_path = BOARD_IMAGE_PATHS[board]
    return f"/board-images/{image_path.name}?board={board}&v={_file_version(image_path)}"


def _file_info(path: Path) -> dict[str, Any]:
    """Return lightweight debug metadata for a static file."""
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": None,
            "mtime": None,
            "sha256_16": None,
        }
    data = path.read_bytes()
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256_16": hashlib.sha256(data).hexdigest()[:16],
    }


def _board_available_holds(board: str) -> list[dict[str, Any]]:
    """Return clickable hold coordinates for a board.

    Some token-metadata rows can contain missing coordinates for hold-role tokens
    that exist in the vocabulary but cannot be plotted directly. Those rows must
    be removed before returning JSON, because FastAPI's JSONResponse rejects NaN.
    """
    token_meta = app.state.token_meta
    holds = token_meta[
        (token_meta["kind"] == "hold")
        & (token_meta["board_key"].astype(str) == str(board))
    ][["board_key", "board_token_prefix", "placement_id", "x", "y"]].drop_duplicates(
        ["board_key", "placement_id"]
    )

    holds = holds.copy()
    holds["x"] = pd.to_numeric(holds["x"], errors="coerce")
    holds["y"] = pd.to_numeric(holds["y"], errors="coerce")
    holds = holds.dropna(subset=["x", "y"])

    return [
        {
            "placement_id": int(row["placement_id"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
        }
        for _, row in holds.sort_values(["y", "x", "placement_id"]).iterrows()
    ]




def _role_limit_validity(tokens: list[str], requested_board_prefix: str) -> dict[str, Any]:
    """Extra webapp validity checks for start/finish counts.

    The lower-level validity check requires at least one start and at least one
    finish, but climbs should not have arbitrarily many starts or
    finishes. For this demo we enforce at most two starts and at most two finishes.
    """
    counts = {
        "start": 0,
        "middle": 0,
        "finish": 0,
        "foot": 0,
        "unknown": 0,
    }

    for record in tokens_to_hold_records(tokens):
        if str(record["board_token_prefix"]) != str(requested_board_prefix):
            continue
        role = str(record["role"])
        counts[role] = counts.get(role, 0) + 1

    n_start = int(counts.get("start", 0))
    n_finish = int(counts.get("finish", 0))

    return {
        "role_counts": counts,
        "n_start_holds": n_start,
        "n_finish_holds": n_finish,
        "has_at_most_two_starts": n_start <= 2,
        "has_at_most_two_finishes": n_finish <= 2,
        "role_limits_valid": n_start <= 2 and n_finish <= 2,
    }


def _combined_validity(tokens: list[str], requested_board_prefix: str) -> dict[str, Any]:
    """Combined structural validity used by webapp generation and prediction."""
    base = validity_summary(tokens, requested_board_prefix=requested_board_prefix)
    role_limits = _role_limit_validity(tokens, requested_board_prefix=requested_board_prefix)

    base_basic = bool(base.get("basic_valid", False))
    combined_basic = bool(base_basic and role_limits["role_limits_valid"])

    return {
        **base,
        **role_limits,
        "base_basic_valid": base_basic,
        "basic_valid": combined_basic,
        "webapp_basic_valid": combined_basic,
    }


def _invalid_prediction_reasons(validity: dict[str, Any]) -> list[str]:
    """Human-readable reasons a route should not be grade-predicted."""
    reasons: list[str] = []

    if int(validity.get("n_hold_tokens", 0)) == 0:
        reasons.append("no hold tokens were found in the frames string")
    if not bool(validity.get("one_board_only", True)):
        reasons.append("the route contains holds from more than one board")
    if not bool(validity.get("matches_requested_board", True)):
        reasons.append("the route contains holds from the wrong board")
    if bool(validity.get("has_duplicate_placements", False)):
        reasons.append("the route contains duplicate placements")
    if not bool(validity.get("has_start", False)):
        reasons.append("the route has no start hold")
    if not bool(validity.get("has_finish", False)):
        reasons.append("the route has no finish hold")
    if int(validity.get("n_start_holds", 0)) > 2:
        reasons.append("the route has more than two start holds")
    if int(validity.get("n_finish_holds", 0)) > 2:
        reasons.append("the route has more than two finish holds")
    if int(validity.get("n_hold_tokens", 0)) < 3:
        reasons.append("the route has fewer than 3 holds")

    return reasons


def _angle_key(angle: Any) -> int:
    """Normalize angle-like values for signatures and selectors."""
    try:
        return int(round(float(angle)))
    except Exception:
        return 0


def _route_signature_from_holds(board: str, angle: Any, holds: list[dict[str, Any]]) -> str | None:
    """Canonical exact-match signature: board + angle + hold-role multiset."""
    parts = []
    for hold in holds:
        try:
            placement_id = int(hold["placement_id"])
            role = str(hold["role"])
        except Exception:
            continue
        parts.append(f"{role}:{placement_id}")

    if not parts:
        return None

    return f"{board}|{_angle_key(angle)}|" + "|".join(sorted(parts))


def _holds_from_sequence(sequence: str) -> list[dict[str, Any]]:
    """Extract exact-match hold-role records from a no-grade token sequence."""
    holds: list[dict[str, Any]] = []
    for record in tokens_to_hold_records(str(sequence).split()):
        holds.append(
            {
                "board_prefix": str(record["board_token_prefix"]),
                "placement_id": int(record["placement_id"]),
                "role": str(record["role"]),
            }
        )
    return holds



def _load_available_angles(path: Path) -> dict[str, list[int]]:
    """Load available wall angles by board from route_sequences.csv.

    Falls back to a conservative common angle set if the processed route table
    is unavailable. This keeps the webapp usable for demos even if only model
    checkpoints/token metadata are present.
    """
    fallback = {
        "tb2": [20, 25, 30, 35, 40, 45, 50],
        "kilter": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    }
    if not path.exists():
        return fallback

    try:
        df = pd.read_csv(path, usecols=["board_key", "angle"])
    except Exception:
        return fallback

    out: dict[str, list[int]] = {}
    for board, frame in df.dropna(subset=["board_key", "angle"]).groupby("board_key"):
        values = sorted({_angle_key(angle) for angle in frame["angle"].tolist()})
        if values:
            out[str(board)] = values

    for board, values in fallback.items():
        out.setdefault(board, values)

    return out



def _load_available_grades(path: Path) -> dict[str, list[int]]:
    """Load available grouped V-grades by board from route_sequences.csv."""
    fallback = {
        "tb2": list(range(0, 16)),
        "kilter": list(range(0, 16)),
    }
    if not path.exists():
        return fallback

    try:
        df = pd.read_csv(path, usecols=["board_key", "grouped_v"])
    except Exception:
        return fallback

    out: dict[str, list[int]] = {}
    for board, frame in df.dropna(subset=["board_key", "grouped_v"]).groupby("board_key"):
        values = sorted({int(v) for v in frame["grouped_v"].tolist()})
        if values:
            out[str(board)] = values

    for board, values in fallback.items():
        out.setdefault(board, values)

    return out


def _load_known_route_lookup(path: Path) -> dict[str, dict[str, Any]]:
    """Load exact known-route signatures once at startup.

    This is intentionally an exact O(1) lookup, not a nearest-neighbor search.
    The key is board + angle + exact hold-role set. Multiple real climbs can
    share the same signature, so each signature stores a count and a few examples.
    """
    lookup: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return lookup

    usecols = [
        "uuid",
        "board_key",
        "climb_name",
        "setter_username",
        "angle",
        "grouped_v",
        "boulder_grade",
        "quality_average",
        "ascensionist_count",
        "frames",
        "sequence_no_grade",
    ]

    try:
        df = pd.read_csv(path, usecols=lambda col: col in set(usecols))
    except Exception:
        return lookup

    for _, row in df.iterrows():
        board = str(row.get("board_key", ""))
        angle = row.get("angle", 0)
        holds = _holds_from_sequence(str(row.get("sequence_no_grade", "")))
        signature = _route_signature_from_holds(board, angle, holds)
        if signature is None:
            continue

        entry = lookup.setdefault(signature, {"count": 0, "examples": []})
        entry["count"] += 1

        if len(entry["examples"]) < 5:
            example = {
                "uuid": str(row.get("uuid", "")),
                "climb_name": None if pd.isna(row.get("climb_name", None)) else str(row.get("climb_name", "")),
                "setter_username": None if pd.isna(row.get("setter_username", None)) else str(row.get("setter_username", "")),
                "angle": _angle_key(angle),
                "grouped_v": None if pd.isna(row.get("grouped_v", None)) else int(row.get("grouped_v")),
                "boulder_grade": None if pd.isna(row.get("boulder_grade", None)) else str(row.get("boulder_grade", "")),
                "quality_average": None if pd.isna(row.get("quality_average", None)) else float(row.get("quality_average")),
                "ascensionist_count": None if pd.isna(row.get("ascensionist_count", None)) else int(row.get("ascensionist_count")),
                "frames": None if pd.isna(row.get("frames", None)) else str(row.get("frames", "")),
            }
            entry["examples"].append(example)

    return lookup


def _known_route_status(board: str, angle: Any, holds: list[dict[str, Any]]) -> dict[str, Any]:
    """Check whether a route exactly matches a tokenized dataset route."""
    signature = _route_signature_from_holds(board, angle, holds)
    if signature is None:
        return {
            "checked": bool(getattr(app.state, "known_route_lookup", None)),
            "is_known": False,
            "match_count": 0,
            "examples": [],
            "signature": None,
        }

    lookup = getattr(app.state, "known_route_lookup", {})
    match = lookup.get(signature)

    return {
        "checked": bool(lookup),
        "is_known": match is not None,
        "match_count": int(match["count"]) if match else 0,
        "examples": match["examples"] if match else [],
        "signature": signature,
    }


def _payload(result: dict[str, Any], tokens: list[str] | None = None) -> dict[str, Any]:
    """Attach drawable holds, canvas data, image URL, and known-route status."""
    board = str(result["board_key"])
    tokens = list(tokens if tokens is not None else result.get("tokens", []))
    extent = [float(v) for v in BOARD_CANVAS[board]["extent"]]
    image_path = BOARD_IMAGE_PATHS[board]
    holds = _tokens_to_holds(board, tokens)
    angle = result.get("requested_angle", result.get("angle", 0))

    return json_safe(
        {
            **result,
            "tokens": tokens,
            "holds": holds,
            "known_climb": _known_route_status(board, angle, holds),
            "canvas": {
                "extent": extent,
                "x_min": extent[0],
                "x_max": extent[1],
                "y_min": extent[2],
                "y_max": extent[3],
                "width": extent[1] - extent[0],
                "height": extent[3] - extent[2],
            },
            "background_url": _static_image_url(board),
        }
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model/checkpoint state once for the process lifetime."""
    if TORCH_THREADS_INT is not None:
        torch.set_num_threads(TORCH_THREADS_INT)

    app.state.board_configs = {
        "tb2": load_board_config("tb2", config_dir=REPO_ROOT / "configs"),
        "kilter": load_board_config("kilter", config_dir=REPO_ROOT / "configs"),
    }
    app.state.token_meta = load_token_metadata(TOKENIZED_DIR)
    app.state.available_angles = _load_available_angles(KNOWN_ROUTES_PATH)
    app.state.available_grades = _load_available_grades(KNOWN_ROUTES_PATH)
    started = time.time()
    app.state.known_route_lookup = _load_known_route_lookup(KNOWN_ROUTES_PATH)
    app.state.known_route_lookup_seconds = round(time.time() - started, 3)

    app.state.generator = None
    if GENERATOR_PATH.exists():
        app.state.generator = load_route_generator(
            GENERATOR_PATH,
            device=DEVICE,
            torch_threads=TORCH_THREADS_INT,
        )

    app.state.grade_predictor = None
    if GRADE_MODEL_PATH.exists():
        app.state.grade_predictor = load_grade_predictor(
            GRADE_MODEL_PATH,
            device=DEVICE,
            torch_threads=TORCH_THREADS_INT,
        )
    yield


app = FastAPI(title="ClimbingBoardGPT", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=REPO_ROOT / "webapp" / "static"), name="static")
app.mount("/board-images", StaticFiles(directory=REPO_ROOT / "images"), name="board-images")


@app.get("/")
def index():
    """Serve the single-page web UI with auto-versioned static asset URLs."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    css_v = _file_version(STATIC_DIR / "app.css")
    js_v = _file_version(STATIC_DIR / "app.js")
    html = re.sub(r'app\.css\?v=[^\s"\']+', f"app.css?v={css_v}", html)
    html = re.sub(r'app\.js\?v=[^\s"\']+', f"app.js?v={js_v}", html)
    return HTMLResponse(content=html)


@app.get("/api/health")
def health():
    """Return runtime readiness and deployment diagnostics."""
    return {
        "ok": True,
        "device": DEVICE,
        "generator_loaded": app.state.generator is not None,
        "grade_predictor_loaded": app.state.grade_predictor is not None,
        "known_route_signatures": len(getattr(app.state, "known_route_lookup", {})),
        "known_route_lookup_seconds": getattr(app.state, "known_route_lookup_seconds", None),
        "available_angles": getattr(app.state, "available_angles", {}),
        "available_grades": getattr(app.state, "available_grades", {}),
        "torch_threads": torch.get_num_threads(),
    }


@app.get("/api/boards")
def boards():
    """Return board metadata needed to initialize client-side controls."""
    payload = {}
    for board, config in app.state.board_configs.items():
        extent = [float(v) for v in BOARD_CANVAS[board]["extent"]]
        payload[board] = {
            "board_key": board,
            "display_name": config.display_name,
            "token_prefix": config.token_prefix,
            "role_definitions": config.role_definitions,
            "available_angles": getattr(app.state, "available_angles", {}).get(board, []),
            "available_grades": getattr(app.state, "available_grades", {}).get(board, []),
            "background_url": _static_image_url(board),
            "canvas": {
                "extent": extent,
                "x_min": extent[0],
                "x_max": extent[1],
                "y_min": extent[2],
                "y_max": extent[3],
                "width": extent[1] - extent[0],
                "height": extent[3] - extent[2],
            },
        }
    return payload


@app.get("/api/board-holds/{board}")
def board_holds(board: str):
    """Return all clickable holds for a board."""
    config = _board_config(board)
    return json_safe({
        "board_key": board,
        "display_name": config.display_name,
        "token_prefix": config.token_prefix,
        "role_definitions": config.role_definitions,
        "holds": _board_available_holds(board),
    })


@app.get("/api/debug/images")
def debug_images():
    """Return static-board-image debug metadata."""
    payload = {}
    for board, image_path in BOARD_IMAGE_PATHS.items():
        payload[board] = {
            "background_url": _static_image_url(board),
            "file": _file_info(image_path),
        }
    return payload


@app.post("/api/generate")
def generate(req: GenerateRequest):
    """Generate a climb and optionally retry until webapp validity passes."""
    generator = _require_generator()
    config = _board_config(req.board)

    attempts = req.max_attempts if req.valid_only else 1
    result = None
    sampled_results = []

    for attempt in range(1, attempts + 1):
        candidate = generate_route(
            generator=generator,
            board_config=config,
            angle=req.angle,
            grade=req.grade,
            temperature=req.temperature,
            top_k=req.top_k,
            max_new_tokens=req.max_new_tokens,
        )
        candidate["generation_attempt"] = attempt
        combined_validity = _combined_validity(
            list(candidate.get("tokens", [])),
            requested_board_prefix=config.token_prefix,
        )
        candidate.update(combined_validity)
        candidate["webapp_validity"] = combined_validity

        sampled_results.append(
            {
                "attempt": attempt,
                "basic_valid": bool(candidate.get("basic_valid")),
                "base_basic_valid": bool(candidate.get("base_basic_valid")),
                "role_limits_valid": bool(candidate.get("role_limits_valid")),
                "n_hold_tokens": int(candidate.get("n_hold_tokens", 0)),
                "n_start_holds": int(candidate.get("n_start_holds", 0)),
                "n_finish_holds": int(candidate.get("n_finish_holds", 0)),
                "frames": candidate.get("frames", ""),
            }
        )

        result = candidate
        if not req.valid_only or bool(candidate.get("basic_valid")):
            break

    if result is None:
        raise HTTPException(status_code=500, detail="Generation failed unexpectedly.")

    result["requested_valid_only"] = bool(req.valid_only)
    result["max_attempts"] = int(attempts)
    result["attempts_used"] = int(result.get("generation_attempt", 1))
    result["sampled_attempts"] = sampled_results

    warnings = []
    if req.valid_only and not bool(result.get("basic_valid")):
        warnings.append(
            f"Could not sample a valid climb after {attempts} attempts. Showing the last sample."
        )
        if not bool(result.get("role_limits_valid", True)):
            warnings.append(
                "Last sample violates start/finish role limits: at most two starts and at most two finishes."
            )
    elif req.valid_only and result["attempts_used"] > 1:
        warnings.append(
            f"Sampled {result['attempts_used']} climbs before finding a valid one."
        )
    elif not bool(result.get("basic_valid")):
        warnings.append("Generated climb is structurally invalid.")

    if app.state.grade_predictor is not None:
        grade_result = predict_route_grade(app.state.grade_predictor, result["tokens"])
        result.update(grade_result)
        result["critic_v_error"] = (
            int(result["predicted_grouped_v"]) - int(result["requested_grouped_v"])
        )

    if "warnings" in result and isinstance(result["warnings"], list):
        result["warnings"].extend(warnings)
    else:
        result["warnings"] = warnings

    return _payload(result)


@app.post("/api/predict")
def predict(req: PredictRequest):
    """Predict grade for a user-supplied frames string after validity checks."""
    predictor = _require_grade_predictor()
    config = _board_config(req.board)

    result = predict_frames_grade(
        grade_predictor=predictor,
        frames=req.frames,
        angle=req.angle,
        board_config=config,
        df_token_meta=app.state.token_meta,
    )

    tokens = list(result["tokens"])
    validity = _combined_validity(tokens, requested_board_prefix=config.token_prefix)
    result.update(validity)
    result["webapp_validity"] = validity

    if not bool(validity.get("basic_valid", False)):
        reasons = _invalid_prediction_reasons(validity)
        raise HTTPException(
            status_code=400,
            detail=json_safe(
                {
                    "message": "Cannot predict grade for an invalid climb.",
                    "reasons": reasons,
                    "validity": validity,
                }
            ),
        )

    return _payload(result, tokens=tokens)
