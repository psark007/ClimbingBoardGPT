"""Visualization utilities for generated ClimbingBoardGPT routes.

The route-overlay functions here deliberately mimic the old TB2/Kilter
notebook convention: draw the board composite image with the product-size
coordinate extent, then scatter route holds in board coordinates.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from .tokenization import parse_tokens, tokens_to_hold_records

# These are the same coordinate windows used in the earlier visualization
# notebooks. They come from the product size geometry rather than from the
# min/max of the actual holds. The hold coordinates are inset by about 4in,
# so using hold min/max directly shifts/stretches the background image.
BOARD_CANVAS = {
    "tb2": {
        "extent": [-68, 68, 0, 144],
        "figsize": (16, 14),
        "image_aspect": "auto",
    },
    "kilter": {
        "extent": [-24, 168, 0, 156],
        "figsize": (17, 12),
        "image_aspect": "equal",
    },
}

ROLE_COLORS = {
    "start": "#2ecc71",
    "middle": "#3498db",
    "finish": "#e74c3c",
    "foot": "#f1c40f",
    "unknown": "#9ca3af",
}

ROLE_MARKERS = {
    "start": "o",
    "middle": "o",
    "finish": "*",
    "foot": "s",
    "unknown": "o",
}

ROLE_SIZES = {
    "start": 150,
    "middle": 150,
    "finish": 230,
    "foot": 95,
    "unknown": 150,
}


def tokens_to_route_records(tokens: Iterable[str]) -> pd.DataFrame:
    """Extract generated hold records from model tokens."""
    return pd.DataFrame(tokens_to_hold_records(tokens))


def load_token_metadata(tokenized_dir: str | Path) -> pd.DataFrame:
    """Load token metadata produced by ``scripts/01_tokenize_routes.py``."""
    tokenized_dir = Path(tokenized_dir)
    path = tokenized_dir / "token_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run scripts/01_tokenize_routes.py first."
        )
    return pd.read_csv(path)


def board_canvas_settings(board_key: str, df_token_meta: pd.DataFrame | None = None) -> dict[str, object]:
    """Return board canvas settings.

    Known boards use hand-calibrated extents from the old notebooks. Unknown
    boards fall back to coordinate bounds from ``token_metadata.csv``.
    """
    board_key = str(board_key)
    if board_key in BOARD_CANVAS:
        return dict(BOARD_CANVAS[board_key])

    if df_token_meta is None:
        raise ValueError(f"No board canvas settings for board_key={board_key!r}.")

    holds = _board_holds(df_token_meta, board_key)
    x_min, x_max = float(holds["x"].min()), float(holds["x"].max())
    y_min, y_max = float(holds["y"].min()), float(holds["y"].max())
    x_pad = max((x_max - x_min) * 0.06, 1.0)
    y_pad = max((y_max - y_min) * 0.06, 1.0)
    return {
        "extent": [x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad],
        "figsize": (8, 10),
        "image_aspect": "auto",
    }


def _board_holds(df_token_meta: pd.DataFrame, board_key: str) -> pd.DataFrame:
    """Return one metadata row per plotted hold for a board."""
    holds = df_token_meta[
        (df_token_meta["kind"] == "hold")
        & (df_token_meta["board_key"].astype(str) == str(board_key))
    ].copy()

    if holds.empty:
        raise ValueError(
            f"No hold metadata found for board_key={board_key!r}. "
            "Check token_metadata.csv and board config."
        )

    holds = holds.drop_duplicates(["board_key", "placement_id"]).copy()
    return holds


def _route_with_coords(
    route_records: pd.DataFrame,
    df_token_meta: pd.DataFrame,
    board_key: str,
) -> pd.DataFrame:
    """Attach x/y coordinates to route hold records using token metadata."""
    holds = _board_holds(df_token_meta, board_key)
    coords = holds[["board_key", "board_token_prefix", "placement_id", "x", "y"]].drop_duplicates(
        ["board_key", "placement_id"]
    )

    merged = route_records.merge(
        coords,
        on=["board_token_prefix", "placement_id"],
        how="left",
    )
    return merged


def visualize_route_tokens(
    tokens: Iterable[str],
    df_token_meta: pd.DataFrame,
    board_key: str,
    title: str | None = None,
    subtitle: str | None = None,
    output_path: str | Path | None = None,
    annotate: bool = False,
    show_all_holds: bool | None = None,
    background_image: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 160,
):
    """Visualize a generated route as a board overlay plot.

    If a background image is supplied, the plot uses the calibrated canvas
    extent from the old project notebooks. If no image is supplied, it falls
    back to a clean coordinate-board style and shows available holds.
    """
    route_records = tokens_to_route_records(tokens)
    if route_records.empty:
        raise ValueError("No hold tokens found in generated sequence.")

    board_holds = _board_holds(df_token_meta, board_key)
    route_df = _route_with_coords(route_records, df_token_meta, board_key)
    route_df = route_df.dropna(subset=["x", "y"]).copy()

    if route_df.empty:
        raise ValueError(
            "Generated route contained hold tokens, but none matched the board metadata."
        )

    canvas = board_canvas_settings(board_key, df_token_meta)
    extent = [float(v) for v in canvas["extent"]]
    x_min, x_max, y_min, y_max = extent
    image_aspect = str(canvas.get("image_aspect", "auto"))
    figsize = figsize or canvas.get("figsize", (8, 10))

    background_exists = background_image is not None and Path(background_image).exists()
    if show_all_holds is None:
        show_all_holds = not background_exists

    fig, ax = plt.subplots(figsize=figsize)

    if background_exists:
        img = plt.imread(Path(background_image))
        ax.imshow(
            img,
            extent=extent,
            aspect=image_aspect,
            alpha=1.0,
            zorder=0,
        )

    if show_all_holds:
        ax.scatter(
            board_holds["x"],
            board_holds["y"],
            s=22,
            c="#d1d5db",
            alpha=0.45,
            linewidths=0,
            label="available holds",
            zorder=1,
        )

    # Draw route holds role-by-role so the legend is meaningful.
    for role, frame in route_df.groupby("role", sort=False):
        ax.scatter(
            frame["x"],
            frame["y"],
            s=ROLE_SIZES.get(role, 150),
            c=ROLE_COLORS.get(role, ROLE_COLORS["unknown"]),
            marker=ROLE_MARKERS.get(role, "o"),
            edgecolors="#111827",
            linewidths=1.0,
            alpha=0.96,
            label=role,
            zorder=3,
        )

    if annotate:
        for _, row in route_df.iterrows():
            ax.text(
                row["x"],
                row["y"],
                str(int(row["placement_id"])),
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
                bbox=dict(
                    boxstyle="circle,pad=0.12",
                    alpha=0.45,
                    facecolor="#111827",
                    edgecolor="white",
                    linewidth=0.8,
                ),
                zorder=4,
            )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    if image_aspect == "equal":
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    # Put the title and subtitle at the figure level, not the axes level.
    # This avoids the old overlap where ax.set_title(...) and ax.text(y=1.01)
    # competed for the same narrow top margin.
    has_header = bool(title or subtitle)
    if title:
        fig.suptitle(
            title,
            fontsize=14,
            fontweight="bold",
            y=0.985,
        )
    if subtitle:
        fig.text(
            0.5,
            0.958,
            subtitle,
            ha="center",
            va="top",
            fontsize=9,
            color="#4b5563",
        )

    if background_exists:
        ax.grid(False)
    else:
        ax.grid(True, alpha=0.18)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

    # Reserve top space for the figure-level title/subtitle.
    if has_header:
        fig.tight_layout(rect=[0, 0, 1, 0.925])
    else:
        fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig, ax, route_df


def visualize_route_result(
    result: dict[str, object],
    df_token_meta: pd.DataFrame,
    output_path: str | Path | None = None,
    annotate: bool = False,
    background_image: str | Path | None = None,
):
    """Visualize a result dictionary returned by ``generate_route``."""
    board_key = str(result["board_key"])
    tokens = parse_tokens(result["tokens"])
    title = (
        f"{str(result.get('board_display_name', board_key))} "
        f"generated V{int(result['requested_grouped_v'])} @ {int(result['requested_angle'])}°"
    )
    subtitle_parts = [
        f"valid={result.get('basic_valid')}",
        f"holds={result.get('n_hold_tokens')}",
    ]
    if "predicted_grouped_v" in result:
        subtitle_parts.append(
            f"predicted V{int(result['predicted_grouped_v'])}"
            f" ({float(result['predicted_display_difficulty']):.2f})"
        )
        if "critic_v_error" in result:
            subtitle_parts.append(f"error {int(result['critic_v_error']):+d}V")
    subtitle_parts.append(f"temperature={result.get('temperature')}")
    subtitle = " | ".join(subtitle_parts)
    return visualize_route_tokens(
        tokens=tokens,
        df_token_meta=df_token_meta,
        board_key=board_key,
        title=title,
        subtitle=subtitle,
        output_path=output_path,
        annotate=annotate,
        background_image=background_image,
    )
