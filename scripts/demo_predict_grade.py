#!/usr/bin/env python3
"""Predict a climb grade from board, angle, and BoardLib frames string.

Examples
--------
Generic:

    python scripts/demo_predict_grade.py \
      --board tb2 \
      --angle 40 \
      --frames 'p652r5p631r6p322r6p326r7'

TB2 wrapper:

    python scripts/demo_predict_tb2.py \
      --angle 40 \
      --frames 'p652r5p631r6p322r6p326r7'

Kilter wrapper:

    python scripts/demo_predict_kilter.py \
      --angle 40 \
      --frames 'p1127r12p1196r13p1388r14'

Add ``--visualize`` to save a PNG/SVG overlay using the board background.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt

from climbingboardgpt.inference import (
    frames_to_grade_model_tokens,
    load_board_for_demo,
    load_grade_predictor,
    predict_frames_grade,
)
from climbingboardgpt.visualization import load_token_metadata, visualize_route_tokens


def default_background_for_board(board: str) -> Path | None:
    candidates = {
        "tb2": REPO_ROOT / "images" / "tb2_board_12x12_composite.png",
        "kilter": REPO_ROOT / "images" / "kilter-original-16x12_composite.png",
    }
    path = candidates.get(board)
    return path if path is not None and path.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict climb grade from board, angle, and frames string.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--board", choices=["tb2", "kilter"], required=True)
    parser.add_argument("--angle", type=int, required=True)
    parser.add_argument("--frames", type=str, required=True)
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or omit for auto.")
    parser.add_argument("--torch-threads", type=int, default=None, help="Optional CPU thread cap.")
    parser.add_argument(
        "--grade-model-path",
        type=Path,
        default=REPO_ROOT / "models" / "joint_transformer_grade_predictor.pth",
    )
    parser.add_argument(
        "--tokenized-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "tokenized",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of human-readable text.")
    parser.add_argument("--show-tokens", action="store_true", help="Print the model token sequence.")
    parser.add_argument("--visualize", action="store_true", help="Save a board-background visualization.")
    parser.add_argument("--annotate", action="store_true", help="Label route holds by placement ID.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "grade_predictions",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output image/table stem. Defaults to <board>_angle_<angle>_prediction.",
    )
    parser.add_argument(
        "--background-image",
        type=Path,
        default=None,
        help="Optional background image override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    board_config = load_board_for_demo(args.board, config_dir=REPO_ROOT / "configs")
    token_meta = load_token_metadata(args.tokenized_dir)
    predictor = load_grade_predictor(
        args.grade_model_path,
        device=args.device,
        torch_threads=args.torch_threads,
    )

    result = predict_frames_grade(
        grade_predictor=predictor,
        frames=args.frames,
        angle=args.angle,
        board_config=board_config,
        df_token_meta=token_meta,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Board:        {result['board_display_name']} ({result['board_key']})")
        print(f"Angle:        {result['requested_angle']}°")
        print(f"Frames:       {result['frames']}")
        print(f"Predicted:    V{result['predicted_grouped_v']}")
        print(f"Difficulty:   {result['predicted_display_difficulty']:.3f}")
        if args.show_tokens:
            print()
            print("Model tokens:")
            print(result["sequence"])

    if args.visualize:
        out_dir = args.out_dir / args.board / f"angle_{args.angle}"
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = args.output_name or f"{args.board}_angle_{args.angle}_prediction"
        png_path = out_dir / f"{stem}.png"
        svg_path = out_dir / f"{stem}.svg"
        json_path = out_dir / f"{stem}.json"

        background_image = args.background_image or default_background_for_board(args.board)
        title = (
            f"{result['board_display_name']} predicted "
            f"V{result['predicted_grouped_v']} @ {args.angle}°"
        )
        subtitle = (
            f"difficulty={result['predicted_display_difficulty']:.2f} | "
            f"frames={args.frames}"
        )

        fig, _, _ = visualize_route_tokens(
            tokens=result["tokens"],
            df_token_meta=token_meta,
            board_key=args.board,
            title=title,
            subtitle=subtitle,
            output_path=png_path,
            annotate=args.annotate,
            background_image=background_image,
        )
        fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)

        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        print()
        if background_image is not None:
            try:
                bg_display = background_image.relative_to(REPO_ROOT)
            except Exception:
                bg_display = background_image
            print(f"Using background image: {bg_display}")
        print(f"Saved PNG:  {png_path}")
        print(f"Saved SVG:  {svg_path}")
        print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
