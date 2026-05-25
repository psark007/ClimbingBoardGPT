#!/usr/bin/env python3
"""Generate ClimbingBoardGPT routes and save board visualizations.

Examples
--------
Generate four TB2 V6 climbs at 40 degrees:

    python scripts/demo_generate_and_visualize.py --board tb2 --angle 40 --grade 6 --n 4

Generate Kilter climbs with placement labels:

    python scripts/demo_generate_and_visualize.py --board kilter --angle 35 --grade 5 --annotate

The script writes:
  - generated_routes.csv
  - generated_route_001.png
  - generated_route_001.svg
  - ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from climbingboardgpt.inference import (
    generate_route,
    load_board_for_demo,
    load_grade_predictor,
    load_route_generator,
    predict_route_grade,
)
from climbingboardgpt.visualization import load_token_metadata, visualize_route_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ClimbingBoardGPT routes and save route visualizations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--board", choices=["tb2", "kilter"], required=True)
    parser.add_argument("--angle", type=int, default=40)
    parser.add_argument("--grade", type=int, default=6, help="Target grouped V-grade.")
    parser.add_argument("--n", type=int, default=4, help="Number of routes to sample.")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--annotate", action="store_true", help="Label route holds by placement ID.")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or omit for auto.")
    parser.add_argument("--torch-threads", type=int, default=None, help="Optional CPU thread cap for VPS demos.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "models" / "joint_route_gpt_generator.pth",
    )
    parser.add_argument(
        "--grade-model-path",
        type=Path,
        default=REPO_ROOT / "models" / "joint_transformer_grade_predictor.pth",
        help="Optional grade-predictor checkpoint used to score generated routes.",
    )
    parser.add_argument(
        "--no-grade-prediction",
        action="store_true",
        help="Skip grade-predictor scoring even if the checkpoint exists.",
    )
    parser.add_argument(
        "--tokenized-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "tokenized",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "demo_routes",
    )
    parser.add_argument(
        "--background-image",
        type=Path,
        default=None,
        help=(
            "Optional board image to draw under the scatter plot. "
            "If omitted, the script automatically uses images/tb2_board_12x12_composite.png "
            "for TB2 and images/kilter-original-16x12_composite.png for Kilter when present."
        ),
    )
    return parser.parse_args()



def default_background_for_board(board: str) -> Path | None:
    candidates = {
        "tb2": REPO_ROOT / "images" / "tb2_board_12x12_composite.png",
        "kilter": REPO_ROOT / "images" / "kilter-original-16x12_composite.png",
    }
    path = candidates.get(board)
    return path if path is not None and path.exists() else None


def main() -> None:
    args = parse_args()

    board_config = load_board_for_demo(args.board, config_dir=REPO_ROOT / "configs")
    generator = load_route_generator(args.model_path, device=args.device, torch_threads=args.torch_threads)
    token_meta = load_token_metadata(args.tokenized_dir)
    background_image = args.background_image or default_background_for_board(args.board)

    grade_predictor = None
    if not args.no_grade_prediction:
        if args.grade_model_path.exists():
            grade_predictor = load_grade_predictor(
                args.grade_model_path,
                device=args.device,
                torch_threads=args.torch_threads,
            )
        else:
            print(f"Grade predictor not found at {args.grade_model_path}; skipping grade prediction.")

    run_dir = args.out_dir / args.board / f"angle_{args.angle}" / f"V{args.grade}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(1, args.n + 1):
        result = generate_route(
            generator=generator,
            board_config=board_config,
            angle=args.angle,
            grade=args.grade,
            temperature=args.temperature,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
        )

        if grade_predictor is not None:
            grade_result = predict_route_grade(grade_predictor, result["tokens"])
            result.update(grade_result)
            result["critic_v_error"] = (
                int(result["predicted_grouped_v"]) - int(result["requested_grouped_v"])
            )

        rows.append(result)

        stem = f"generated_route_{i:03d}"
        png_path = run_dir / f"{stem}.png"
        svg_path = run_dir / f"{stem}.svg"

        fig, _, _ = visualize_route_result(
            result,
            df_token_meta=token_meta,
            output_path=png_path,
            annotate=args.annotate,
            background_image=background_image,
        )
        fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)

        print(f"[{i}/{args.n}] {result['frames']}")
        print(f"       valid={result['basic_valid']} holds={result['n_hold_tokens']}")
        if "predicted_grouped_v" in result:
            print(
                f"       predicted=V{result['predicted_grouped_v']} "
                f"(difficulty={result['predicted_display_difficulty']:.2f}, "
                f"error={result['critic_v_error']:+d} V)"
            )
        try:
            png_display = png_path.resolve().relative_to(REPO_ROOT.resolve())
        except Exception:
            png_display = png_path
        print(f"       saved {png_display}")

    df = pd.DataFrame(rows)
    df["tokens_json"] = df["tokens"].apply(json.dumps)
    df.drop(columns=["tokens"]).to_csv(run_dir / "generated_routes.csv", index=False)

    if background_image is not None:
        try:
            bg_display = background_image.relative_to(REPO_ROOT)
        except Exception:
            bg_display = background_image
        print(f"Using background image: {bg_display}")
    else:
        print("Using background image: none (coordinate-board style only)")

    print("\nSaved route table:")
    print(run_dir / "generated_routes.csv")
    print("\nOutput directory:")
    print(run_dir)


if __name__ == "__main__":
    main()
