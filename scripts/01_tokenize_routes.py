#!/usr/bin/env python3
"""
ClimbingBoardGPT — Route Tokenization Script

This script converts raw climbing route data from SQLite databases into
tokenized sequences suitable for training transformer models.

What is tokenization?
---------------------
In NLP, tokenization converts raw text into discrete symbols (tokens) that
a model can process. For example, GPT-2 uses Byte-Pair Encoding (BPE) to
split "climbing" into ["cl", "imb", "ing"].

For climbing routes, we tokenize differently:
- Each hold on the board becomes a unique token (e.g., <TB2_p344_start>)
- Board identity, angle, and grade become conditioning tokens
- Special tokens mark sequence boundaries (<BOS>, <EOS>, etc.)

The key insight: climbing routes ARE sequences, just like sentences. The
same transformer architectures that learn English grammar can learn "climb
grammar" — which holds tend to follow which, how start holds differ from
finish holds, etc.

This script:
1. Loads board configurations from JSON files
2. Queries SQLite databases for climb and placement data
3. Parses frame strings (e.g., "p344r5p369r6p603r7") into structured data
4. Maps board-specific role IDs to shared semantic roles
5. Canonicalizes hold order (starts first, then middles by Y, etc.)
6. Generates two token sequences per route:
   - with_grade: includes <GRADE_V6> for GPT training
   - without_grade: excludes grade for BERT-style prediction
7. Builds vocabulary, train/val/test splits, and saves all artifacts

Usage:
    python scripts/01_tokenize_routes.py --boards tb2,kilter
    python scripts/01_tokenize_routes.py --boards tb2
    python scripts/01_tokenize_routes.py --boards kilter
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Set up the project root so we can import our custom package
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from climbingboardgpt.config import load_board_configs, parse_board_keys
from climbingboardgpt.data import load_multi_board_data
from climbingboardgpt.tokenization import (
    build_route_records,
    build_token_metadata,
    build_vocab,
    encode,
    make_placement_lookup,
    vocab_payload,
)
from climbingboardgpt.utils import assign_group_splits, json_safe, set_seed, write_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the tokenization script.
    
    Key arguments:
        --boards: Which boards to tokenize (comma-separated). Default: "tb2,kilter"
        --out-dir: Where to save tokenized artifacts
        --seed: Random seed for reproducible train/val/test splits
    """
    parser = argparse.ArgumentParser(
        description="Tokenize TB2 and/or Kilter routes for ClimbingBoardGPT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tokenize both boards (default)
  python scripts/01_tokenize_routes.py --boards tb2,kilter

  # Tokenize only TB2
  python scripts/01_tokenize_routes.py --boards tb2

  # Custom output directory
  python scripts/01_tokenize_routes.py --out-dir /path/to/output
        """,
    )
    parser.add_argument(
        "--boards",
        type=str,
        default="tb2,kilter",
        help="Comma-separated board config names (default: tb2,kilter)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "tokenized",
        help="Output directory for tokenized artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3,
        help="Random seed for reproducible splits (default: 3)",
    )
    parser.add_argument(
        "--max-routes-per-board",
        type=int,
        default=None,
        help="Optional smoke-test row limit per board before tokenization.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for route tokenization.
    
    This function orchestrates the entire tokenization pipeline:
    1. Load board configurations
    2. Query databases for raw climb and placement data
    3. Parse frames strings into structured hold records
    4. Build tokenized route records with canonical hold ordering
    5. Construct vocabulary from all unique tokens
    6. Split data into train/val/test sets (stratified by board × grade)
    7. Build token metadata (coordinates, roles, etc.)
    8. Save all artifacts to disk
    """
    args = parse_args()
    if args.max_routes_per_board is not None and args.max_routes_per_board < 3:
        raise ValueError("--max-routes-per-board must be at least 3 so train/val/test splits can exist.")
    
    # Set random seed for reproducibility
    # This ensures train/val/test splits are the same across runs
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Load board configurations
    # ─────────────────────────────────────────────────────────────────────
    # Each board has a JSON config file specifying:
    # - layout_id: Which layout in the database to use
    # - role_definitions: Maps semantic roles (start, middle, etc.) to numeric IDs
    # - max_angle: Filter out routes steeper than this
    # - token_prefix: Namespace for hold tokens (prevents ID collisions)
    # 
    # This config-driven approach means adding a new board only requires
    # creating a new JSON file, not modifying code.
    board_keys = parse_board_keys(args.boards)
    configs = load_board_configs(board_keys)
    configs_by_key = {config.board_key: config for config in configs}
    configs_by_prefix = {config.token_prefix: config for config in configs}

    print(f"Loaded {len(configs)} board configuration(s):")
    for config in configs:
        print(f"  {config.display_name} (key={config.board_key}, prefix={config.token_prefix})")
        print(f"    layout_id={config.layout_id}, max_angle={config.max_angle}")
        print(f"    role_definitions={config.role_definitions}")

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Load raw data from SQLite databases
    # ─────────────────────────────────────────────────────────────────────
    # Each board has its own SQLite database containing:
    # - climbs table: route metadata (name, setter, frames string, etc.)
    # - climb_stats table: angle, difficulty, ascensionist count, quality
    # - placements table: physical hold positions and default roles
    # - holes table: (x, y) coordinates for each placement
    # - difficulty_grades table: mapping from numeric difficulty to V-grades
    #
    # The frames string is the core data — it encodes which holds are used
    # and their roles, e.g., "p344r5p369r6p603r7" means:
    #   placement 344 with role 5 (start)
    #   placement 369 with role 6 (middle)
    #   placement 603 with role 7 (finish)
    print("\nLoading data from databases...")
    if args.max_routes_per_board is not None:
        print(f"  Smoke-test limit: loading at most {args.max_routes_per_board:,} climb-angle rows per board")
    df_climbs, df_placements = load_multi_board_data(
        configs,
        project_root=REPO_ROOT,
        max_climbs_per_board=args.max_routes_per_board,
    )
    placement_lookup = make_placement_lookup(df_placements)

    print(f"  Total climb-angle entries: {len(df_climbs):,}")
    print(f"  Total placements: {len(df_placements):,}")
    print(f"  Per board:")
    for board_key in df_climbs["board_key"].unique():
        n = (df_climbs["board_key"] == board_key).sum()
        print(f"    {board_key}: {n:,} entries")

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Build tokenized route records
    # ─────────────────────────────────────────────────────────────────────
    # This is the core tokenization step. For each climb:
    # 1. Parse the frames string into (placement_id, role_id) pairs
    # 2. Map role IDs to semantic names using board config
    # 3. Sort holds canonically: starts first, then middles by Y, etc.
    # 4. Generate two token sequences:
    #    - with_grade: <BOS> <BOARD_X> <ANGLE_Y> <GRADE_VZ> <holds...> <EOS>
    #    - without_grade: <BOS> <BOARD_X> <ANGLE_Y> <holds...> <EOS>
    #
    # The grade-included version is for the GPT generator (which conditions
    # on grade). The grade-excluded version is for the BERT-style predictor
    # (which must predict grade, not see it).
    print("\nBuilding tokenized route records...")
    df_routes = build_route_records(
        df_climbs=df_climbs,
        configs_by_key=configs_by_key,
        placement_lookup=placement_lookup,
    )
    if df_routes.empty:
        raise RuntimeError("No routes were tokenized. Check raw DBs and board configs.")

    print(f"  Tokenized routes: {len(df_routes):,}")
    print(f"  Per board:")
    for board_key in df_routes["board_key"].unique():
        n = (df_routes["board_key"] == board_key).sum()
        print(f"    {board_key}: {n:,} routes")

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: Build the shared vocabulary
    # ─────────────────────────────────────────────────────────────────────
    # The vocabulary maps each unique token to an integer ID.
    # This is analogous to how GPT-2's tokenizer maps subwords to IDs.
    #
    # Vocabulary structure:
    # 1. Special tokens (IDs 0-5): <PAD>, <UNK>, <BOS>, <EOS>, <CLS>, <MASK>
    # 2. Board tokens: <BOARD_TB2>, <BOARD_KILTER>
    # 3. Angle tokens: <ANGLE_10>, <ANGLE_15>, ..., <ANGLE_55>
    # 4. Grade tokens: <GRADE_V0>, <GRADE_V1>, ..., <GRADE_V16>
    # 5. Hold tokens: <TB2_p344_start>, <KILTER_p1084_middle>, etc.
    #
    # Hold tokens are namespaced by board to prevent ID collisions.
    # TB2 placement 344 and Kilter placement 344 are different physical holds.
    print("\nBuilding vocabulary...")
    vocab_tokens, stoi, itos = build_vocab(df_routes)

    print(f"  Vocabulary size: {len(stoi):,}")
    special_count = sum(1 for t in vocab_tokens if t in ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<CLS>", "<MASK>"])
    board_count = sum(1 for t in vocab_tokens if t.startswith("<BOARD_"))
    angle_count = sum(1 for t in vocab_tokens if t.startswith("<ANGLE_"))
    grade_count = sum(1 for t in vocab_tokens if t.startswith("<GRADE_"))
    hold_count = sum(1 for t in vocab_tokens if "_p" in t)
    print(f"  Special tokens: {special_count}")
    print(f"  Board tokens: {board_count}")
    print(f"  Angle tokens: {angle_count}")
    print(f"  Grade tokens: {grade_count}")
    print(f"  Hold tokens: {hold_count}")

    # ─────────────────────────────────────────────────────────────────────
    # Step 5: Encode token sequences as integer IDs
    # ─────────────────────────────────────────────────────────────────────
    # Convert string tokens to integer IDs for model input.
    # This is the same as encoding text with a tokenizer:
    #   "The cat sat" → [464, 3797, 3290]
    #   "<BOS> <BOARD_TB2> <TB2_p344_start>" → [2, 6, 42]
    df_routes["ids_with_grade"] = df_routes["tokens_with_grade"].apply(lambda tokens: encode(tokens, stoi))
    df_routes["ids_no_grade"] = df_routes["tokens_no_grade"].apply(lambda tokens: encode(tokens, stoi))


    # ─────────────────────────────────────────────────────────────────────
    # Step 6: Train/val/test split (grouped by logical climb)
    # ─────────────────────────────────────────────────────────────────────
    # A single climb UUID can appear at multiple wall angles. We therefore
    # split by (board_key, uuid), not by individual rows. This avoids putting
    # one angle of a climb in train and another angle of the same climb in test.
    #
    # The split is stratified by board_key × grouped_v at the group level when
    # possible. The row proportions may differ slightly from 80/10/10 because
    # some climbs have more angle entries than others, but this is preferable
    # to route-level leakage or brittle UUID-overwrite logic.
    df_routes["split_stratum"] = (
        df_routes["board_key"].astype(str)
        + "__V"
        + df_routes["grouped_v"].astype(str)
    )
    df_routes["split"] = assign_group_splits(
        df_routes,
        group_cols=["board_key", "uuid"],
        test_size=0.20,
        val_size_within_temp=0.50,
        random_state=args.seed,
        stratify_col="split_stratum",
    )

    print("\nSplit counts:")
    print(df_routes.groupby(["board_key", "split"]).size().unstack(fill_value=0))

    # ─────────────────────────────────────────────────────────────────────
    # Step 7: Build token metadata
    # ─────────────────────────────────────────────────────────────────────
    # Each token has associated metadata:
    # - kind: "special", "board", "angle", "grade", or "hold"
    # - For hold tokens: board_key, placement_id, role, x, y, x_norm, y_norm
    # - For angle tokens: the angle value
    # - For grade tokens: the V-grade value
    #
    # The coordinate features (x_norm, y_norm, is_hold) are injected into
    # the grade predictor model as additional embeddings alongside token
    # embeddings. This gives the model direct spatial information.
    print("\nBuilding token metadata...")
    df_token_meta = build_token_metadata(
        vocab_tokens=vocab_tokens,
        stoi=stoi,
        df_placements=df_placements,
        placement_lookup=placement_lookup,
        configs_by_prefix=configs_by_prefix,
    )
    print(f"  Token metadata rows: {len(df_token_meta):,}")

    # ─────────────────────────────────────────────────────────────────────
    # Step 8: Save all artifacts
    # ─────────────────────────────────────────────────────────────────────
    # Save multiple file formats for different use cases:
    # - CSV: Easy to load in pandas for analysis
    # - JSONL: Easy to stream for training
    # - JSON: Vocabulary mapping for model loading
    print("\nSaving artifacts...")
    jsonl_path = args.out_dir / "routes_tokenized.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in df_routes.to_dict(orient="records"):
            handle.write(json.dumps(json_safe(record)) + "\n")

    csv_cols = [
        "uuid", "board_key", "board_display_name", "board_token_prefix", "board_token",
        "climb_name", "setter_username", "layout_id", "layout_name", "board_name",
        "frames", "angle", "display_difficulty", "grouped_v", "boulder_grade",
        "ascensionist_count", "quality_average", "fa_at",
        "n_holds", "n_start", "n_middle", "n_foot", "n_finish",
        "sequence_with_grade", "sequence_no_grade", "split",
    ]
    df_routes[csv_cols].to_csv(args.out_dir / "route_sequences.csv", index=False)
    df_placements.to_csv(args.out_dir / "placement_metadata.csv", index=False)
    df_token_meta.to_csv(args.out_dir / "token_metadata.csv", index=False)
    write_json(args.out_dir / "token_vocab.json", vocab_payload(stoi, itos, configs_by_key))

    # Board summary statistics
    board_summary = (
        df_routes.groupby("board_key")
        .agg(
            n_routes=("uuid", "count"),
            mean_angle=("angle", "mean"),
            mean_display_difficulty=("display_difficulty", "mean"),
            mean_holds=("n_holds", "mean"),
        )
        .reset_index()
    )
    board_summary.to_csv(args.out_dir / "board_summary.csv", index=False)

    print(f"\n{'='*60}")
    print(f"Tokenization complete!")
    print(f"{'='*60}")
    print(f"Boards: {board_keys}")
    print(f"Tokenized routes: {len(df_routes):,}")
    print(f"Vocabulary size: {len(stoi):,}")
    print(f"Split counts:")
    print(df_routes.groupby(["board_key", "split"]).size().unstack(fill_value=0))
    print(f"\nSaved artifacts to: {args.out_dir}")
    for f in sorted(args.out_dir.iterdir()):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
