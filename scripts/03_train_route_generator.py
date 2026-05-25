#!/usr/bin/env python3
"""
ClimbingBoardGPT — Route Generation Training Script

This script trains a GPT-style causal transformer to generate new climbing
routes conditioned on board type, angle, and target grade.

Architecture Overview:
----------------------
The model is a causal (autoregressive) transformer decoder:

    Input: <BOS> <BOARD_TB2> <ANGLE_40> <GRADE_V6> <TB2_p344_start> ...
              ↓
    Token Embedding + Position Embedding
              ↓
    Causal Transformer (4 layers, 4 heads, d_embd=128)
    [Each position can only attend to previous positions]
              ↓
    Language Modeling Head → next token logits
              ↓
    Sample next token → append to sequence → repeat

Key Concepts:
1. Causal masking: Unlike BERT which sees all tokens, GPT can only
   attend to previous tokens. This enables autoregressive generation.

2. Teacher forcing: During training, we feed the ground-truth previous
   token. During generation, we feed the model's own prediction.

3. Weight tying: The output projection shares weights with the input
   embedding. This reduces parameters and improves training stability.

4. Temperature & top-k sampling: Control generation diversity.
   - Low temperature (0.3) → conservative, realistic routes
   - High temperature (1.5) → creative, unusual routes
   - Top-k (default 50) → only consider the 50 most likely next tokens

5. Conditioning: The prompt tokens (<BOARD_...>, <ANGLE_...>, <GRADE_...>)
   tell the model what kind of route to generate, similar to how
   ChatGPT uses system prompts.

Usage:
    python scripts/03_train_route_generator.py
    python scripts/03_train_route_generator.py --epochs 100 --temperature 0.7
    python scripts/03_train_route_generator.py --generate-board tb2 --generate-grades 3,5,7
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from climbingboardgpt.config import load_board_configs, parse_board_keys
from climbingboardgpt.datasets import RouteGPTDataset
from climbingboardgpt.generation import generate_one
from climbingboardgpt.models import JointRouteGPT
from climbingboardgpt.tokenization import encode as encode_tokens
from climbingboardgpt.utils import set_seed


def csv_ints(value: str | None) -> list[int] | None:
    """Parse a comma-separated string of integers, or return None."""
    if value is None or not value.strip():
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for route generator training."""
    parser = argparse.ArgumentParser(
        description="Train a joint TB2/Kilter GPT-style route generator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
After training, the script generates sample routes for each board at
common angles and grades. Use --generate-board to generate for a
specific board, or leave unset to generate for all boards.
        """,
    )
    parser.add_argument("--tokenized-dir", type=Path, default=REPO_ROOT / "data" / "processed" / "tokenized")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "processed" / "generation")
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--boards", type=str, default="tb2,kilter", help="Board configs for role reconstruction")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--n-embd", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n-head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layer", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.10, help="Dropout probability")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--max-new-tokens", type=int, default=40, help="Max tokens to generate")
    parser.add_argument("--n-per-condition", type=int, default=10, help="Routes to generate per condition")
    parser.add_argument("--generate-board", type=str, default=None, help="Board key: tb2 or kilter")
    parser.add_argument("--generate-angles", type=str, default=None, help="Comma-separated angles")
    parser.add_argument("--generate-grades", type=str, default=None, help="Comma-separated V-grades")
    parser.add_argument("--seed", type=int, default=3, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a tiny CPU model, one epoch, and a tiny generation grid to exercise the full code path.",
    )
    return parser.parse_args()


def apply_smoke_test_defaults(args: argparse.Namespace) -> None:
    """Mutate args to a tiny deterministic configuration for code-path checks."""
    if not args.smoke_test:
        return
    args.epochs = 1
    args.patience = 1
    args.batch_size = min(args.batch_size, 16)
    args.n_embd = 32
    args.n_head = 2
    args.n_layer = 1
    args.dropout = 0.0
    args.max_new_tokens = min(args.max_new_tokens, 16)
    args.n_per_condition = 1
    args.device = "cpu"
    args.num_workers = 0


def evaluate_loss(model, loader, device) -> float:
    """Evaluate the model on a data loader, returning average loss.
    
    This is used for validation and test evaluation. The model is set to
    eval mode and no gradients are computed.
    """
    model.eval()
    losses = []
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["input_ids"].to(device)
            y = batch["target_ids"].to(device)
            _, loss = model(x, y)
            batch_size = x.size(0)
            losses.append(loss.item() * batch_size)
            n += batch_size
    return sum(losses) / max(1, n)


def train_one_epoch(model, loader, optimizer, device) -> float:
    """Train for one epoch, returning average loss.
    
    Uses teacher forcing: the model receives ground-truth previous tokens
    and predicts the next token. This is standard for language model training.
    """
    model.train()
    losses = []
    n = 0
    for batch in loader:
        x = batch["input_ids"].to(device)
        y = batch["target_ids"].to(device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        # Gradient clipping prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_size = x.size(0)
        losses.append(loss.item() * batch_size)
        n += batch_size
    return sum(losses) / max(1, n)


def main() -> None:
    """Main training and generation loop.
    
    Steps:
    1. Load tokenized data and vocabulary
    2. Prepare input/target pairs for causal language modeling
    3. Create train/val DataLoaders
    4. Initialize GPT model
    5. Train with early stopping
    6. Generate sample routes for evaluation
    7. Save model checkpoint and generated routes
    """
    args = parse_args()
    apply_smoke_test_defaults(args)
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Load data
    # ─────────────────────────────────────────────────────────────────────
    seq_path = args.tokenized_dir / "route_sequences.csv"
    vocab_path = args.tokenized_dir / "token_vocab.json"
    if not seq_path.exists() or not vocab_path.exists():
        raise FileNotFoundError("Missing tokenized artifacts. Run scripts/01_tokenize_routes.py first.")

    df_routes = pd.read_csv(seq_path)
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    stoi = {str(k): int(v) for k, v in vocab["stoi"].items()}
    itos = {int(k): str(v) for k, v in vocab["itos"].items()}
    pad_id = stoi["<PAD>"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Prepare sequences for causal language modeling
    # ─────────────────────────────────────────────────────────────────────
    # For GPT training, we use the "with grade" version because the model
    # needs to learn the relationship between grade and hold selection.
    #
    # Input:  <BOS> <BOARD_TB2> <ANGLE_40> <GRADE_V6> <TB2_p344_start> ...
    # Target: <BOARD_TB2> <ANGLE_40> <GRADE_V6> <TB2_p344_start> <TB2_p369_middle> ...
    #
    # The input is shifted right by one position compared to the target.
    # This is the standard causal language modeling setup.
    df_routes["gpt_tokens"] = df_routes["sequence_with_grade"].fillna("").str.split()
    df_routes["gpt_ids"] = df_routes["gpt_tokens"].apply(lambda tokens: encode_tokens(tokens, stoi))
    df_routes["seq_len"] = df_routes["gpt_ids"].apply(len)
    max_len = int(df_routes["seq_len"].max())
    if max_len < 2:
        raise RuntimeError("Token sequences are too short to train the causal model.")
    block_size = max_len - 1  # Input length (one less than full sequence)

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Create DataLoaders
    # ─────────────────────────────────────────────────────────────────────
    train_df = df_routes[df_routes["split"] == "train"].reset_index(drop=True)
    val_df = df_routes[df_routes["split"] == "val"].reset_index(drop=True)
    test_df = df_routes[df_routes["split"] == "test"].reset_index(drop=True)

    train_ds = RouteGPTDataset(train_df, max_len=max_len, pad_id=pad_id)
    val_ds = RouteGPTDataset(val_df, max_len=max_len, pad_id=pad_id)
    test_ds = RouteGPTDataset(test_df, max_len=max_len, pad_id=pad_id)

    loader_kwargs = {
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: Initialize model
    # ─────────────────────────────────────────────────────────────────────
    model = JointRouteGPT(
        vocab_size=len(stoi),
        block_size=block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        pad_id=pad_id,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Device: {device}")
    print(f"Train/val/test: {len(train_ds):,}, {len(val_ds):,}, {len(test_ds):,}")
    print(f"Vocabulary size: {len(stoi):,}")
    print(f"Block size: {block_size}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ─────────────────────────────────────────────────────────────────────
    # Step 5: Training loop with early stopping
    # ─────────────────────────────────────────────────────────────────────
    # We track perplexity (exp(loss)) as well as raw loss.
    # Perplexity answers: "On average, how many tokens was the model
    # choosing between at each step?"
    # Lower perplexity = better model.
    history = []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    print("\nStarting GPT training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_perplexity": math.exp(min(train_loss, 20)),
            "val_perplexity": math.exp(min(val_loss, 20)),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == best_epoch:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss {train_loss:.3f} | "
                f"val loss {val_loss:.3f} | "
                f"val ppl {math.exp(min(val_loss, 20)):.1f}"
            )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ─────────────────────────────────────────────────────────────────────
    # Step 6: Test evaluation
    # ─────────────────────────────────────────────────────────────────────
    test_loss = evaluate_loss(model, test_loader, device)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test perplexity: {math.exp(min(test_loss, 20)):.1f}")

    # ─────────────────────────────────────────────────────────────────────
    # Step 7: Generate sample routes
    # ─────────────────────────────────────────────────────────────────────
    # For each board, generate routes at common angles and grades.
    # This demonstrates the model's ability to produce novel routes
    # conditioned on board, angle, and difficulty.
    configs = load_board_configs(parse_board_keys(args.boards))
    configs_by_key = {config.board_key: config for config in configs}

    board_keys_to_generate = [args.generate_board] if args.generate_board else sorted(df_routes["board_key"].unique())
    requested_angles = csv_ints(args.generate_angles)
    requested_grades = csv_ints(args.generate_grades)

    generated = []
    for board_key in board_keys_to_generate:
        board_frame = df_routes[df_routes["board_key"] == board_key]
        if board_frame.empty:
            continue
        config = configs_by_key[board_key]
        # Use common angles if none specified
        angles = requested_angles or (
            board_frame["angle"].astype(int).value_counts().head(5).index.sort_values().tolist()
        )
        # Use common grades if none specified
        grades = requested_grades or (
            board_frame["grouped_v"].astype(int).value_counts().head(8).index.sort_values().tolist()
        )
        for angle in angles:
            for grade in grades:
                for _ in range(args.n_per_condition):
                    generated.append({
                        "board_key": board_key,
                        **generate_one(
                            model=model,
                            stoi=stoi,
                            itos=itos,
                            device=device,
                            board_prefix=config.token_prefix,
                            angle=int(angle),
                            grouped_v=int(grade),
                            role_name_to_id=config.role_definitions,
                            temperature=args.temperature,
                            top_k=args.top_k,
                            max_new_tokens=args.max_new_tokens,
                        ),
                    })

    generated_df = pd.DataFrame(generated)
    if not generated_df.empty:
        print(f"\nGenerated routes: {len(generated_df):,}")
        print("Basic validity by board:")
        print(generated_df.groupby("board_key")["basic_valid"].mean())

    # ─────────────────────────────────────────────────────────────────────
    # Step 8: Save artifacts
    # ─────────────────────────────────────────────────────────────────────
    pd.DataFrame(history).to_csv(args.out_dir / "training_history.csv", index=False)
    generated_df.to_csv(args.out_dir / "generated_routes.csv", index=False)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": len(stoi),
            "block_size": block_size,
            "n_embd": args.n_embd,
            "n_head": args.n_head,
            "n_layer": args.n_layer,
            "dropout": args.dropout,
            "pad_id": pad_id,
        },
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
    }
    model_path = args.model_dir / "joint_route_gpt_generator.pth"
    torch.save(checkpoint, model_path)

    print("\nSaved:")
    print(f"  {args.out_dir}")
    print(f"  {model_path}")


if __name__ == "__main__":
    main()
