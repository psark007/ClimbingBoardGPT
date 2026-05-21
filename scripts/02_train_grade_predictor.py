#!/usr/bin/env python3
"""
ClimbingBoardGPT — Grade Prediction Training Script

This script trains a BERT-style transformer encoder to predict climb difficulty
from tokenized route sequences.

Architecture Overview:
----------------------
The model is a Transformer Encoder (similar to BERT) with a regression head:

    Input: <CLS> <BOARD_TB2> <ANGLE_40> <TB2_p344_start> ... <TB2_p603_finish>
              ↓
    Token Embedding + Position Embedding + Coordinate Features
              ↓
    Transformer Encoder (4 layers, 4 heads, d_model=128)
              ↓
    <CLS> token output (pooled representation of the entire sequence)
              ↓
    MLP Head → single scalar (predicted difficulty)

Key Concepts:
1. <CLS> pooling: The <CLS> token aggregates information from the entire
   sequence via self-attention. This is the standard BERT approach for
   sequence-level tasks.

2. Coordinate features: Each hold token has physical (x, y) position
   information that gets projected and added to the embedding. This gives
   the model direct spatial knowledge without needing to learn it from data.

3. No grade token in input: The grade predictor must PREDICT the grade,
   not see it. We use the "no_grade" token sequence.

4. MSE loss: Since we're predicting a continuous value (difficulty score),
   we use Mean Squared Error loss rather than cross-entropy.

5. Joint training: Both TB2 and Kilter routes are trained together,
   with <BOARD_TB2> / <BOARD_KILTER> tokens telling the model which
   board it's operating on.

Usage:
    python scripts/02_train_grade_predictor.py
    python scripts/02_train_grade_predictor.py --epochs 100 --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from climbingboardgpt.datasets import RouteGradeDataset
from climbingboardgpt.grades import to_grouped_v
from climbingboardgpt.metrics import metrics_by_board, print_metrics, regression_metrics
from climbingboardgpt.models import JointRouteTransformerRegressor
from climbingboardgpt.utils import set_seed, write_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for grade predictor training.
    
    Key hyperparameters:
        --epochs: Maximum training epochs (default: 75)
        --patience: Early stopping patience (default: 12)
        --batch-size: Batch size for training (default: 128)
        --lr: Learning rate (default: 3e-4)
        --d-model: Transformer embedding dimension (default: 128)
        --nhead: Number of attention heads (default: 4)
        --num-layers: Number of transformer layers (default: 4)
    """
    parser = argparse.ArgumentParser(
        description="Train a joint TB2/Kilter transformer grade predictor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The model predicts display_difficulty (a continuous value) from tokenized
route sequences. Evaluation metrics include MAE, RMSE, R², and V-grade
accuracy (within ±1 V-grade).
        """,
    )
    parser.add_argument("--tokenized-dir", type=Path, default=REPO_ROOT / "data" / "processed" / "tokenized")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "processed" / "grade_prediction")
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--epochs", type=int, default=75, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--d-model", type=int, default=128, help="Transformer embedding dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--dim-feedforward", type=int, default=256, help="Feedforward dimension")
    parser.add_argument("--dropout", type=float, default=0.10, help="Dropout probability")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    return parser.parse_args()


def build_coord_features(df_token_meta: pd.DataFrame, vocab_size: int) -> torch.Tensor:
    """Build coordinate feature matrix for the transformer model.
    
    Each token gets a 3-dimensional feature vector:
    - x_norm: Normalized horizontal position on the board (-1 to 1)
    - y_norm: Normalized vertical position on the board (-1 to 1)
    - is_hold: 1 if this token represents a hold, 0 otherwise
    
    These features are projected through a linear layer and added to
    the token embeddings, giving the model direct spatial information.
    This is analogous to how some vision-language models inject spatial
    features from images alongside text tokens.
    
    Args:
        df_token_meta: DataFrame with token metadata
        vocab_size: Total vocabulary size
        
    Returns:
        Tensor of shape (vocab_size, 3) with coordinate features
    """
    features = np.zeros((vocab_size, 3), dtype=np.float32)
    for _, row in df_token_meta.iterrows():
        token_id = int(row["token_id"])
        features[token_id, 0] = 0.0 if pd.isna(row.get("x_norm", 0.0)) else float(row.get("x_norm", 0.0))
        features[token_id, 1] = 0.0 if pd.isna(row.get("y_norm", 0.0)) else float(row.get("y_norm", 0.0))
        features[token_id, 2] = 0.0 if pd.isna(row.get("is_hold", 0.0)) else float(row.get("is_hold", 0.0))
    return torch.tensor(features, dtype=torch.float32)


def run_epoch(model, loader, device, optimizer=None):
    """Run one epoch of training or evaluation.
    
    Args:
        model: The transformer model
        loader: DataLoader for this epoch
        device: torch device (cpu or cuda)
        optimizer: If provided, run training (with gradient updates).
                   If None, run evaluation (no gradient updates).
    
    Returns:
        Tuple of (average_loss, predictions, targets, uuids, board_keys)
    """
    is_train = optimizer is not None
    model.train(is_train)
    criterion = nn.MSELoss()

    losses, preds, targets, uuids, boards = [], [], [], [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target = batch["target"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        # Forward pass: model predicts difficulty from token sequence
        pred = model(input_ids, attention_mask)
        loss = criterion(pred, target)

        if is_train:
            # Backward pass: compute gradients and update weights
            loss.backward()
            # Gradient clipping prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        losses.append(loss.item() * input_ids.size(0))
        preds.extend(pred.detach().cpu().numpy().tolist())
        targets.extend(target.detach().cpu().numpy().tolist())
        uuids.extend(batch["uuid"])
        boards.extend(batch["board_key"])

    avg_loss = sum(losses) / max(1, len(loader.dataset))
    return avg_loss, np.asarray(preds), np.asarray(targets), uuids, boards


def main() -> None:
    """Main training loop for the grade predictor.
    
    Steps:
    1. Load tokenized data and vocabulary
    2. Prepare input sequences (with <CLS> token, without grade)
    3. Build coordinate features matrix
    4. Create train/val/test DataLoaders
    5. Initialize transformer model
    6. Train with early stopping
    7. Evaluate on test set
    8. Save model checkpoint and metrics
    """
    args = parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Load tokenized data
    # ─────────────────────────────────────────────────────────────────────
    seq_path = args.tokenized_dir / "route_sequences.csv"
    vocab_path = args.tokenized_dir / "token_vocab.json"
    meta_path = args.tokenized_dir / "token_metadata.csv"
    if not seq_path.exists() or not vocab_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Missing tokenized artifacts. Run scripts/01_tokenize_routes.py first.")

    df_routes = pd.read_csv(seq_path)
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    stoi = {str(k): int(v) for k, v in vocab["stoi"].items()}
    itos = {int(k): str(v) for k, v in vocab["itos"].items()}
    df_token_meta = pd.read_csv(meta_path)

    pad_id = stoi["<PAD>"]
    unk_id = stoi["<UNK>"]

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Prepare input sequences
    # ─────────────────────────────────────────────────────────────────────
    # For grade prediction, we use the "no_grade" version of the sequence
    # and prepend <CLS> for sequence-level pooling.
    # The model must PREDICT the grade, not see it in the input!
    def encode(tokens):
        return [stoi.get(token, unk_id) for token in tokens]

    df_routes["tokens_no_grade"] = df_routes["sequence_no_grade"].fillna("").str.split()
    df_routes["model_tokens"] = df_routes["tokens_no_grade"].apply(
        lambda tokens: ["<CLS>"] + tokens[1:] if tokens else ["<CLS>"]
    )
    df_routes["model_ids"] = df_routes["model_tokens"].apply(encode)
    df_routes["seq_len"] = df_routes["model_ids"].apply(len)
    max_len = int(df_routes["seq_len"].max())

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Create DataLoaders
    # ─────────────────────────────────────────────────────────────────────
    train_df = df_routes[df_routes["split"] == "train"].reset_index(drop=True)
    val_df = df_routes[df_routes["split"] == "val"].reset_index(drop=True)
    test_df = df_routes[df_routes["split"] == "test"].reset_index(drop=True)

    train_ds = RouteGradeDataset(train_df, max_len=max_len, pad_id=pad_id)
    val_ds = RouteGradeDataset(val_df, max_len=max_len, pad_id=pad_id)
    test_ds = RouteGradeDataset(test_df, max_len=max_len, pad_id=pad_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: Initialize model
    # ─────────────────────────────────────────────────────────────────────
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    coord_features = build_coord_features(df_token_meta, vocab_size=len(stoi))

    model = JointRouteTransformerRegressor(
        vocab_size=len(stoi),
        max_len=max_len,
        coord_features=coord_features,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pad_id=pad_id,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Device: {device}")
    print(f"Train/val/test: {len(train_ds):,}, {len(val_ds):,}, {len(test_ds):,}")
    print(f"Vocabulary size: {len(stoi):,}")
    print(f"Max sequence length: {max_len}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ─────────────────────────────────────────────────────────────────────
    # Step 5: Training loop with early stopping
    # ─────────────────────────────────────────────────────────────────────
    history = []
    best_val_mae = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_pred, train_true, _, _ = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_pred, val_true, _, _ = run_epoch(model, val_loader, device, optimizer=None)

        train_metrics = regression_metrics(train_true, train_pred)
        val_metrics = regression_metrics(val_true, val_pred)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "val_r2": val_metrics["r2"],
            "val_within_1_vgrade": val_metrics["within_1_vgrade"],
        })

        # Track best model by validation MAE
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == best_epoch:
            print(
                f"Epoch {epoch:03d} | "
                f"train MAE {train_metrics['mae']:.3f} | "
                f"val MAE {val_metrics['mae']:.3f} | "
                f"val R² {val_metrics['r2']:.3f} | "
                f"val ±1V {val_metrics['within_1_vgrade']:.1f}%"
            )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # ─────────────────────────────────────────────────────────────────────
    # Step 6: Test set evaluation
    # ─────────────────────────────────────────────────────────────────────
    test_loss, test_pred, test_true, test_uuid, test_board = run_epoch(model, test_loader, device, optimizer=None)
    overall_metrics = regression_metrics(test_true, test_pred)

    pred_df = pd.DataFrame({
        "uuid": test_uuid,
        "board_key": test_board,
        "y_true": test_true,
        "y_pred": test_pred,
        "abs_error": np.abs(test_true - test_pred),
        "true_v": [to_grouped_v(value) for value in test_true],
        "pred_v": [to_grouped_v(value) for value in test_pred],
    })
    pred_df = pred_df.merge(
        df_routes[["uuid", "climb_name", "angle", "boulder_grade", "sequence_no_grade"]],
        on="uuid",
        how="left",
    )
    board_metrics_df = metrics_by_board(pred_df)

    print_metrics("Overall joint test performance", overall_metrics)
    print("\nBoard-specific test performance:")
    print(board_metrics_df.to_string(index=False))

    # ─────────────────────────────────────────────────────────────────────
    # Step 7: Save artifacts
    # ─────────────────────────────────────────────────────────────────────
    pd.DataFrame(history).to_csv(args.out_dir / "training_history.csv", index=False)
    pred_df.to_csv(args.out_dir / "test_predictions.csv", index=False)
    board_metrics_df.to_csv(args.out_dir / "board_metrics.csv", index=False)
    write_json(args.out_dir / "overall_metrics.json", overall_metrics)

    # Save model checkpoint with all necessary info for loading
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": len(stoi),
            "max_len": max_len,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
            "pad_id": pad_id,
        },
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
        "coord_features": coord_features.cpu(),
        "overall_metrics": overall_metrics,
    }
    model_path = args.model_dir / "joint_transformer_grade_predictor.pth"
    torch.save(checkpoint, model_path)

    print("\nSaved:")
    print(f"  {args.out_dir}")
    print(f"  {model_path}")


if __name__ == "__main__":
    main()