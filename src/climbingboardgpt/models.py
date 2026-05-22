from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointRouteTransformerRegressor(nn.Module):
    """Transformer encoder for joint TB2/Kilter route difficulty prediction."""

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        coord_features: torch.Tensor,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.10,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.pad_id = pad_id

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.register_buffer("coord_features", coord_features.clone().float())
        self.coord_proj = nn.Linear(coord_features.shape[1], d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = x + self.coord_proj(self.coord_features[input_ids])

        key_padding_mask = ~attention_mask.bool()
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)

        cls_state = h[:, 0, :]
        return self.head(cls_state).squeeze(-1)


class JointRouteGPT(nn.Module):
    """Tiny GPT-style causal transformer for board-conditioned route generation."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        dropout: float = 0.10,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.pad_id = pad_id

        self.token_emb = nn.Embedding(vocab_size, n_embd, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer,
            num_layers=n_layer,
            enable_nested_tensor=False,
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.shape
        if seq_len > self.block_size:
            idx = idx[:, -self.block_size :]
            seq_len = idx.shape[1]

        positions = torch.arange(seq_len, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(positions))

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=idx.device, dtype=torch.bool),
            diagonal=1,
        )
        key_padding_mask = idx.eq(self.pad_id)

        h = self.blocks(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        h = self.ln_f(h)
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.pad_id,
            )

        return logits, loss
