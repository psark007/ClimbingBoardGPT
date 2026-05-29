"""PyTorch dataset adapters for tokenized climbing-board routes."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class RouteGradeDataset(Dataset):
    """Dataset for transformer encoder grade prediction.

    Each item returns a padded token sequence, a boolean attention mask, the
    continuous display-difficulty target, and a small amount of route identity
    metadata used when writing prediction CSVs.
    """

    def __init__(self, df, max_len: int, pad_id: int):
        """Store model IDs and labels from a tokenized route DataFrame."""
        self.row_ids = df["row_id"].tolist() if "row_id" in df.columns else df.index.tolist()
        self.ids = df["model_ids"].tolist()
        self.targets = df["display_difficulty"].astype(float).values
        self.uuids = df["uuid"].tolist()
        self.boards = df["board_key"].astype(str).tolist()
        self.max_len = int(max_len)
        self.pad_id = int(pad_id)

    def __len__(self) -> int:
        """Return the number of route examples."""
        return len(self.ids)

    def __getitem__(self, idx: int):
        """Return one padded encoder example and its regression target."""
        ids = list(self.ids[idx])[: self.max_len]
        mask = [1] * len(ids)
        if len(ids) < self.max_len:
            pad_n = self.max_len - len(ids)
            ids += [self.pad_id] * pad_n
            mask += [0] * pad_n

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.bool),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
            "row_id": int(self.row_ids[idx]),
            "uuid": self.uuids[idx],
            "board_key": self.boards[idx],
        }


class RouteGPTDataset(Dataset):
    """Dataset for causal next-token route generation.

    The full sequence is padded once, then split into ``input_ids`` and
    ``target_ids`` shifted by one position for teacher-forced language-model
    training.
    """

    def __init__(self, df, max_len: int, pad_id: int):
        """Store GPT token ID sequences from a tokenized route DataFrame."""
        self.ids = df["gpt_ids"].tolist()
        self.max_len = int(max_len)
        self.pad_id = int(pad_id)

    def __len__(self) -> int:
        """Return the number of route examples."""
        return len(self.ids)

    def __getitem__(self, idx: int):
        """Return one padded causal-language-model training example."""
        ids = list(self.ids[idx])[: self.max_len]
        if len(ids) < self.max_len:
            ids += [self.pad_id] * (self.max_len - len(ids))

        return {
            "input_ids": torch.tensor(ids[:-1], dtype=torch.long),
            "target_ids": torch.tensor(ids[1:], dtype=torch.long),
        }
