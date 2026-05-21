from __future__ import annotations

import re
from typing import Iterable

import torch
import torch.nn.functional as F

HOLD_TOKEN_PATTERN = re.compile(r"^<([A-Z0-9_]+)_p(\d+)_(start|middle|finish|foot|unknown)>$")


def top_k_filter(logits: torch.Tensor, k: int | None) -> torch.Tensor:
    if k is None or k <= 0 or k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, k)
    cutoff = values[:, [-1]]
    return torch.where(logits < cutoff, torch.full_like(logits, -float("inf")), logits)


@torch.no_grad()
def sample_ids(
    model,
    prompt_ids: list[int],
    device: torch.device,
    max_new_tokens: int = 40,
    temperature: float = 0.9,
    top_k: int | None = 50,
    eos_id: int | None = None,
    forbidden_ids: Iterable[int] | None = None,
) -> list[int]:
    model.eval()
    sequence = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    forbidden_ids = set(forbidden_ids or [])

    for _ in range(max_new_tokens):
        idx_cond = sequence[:, -model.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        for token_id in forbidden_ids:
            logits[:, int(token_id)] = -float("inf")

        logits = top_k_filter(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        sequence = torch.cat([sequence, next_id], dim=1)

        if eos_id is not None and int(next_id.item()) == int(eos_id):
            break

    return sequence[0].detach().cpu().tolist()


def prompt_tokens(board_prefix: str, angle: int, grouped_v: int) -> list[str]:
    return [
        "<BOS>",
        f"<BOARD_{board_prefix}>",
        f"<ANGLE_{int(angle)}>",
        f"<GRADE_V{int(grouped_v)}>",
    ]


def hold_records(tokens: Iterable[str]) -> list[dict[str, object]]:
    rows = []
    for token in tokens:
        match = HOLD_TOKEN_PATTERN.match(token)
        if match is None:
            continue
        rows.append(
            {
                "board_prefix": match.group(1),
                "placement_id": int(match.group(2)),
                "role": match.group(3),
                "token": token,
            }
        )
    return rows


def validity_summary(tokens: Iterable[str]) -> dict[str, object]:
    records = hold_records(tokens)
    placements = [record["placement_id"] for record in records]
    roles = [record["role"] for record in records]
    prefixes = [record["board_prefix"] for record in records]

    one_board_only = len(set(prefixes)) <= 1
    no_duplicates = len(placements) == len(set(placements))
    has_start = "start" in roles
    has_finish = "finish" in roles
    enough_holds = len(records) >= 3

    return {
        "n_hold_tokens": len(records),
        "n_unique_placements": len(set(placements)),
        "has_duplicate_placements": not no_duplicates,
        "one_board_only": one_board_only,
        "has_start": has_start,
        "has_middle": "middle" in roles,
        "has_finish": has_finish,
        "n_start": roles.count("start"),
        "n_middle": roles.count("middle"),
        "n_foot": roles.count("foot"),
        "n_finish": roles.count("finish"),
        "basic_valid": bool(one_board_only and no_duplicates and has_start and has_finish and enough_holds),
    }


def generated_tokens_to_frames(tokens: Iterable[str], role_name_to_id: dict[str, int]) -> str:
    pieces = []
    seen = set()
    for record in hold_records(tokens):
        placement_id = int(record["placement_id"])
        role = str(record["role"])
        if placement_id in seen or role not in role_name_to_id:
            continue
        seen.add(placement_id)
        pieces.append(f"p{placement_id}r{int(role_name_to_id[role])}")
    return "".join(pieces)


def generate_one(
    model,
    stoi: dict[str, int],
    itos: dict[int, str],
    device: torch.device,
    board_prefix: str,
    angle: int,
    grouped_v: int,
    role_name_to_id: dict[str, int],
    temperature: float = 0.9,
    top_k: int | None = 50,
    max_new_tokens: int = 40,
) -> dict[str, object]:
    unk_id = stoi["<UNK>"]
    eos_id = stoi["<EOS>"]
    forbidden_ids = [
        stoi["<PAD>"],
        stoi["<UNK>"],
        stoi["<BOS>"],
        stoi["<CLS>"],
        stoi["<MASK>"],
    ]

    prompt = prompt_tokens(board_prefix, angle, grouped_v)
    prompt_ids = [stoi.get(token, unk_id) for token in prompt]
    token_ids = sample_ids(
        model=model,
        prompt_ids=prompt_ids,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=eos_id,
        forbidden_ids=forbidden_ids,
    )
    tokens = [itos.get(int(idx), "<UNK>") for idx in token_ids]
    validity = validity_summary(tokens)

    return {
        "requested_board_prefix": board_prefix,
        "requested_angle": int(angle),
        "requested_grouped_v": int(grouped_v),
        "temperature": float(temperature),
        "top_k": None if top_k is None else int(top_k),
        "tokens": tokens,
        "sequence": " ".join(tokens),
        "frames": generated_tokens_to_frames(tokens, role_name_to_id),
        **validity,
    }
