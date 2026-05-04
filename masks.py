from __future__ import annotations

import torch


def make_causal_mask(tokens: torch.Tensor) -> torch.Tensor:
    length = tokens.size(1)
    mask = torch.triu(torch.ones((length, length), device=tokens.device), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(1)


def make_pad_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    return (tokens == pad_id).unsqueeze(1).unsqueeze(2)
