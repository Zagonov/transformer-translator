from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerFast

from .masks import make_causal_mask, make_pad_mask
from .model import TransformerTranslator


@torch.inference_mode()
def translate(
    texts: list[str],
    model: TransformerTranslator,
    tokenizer: PreTrainedTokenizerFast,
    max_len: int,
    device: str | torch.device,
) -> list[str]:
    model.eval()
    model.to(device)
    tokenized = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_len).to(device)
    source_tokens = tokenized.input_ids
    source_mask = (tokenized.attention_mask == 0).unsqueeze(1).unsqueeze(2)
    encoder_output = model.encode(source_tokens, source_mask)

    batch_size = len(texts)
    target_tokens = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    pad_token = torch.tensor(tokenizer.pad_token_id, device=device)

    for _ in range(max_len - 1):
        target_mask = make_pad_mask(target_tokens, tokenizer.pad_token_id) | make_causal_mask(target_tokens)
        decoder_output = model.decode(target_tokens, encoder_output, source_mask, target_mask)
        next_tokens = model.generator(decoder_output[:, -1, :]).argmax(dim=-1)
        next_tokens = torch.where(finished, pad_token, next_tokens)
        target_tokens = torch.cat([target_tokens, next_tokens.unsqueeze(1)], dim=1)
        finished = finished | (next_tokens == tokenizer.eos_token_id)

    return tokenizer.batch_decode(target_tokens, skip_special_tokens=True)
