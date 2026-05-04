from __future__ import annotations

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        frequencies = 1 / torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float) / d_model)
        pe = torch.zeros((max_len, d_model))
        pe[:, 0::2] = torch.sin(positions * frequencies)
        pe[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    d_head = q.size(-1)
    scores = q @ k.transpose(-2, -1)
    scores = scores / d_head**0.5
    scores = scores.masked_fill(mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        q_source: torch.Tensor,
        k_source: torch.Tensor,
        v_source: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = q_source.size(0)
        q = self.w_q(q_source).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(k_source).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(v_source).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        x = scaled_dot_product_attention(q, k, v, mask)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_head)
        return self.w_o(x)


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        x = x + self.dropout(self.self_attention(norm_x, norm_x, norm_x, pad_mask))
        norm_x = self.norm2(x)
        return x + self.dropout(self.ffn(norm_x))


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        encoder_output: torch.Tensor,
        decoder_input: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        norm_x = self.norm1(decoder_input)
        decoder_input = decoder_input + self.dropout(self.self_attention(norm_x, norm_x, norm_x, decoder_mask))
        norm_x = self.norm2(decoder_input)
        decoder_input = decoder_input + self.dropout(self.cross_attention(norm_x, encoder_output, encoder_output, encoder_mask))
        norm_x = self.norm3(decoder_input)
        return decoder_input + self.dropout(self.ffn(norm_x))


class TransformerTranslator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_encoders: int,
        n_decoders: int,
        d_model: int,
        n_heads: int,
        max_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.shared_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.encoders = nn.ModuleList([EncoderBlock(d_model, n_heads, dropout) for _ in range(n_encoders)])
        self.decoders = nn.ModuleList([DecoderBlock(d_model, n_heads, dropout) for _ in range(n_decoders)])
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.generator = nn.Linear(d_model, vocab_size)

    def encode(self, source_tokens: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        x = self.shared_embedding(source_tokens)
        x = self.dropout(self.positional_encoding(x))
        for encoder in self.encoders:
            x = encoder(x, source_mask)
        return self.encoder_norm(x)

    def decode(
        self,
        target_tokens: torch.Tensor,
        encoder_output: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.shared_embedding(target_tokens)
        x = self.dropout(self.positional_encoding(x))
        for decoder in self.decoders:
            x = decoder(encoder_output, x, source_mask, target_mask)
        return self.decoder_norm(x)

    def forward(
        self,
        source_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoder_output = self.encode(source_tokens, source_mask)
        decoder_output = self.decode(target_tokens, encoder_output, source_mask, target_mask)
        return self.generator(decoder_output)
