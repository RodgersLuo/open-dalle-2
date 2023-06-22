"""
Adapted from: https://github.com/openai/CLIP/blob/main/clip/model.py
"""


from collections import OrderedDict

import torch
from torch import nn
from einops import rearrange


class GELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def forward(self, x: torch.Tensor):
        # Multi-head attention
        h = self.ln_1(x)
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=h.dtype, device=h.device)
        h = self.attn(h, h, h, need_weights=False, attn_mask=self.attn_mask)[0]

        # Residual connection 1
        x = x + h

        # Norm and Feedforward network
        h = self.mlp(self.ln_2(x))
        # Residual connection 2
        x = x + h
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, vocab_size: int, context_length: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

        self.token_embedding = None
        if vocab_size is not None:
            self.vocab_size = vocab_size
            self.token_embedding = nn.Embedding(vocab_size, width)

        self.positional_embedding = nn.Parameter(torch.randn(context_length, width))
        self.ln_final = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor):
        if self.token_embedding is not None:
            x = self.token_embedding(x)

        x = x + self.positional_embedding
        x = rearrange(x, 'b n d -> n b d')
        x = self.resblocks(x)
        x = rearrange(x, 'n b d -> b n d')
        x = self.ln_final(x)
        return x
