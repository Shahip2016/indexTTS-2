import torch
import torch.nn as nn
import math

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        return x + attn_out

class ConditioningEncoder(nn.Module):
    def __init__(self, spec_dim, embedding_dim, attn_blocks=6, num_attn_heads=4, mean=False):
        super().__init__()
        self.init = nn.Linear(spec_dim, embedding_dim)
        self.attn = nn.Sequential(*[
            AttentionBlock(embedding_dim, num_attn_heads) for _ in range(attn_blocks)
        ])
        self.mean = mean

    def forward(self, x):
        # x: (B, T, spec_dim)
        h = self.init(x)
        h = self.attn(h)
        if self.mean:
            return h.mean(dim=1, keepdim=True)
        else:
            return h

class PerceiverResampler(nn.Module):
    def __init__(self, dim, dim_context=None, num_latents=32, heads=8, dim_head=64, ff_mult=4):
        super().__init__()
        dim_context = dim if dim_context is None else dim_context
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim_context)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim)
        )

    def forward(self, x, mask=None):
        # x: (B, T, dim_context)
        B = x.shape[0]
        latents = self.latents.repeat(B, 1, 1)
        
        # Cross attention
        # simplified for brevity
        # attn_out, _ = self.attn(latents, x, x)
        # latents = latents + attn_out
        
        # Self attention layers...
        
        return latents
