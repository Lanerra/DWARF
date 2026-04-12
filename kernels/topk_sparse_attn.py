"""
TopK Sparse Attention — content-addressed lookup over the full causal prefix.

Scores all N past positions, keeps top-k, softmaxes over the survivors.
O(N²) scoring — intentional for mechanism validation, not production.
Gradients flow through all selected positions (soft threshold, not hard argmax).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSparseAttention(nn.Module):

    def __init__(self, D: int, H: int, hd: int, k: int = 64):
        super().__init__()
        self.H, self.hd, self.k = H, hd, k
        self.W_q = nn.Linear(D, H * hd, bias=False)
        self.W_k = nn.Linear(D, H * hd, bias=False)
        self.W_v = nn.Linear(D, H * hd, bias=False)
        self.W_o = nn.Linear(H * hd, D, bias=False)

    def forward(self, x: torch.Tensor, kv_inject=None) -> torch.Tensor:
        B, N, D = x.shape
        H, hd = self.H, self.hd

        def reshape(t):
            return t.reshape(B, N, H, hd).transpose(1, 2)

        Q = reshape(self.W_q(x))
        K = reshape(self.W_k(x))
        V = reshape(self.W_v(x))

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(hd)

        causal_mask = torch.triu(
            torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask[None, None], float('-inf'))

        if self.k < N:
            topk_vals, _ = scores.topk(self.k, dim=-1)
            threshold = topk_vals[..., -1:]
            scores = scores.masked_fill(scores < threshold, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, N, H * hd)
        return self.W_o(out)
