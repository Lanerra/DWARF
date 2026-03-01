#!/usr/bin/env python3
"""
Unified Evaluation Suite for DWARF ablation models.

Runs four evaluations on any registered model:
  1. Calibration        — logit entropy, top-1 confidence, effective vocab size
  2. Distance-conditioned loss — per-position CE loss bucketed by lookback distance
                                  (wikitext-103, 17 full 2048-token chunks)
  3. Few-shot string copy — template-following / induction-head capability
  4. Passkey retrieval  — content-addressed memory at varying distances

Registered models:
  standard_13m   D=256 standard transformer  (untied emb)  13M.pt
  standard_85m   D=640 standard transformer  (untied emb)  best.pt
  standard_27m   D=400 standard transformer  (untied emb)  27M.ptrom
  condp_13m      D=256 condP pure DSQG       (tied emb)    best.pt
  condp_27m      D=400 condP pure DSQG       (tied emb)    best.ptrom
  condm_layer0   D=256 condM full_attn=0     (tied emb)    best.pt
  condm_layer3   D=256 condM full_attn=3     (tied emb)    best.pt
  condm_layer5   D=256 condM full_attn=5     (tied emb)    13M.pt
  condm_27m      D=400 condM full_attn=5     (tied emb)    27M.ptrom
  condm_v2       D=256 condM-v2 (SwiGLU+RMSNorm+RoPE) full_attn=5  best.pt
  condm_85m      D=640 condM full_attn=11    (tied emb)    best.pt
  condm_periodic_13m  D=224 condM-periodic [3:1:3:1] full_attn=[3,7]  best.pt

Usage:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 benchmarks/eval_suite.py --model standard_27m
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 benchmarks/eval_suite.py --model condm_layer0

Results: benchmarks/logs/eval_suite_<model>_<timestamp>.json
"""

import argparse, json, math, os, sys, time, datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

# CondM-v2 architecture lives in the training script; import lazily.
def _import_condm_v2():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from train_2048_condM_v2 import CondMV2Transformer
    return CondMV2Transformer

def _import_condm_periodic():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from train_2048_condM_periodic_13m import CondMPeriodicTransformer
    return CondMPeriodicTransformer

# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
TOKENIZER   = os.path.join(SCRIPT_DIR, 'results', '2048_condI_tokenizer.json')
CKPT_ROOT   = os.path.join(REPO_ROOT, 'checkpoints')
LOGS_DIR    = os.path.join(SCRIPT_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

MAX_SEQ_LEN = 2048
VOCAB_SIZE  = 32000

# ─── Model registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    'standard_13m': {
        'arch':       'standard',
        'D':          256, 'H': 8, 'FFN': 1024, 'L': 6,
        'checkpoint': os.path.join(CKPT_ROOT, '2048_standard_baseline_checkpoints', '13M.pt'),
        'label':      'Standard Transformer 13M',
        'params_ref': 21_631_008,
    },
    'standard_85m': {
        'arch':       'standard',
        'D':          640, 'H': 8, 'FFN': 2560, 'L': 12,
        'checkpoint': os.path.join(CKPT_ROOT, '85m_standard_baseline', 'best.pt'),
        'label':      'Standard Transformer 85M',
        'params_ref': 101_361_920,
    },
    'standard_27m': {
        'arch':       'standard',
        'D':          400, 'H': 8, 'FFN': 1600, 'L': 6,
        'checkpoint': os.path.join(CKPT_ROOT, '2048_standard_baseline_checkpoints', '27M.ptrom'),
        'label':      'Standard Transformer 27M',
        'params_ref': 37_971_200,
    },
    'condp_13m': {
        'arch':       'condp',
        'D':          256, 'H': 8, 'FFN': 1024, 'L': 6,
        'checkpoint': os.path.join(CKPT_ROOT, '2048_condP_checkpoints', 'best.pt'),
        'label':      'condP 13M (pure DSQG, 74 offsets)',
        'params_ref': None,
    },
    'condp_27m': {
        'arch':       'condp',
        'D':          400, 'H': 8, 'FFN': 1600, 'L': 6,
        'checkpoint': os.path.join(CKPT_ROOT, '27m_2048__condP_checkpoints', 'best.ptrom'),
        'label':      'condP 27M (pure DSQG, 74 offsets)',
        'params_ref': 26_781_152,
    },
    'condm_layer0': {
        'arch':       'condm',
        'D':          256, 'H': 8, 'FFN': 1024, 'L': 6, 'full_layer': 0,
        'checkpoint': os.path.join(SCRIPT_DIR, '2048_condM_layer0_checkpoints', 'best.pt'),
        'label':      'condM 13M Layer 0 (1 full attn → 5 DSQG)',
        'params_ref': 14_116_576,
    },
    'condm_layer3': {
        'arch':       'condm',
        'D':          256, 'H': 8, 'FFN': 1024, 'L': 6, 'full_layer': 3,
        'checkpoint': os.path.join(SCRIPT_DIR, '2048_condM_layer3_checkpoints', 'best.pt'),
        'label':      'condM 13M Layer 3 (3 DSQG → 1 full attn → 2 DSQG)',
        'params_ref': 14_116_576,
    },
    'condm_layer5': {
        'arch':       'condm',
        'D':          256, 'H': 8, 'FFN': 1024, 'L': 6, 'full_layer': 5,
        'checkpoint': os.path.join(CKPT_ROOT, '2048_condM_checkpoints', '13M.pt'),
        'label':      'condM 13M Layer 5 (5 DSQG → 1 full attn)',
        'params_ref': 13_984_480,
    },
    'condm_chinchilla_repeated': {
        'arch':       'condm',
        'D':          256, 'H': 8, 'FFN': 1024, 'L': 6, 'full_layer': 5,
        'checkpoint': os.path.join(CKPT_ROOT, 'condm_chinchilla_repeated', 'best.pt'),
        'label':      'condM 13M Chinchilla-Repeated (100K docs x 10 epochs)',
        'params_ref': 13_917_664,
    },
    'condm_27m': {
        'arch':       'condm',
        'D':          400, 'H': 8, 'FFN': 1600, 'L': 6, 'full_layer': 5,
        'checkpoint': os.path.join(CKPT_ROOT, '2048_condM_checkpoints', '27M.ptrom'),
        'label':      'condM 27M Layer 5 (5 DSQG → 1 full attn)',
        'params_ref': 26_457_760,
    },
    'condm_v2': {
        'arch':       'condm_v2',
        'D':          256, 'H': 8, 'FFN': 1024, 'L': 6, 'full_layer': 5,
        'checkpoint': os.path.join(CKPT_ROOT, '2048_condM_v2_checkpoints', 'best.pt'),
        'label':      'condM-v2 13M (SwiGLU+RMSNorm+RoPE, Layer 5)',
        'params_ref': None,
    },
    'condm_85m': {
        'arch':       'condm',
        'D':          640, 'H': 8, 'FFN': 2560, 'L': 12, 'full_layer': 11,
        'interference': 3,
        'checkpoint': os.path.join(CKPT_ROOT, '2048_condM_85m_checkpoints', 'best.ptrom'),
        'label':      'condM 85M (11 DSQG + 1 full attn, Layer 11)',
        'params_ref': 88_267_552,
    },
    'condm_chinchilla_13m': {
        'arch':       'condm',
        'D':          256, 'H': 8, 'FFN': 1024, 'L': 6, 'full_layer': 5,
        'checkpoint': os.path.join(CKPT_ROOT, '2048_condM_chinchilla_13m_checkpoints', 'best.ptrom'),
        'label':      'condM 13M Chinchilla (400K docs, 1 epoch, 31.2 tok/param)',
        'params_ref': 13_984_480,
    },
    'condm_periodic_13m': {
        'arch':            'condm_periodic',
        'D':               224, 'H': 8, 'FFN': 896, 'L': 8,
        'full_attn_layers': [3, 7],
        'interference':     3,
        'checkpoint': os.path.join(CKPT_ROOT, '2048_condM_periodic_13m_checkpoints', 'best.pt'),
        'label':      'condM-periodic 13M [3:1:3:1] (2x DSQG3+Full)',
        'params_ref': 13_075_648,
    },
}

# ─── Offsets ──────────────────────────────────────────────────────────────────

OFFSETS_N = sorted(set(range(0, 33)) | {48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536})
OFFSETS_P = sorted(set(range(0, 65)) | {96, 128, 192, 256, 384, 512, 768, 1024, 1536})
assert len(OFFSETS_N) == 44
assert len(OFFSETS_P) == 74

# ─── Standard Transformer ─────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, D, H, seq_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.H, self.HD = H, D // H
        self.scale      = self.HD ** -0.5
        self.qkv_proj   = nn.Linear(D, 3 * D, bias=True)
        self.out_proj   = nn.Linear(D, D, bias=True)
        self.dropout    = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
        self.register_buffer('causal_mask', mask)

    def forward(self, x):
        B, N, D = x.shape
        H, HD   = self.H, self.HD
        qkv     = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(self.causal_mask[:, :, :N, :N] == 0, float('-inf'))
        alpha  = self.dropout(F.softmax(scores, dim=-1))
        out    = torch.matmul(alpha, v)
        return self.out_proj(out.permute(0, 2, 1, 3).reshape(B, N, D))


class StdFFN(nn.Module):
    def __init__(self, D, FFN, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(D, FFN)
        self.fc2  = nn.Linear(FFN, D)
        self.drop = nn.Dropout(dropout)
    def forward(self, x): return self.fc2(self.drop(F.gelu(self.fc1(x))))


class StdBlock(nn.Module):
    def __init__(self, D, H, FFN, seq_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(D)
        self.attn = CausalSelfAttention(D, H, seq_len, dropout)
        self.ln2  = nn.LayerNorm(D)
        self.ffn  = StdFFN(D, FFN, dropout)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class StandardTransformer(nn.Module):
    """Standard transformer with untied embeddings (consistent with baseline training)."""
    def __init__(self, D=256, H=8, FFN=1024, L=6, seq_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, D)
        self.pos_emb   = nn.Embedding(seq_len, D)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([StdBlock(D, H, FFN, seq_len, dropout) for _ in range(L)])
        self.ln_final  = nn.LayerNorm(D)
        self.out_proj  = nn.Linear(D, VOCAB_SIZE, bias=False)  # NOT tied
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, idx):
        B, N = idx.shape
        x = self.drop(self.token_emb(idx) + self.pos_emb(torch.arange(N, device=idx.device).unsqueeze(0)))
        for block in self.blocks: x = block(x)
        return self.out_proj(self.ln_final(x))

    def param_count(self): return sum(p.numel() for p in self.parameters())


# ─── DSQG Attention (condN offsets — used by condM) ──────────────────────────

class DSQGAttentionN(nn.Module):
    def __init__(self, D, H, seq_len=MAX_SEQ_LEN, offsets=None, dropout=0.1):
        super().__init__()
        if offsets is None: offsets = OFFSETS_N
        self.H, self.HD    = H, D // H
        self._offsets_list = offsets  # kept for n_offsets; buffer "offsets" registered below
        self.n_offsets     = len(offsets)
        self.qkv_proj      = nn.Linear(D, 3 * D, bias=True)
        self.out_proj      = nn.Linear(D, D, bias=True)
        self.gate_proj     = nn.Linear(D, D, bias=True)
        nn.init.constant_(self.gate_proj.bias, 2.0)
        pos_bias           = torch.zeros(len(offsets), H)
        self.pos_bias      = nn.Parameter(pos_bias)
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.long))

    def forward(self, x):
        # Exact copy of training script forward — (B,H,N,HD) layout, causal mask
        B, N, D = x.shape
        H, HD   = self.H, self.HD
        qkv     = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)   # B,H,N,HD
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)

        scale = HD ** -0.5
        K_list, V_list = [], []
        for delta in self.offsets.tolist():
            if delta == 0:
                K_list.append(k); V_list.append(v)
            elif delta >= N:
                K_list.append(torch.zeros_like(k))
                V_list.append(torch.zeros_like(v))
            else:
                pad = k.new_zeros(B, H, delta, HD)
                K_list.append(torch.cat([pad, k[:, :, :N - delta, :]], dim=2))
                V_list.append(torch.cat([pad, v[:, :, :N - delta, :]], dim=2))

        K_all = torch.stack(K_list, dim=3)   # B,H,N,n_off,HD
        V_all = torch.stack(V_list, dim=3)
        scores = (q.unsqueeze(3) * K_all).sum(-1) * scale   # B,H,N,n_off
        scores = scores + self.pos_bias.T.unsqueeze(0).unsqueeze(2)  # [1,H,1,n_off]

        n_idx = torch.arange(N, device=x.device).unsqueeze(1)
        d_idx = self.offsets.unsqueeze(0)
        scores = scores.masked_fill(
            (n_idx < d_idx).unsqueeze(0).unsqueeze(0), float('-inf'))

        alpha = F.softmax(scores, dim=-1)
        out   = (alpha.unsqueeze(-1) * V_all).sum(dim=3)    # B,H,N,HD
        out   = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate  = torch.sigmoid(self.gate_proj(x))
        return self.out_proj(out * gate)


class FullCausalAttention(nn.Module):
    def __init__(self, D, H, dropout=0.1):
        super().__init__()
        self.H, self.HD = H, D // H
        self.qkv_proj   = nn.Linear(D, 3 * D, bias=True)
        self.out_proj   = nn.Linear(D, D, bias=True)
        self.gate_proj  = nn.Linear(D, D, bias=True)
        nn.init.constant_(self.gate_proj.bias, 2.0)
        self.dropout_p  = dropout

    def forward(self, x):
        B, N, D = x.shape
        H, HD   = self.H, self.HD
        qkv     = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
              dropout_p=self.dropout_p if self.training else 0.0)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate = torch.sigmoid(self.gate_proj(x))
        return self.out_proj(out * gate)


class DSQGFFN(nn.Module):
    def __init__(self, D, FFN, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(D, FFN)
        self.fc2  = nn.Linear(FFN, D)
        self.drop = nn.Dropout(dropout)
    def forward(self, x): return self.fc2(self.drop(F.gelu(self.fc1(x))))


class DSQGBlock(nn.Module):
    def __init__(self, D, H, FFN, seq_len=MAX_SEQ_LEN, offsets=None, interference=False, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        self.attn  = DSQGAttentionN(D, H, seq_len, offsets, dropout)
        self.ffn   = DSQGFFN(D, FFN, dropout)
        self.interference = interference
        if interference:
            self.inter_norm = nn.LayerNorm(D)
            self.inter_gate = nn.Linear(D, D)
            self.inter_pool = nn.Linear(D, D)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        if self.interference:
            xi   = self.inter_norm(x)
            pool = xi.cumsum(dim=1) / torch.arange(1, xi.size(1) + 1,
                   device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(-1)
            x = x + torch.sigmoid(self.inter_gate(xi)) * self.inter_pool(pool)
        x = x + self.ffn(self.norm2(x))
        return x


class FullAttnBlock(nn.Module):
    def __init__(self, D, H, FFN, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        self.attn  = FullCausalAttention(D, H, dropout)
        self.ffn   = DSQGFFN(D, FFN, dropout)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CondMTransformer(nn.Module):
    """condM: configurable N DSQG layers + 1 full causal attention at full_layer."""
    def __init__(self, D=256, H=8, FFN=1024, L=6, full_layer=5, interf=3,
                 seq_len=MAX_SEQ_LEN, offsets=None, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D)
        self.pos_embed = nn.Embedding(seq_len + 2, D)
        self.drop      = nn.Dropout(dropout)
        blocks = []
        for i in range(L):
            if i == full_layer:
                blocks.append(FullAttnBlock(D, H, FFN, dropout))
            else:
                blocks.append(DSQGBlock(D, H, FFN, seq_len, offsets,
                               interference=(i % interf == interf - 1), dropout=dropout))
        self.blocks = nn.ModuleList(blocks)
        self.norm   = nn.LayerNorm(D)
        self.out    = nn.Linear(D, VOCAB_SIZE, bias=False)
        self.out.weight = self.embedding.weight  # tied

    def forward(self, idx):
        B, N = idx.shape
        x = self.drop(self.embedding(idx) +
            self.pos_embed(torch.arange(N, device=idx.device).unsqueeze(0)))
        for block in self.blocks: x = block(x)
        return self.out(self.norm(x))

    def param_count(self): return sum(p.numel() for p in self.parameters())


# ─── condP DSQG Attention (74 offsets) ───────────────────────────────────────

class DSQGAttentionP(nn.Module):
    def __init__(self, D, H, seq_len=MAX_SEQ_LEN, offsets=None, dropout=0.1):
        super().__init__()
        if offsets is None: offsets = OFFSETS_P
        self.H, self.HD = H, D // H
        self._offsets_list = offsets  # kept for n_offsets; buffer "offsets" registered below
        self.n_offsets  = len(offsets)
        self.qkv_proj   = nn.Linear(D, 3 * D, bias=True)
        self.out_proj   = nn.Linear(D, D, bias=True)
        self.gate_proj  = nn.Linear(D, D, bias=True)
        nn.init.constant_(self.gate_proj.bias, 2.0)
        self.pos_bias   = nn.Parameter(torch.zeros(len(offsets), H))
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.long))

    def forward(self, x):
        # Exact copy of condP training script forward — (B,H,N,HD) layout, causal mask
        B, N, D = x.shape
        H, HD   = self.H, self.HD
        qkv     = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)   # B,H,N,HD
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)

        scale = HD ** -0.5
        K_list, V_list = [], []
        for delta in self.offsets.tolist():
            if delta == 0:
                K_list.append(k); V_list.append(v)
            elif delta >= N:
                K_list.append(torch.zeros_like(k))
                V_list.append(torch.zeros_like(v))
            else:
                pad = k.new_zeros(B, H, delta, HD)
                K_list.append(torch.cat([pad, k[:, :, :N - delta, :]], dim=2))
                V_list.append(torch.cat([pad, v[:, :, :N - delta, :]], dim=2))

        K_all  = torch.stack(K_list, dim=3)   # B,H,N,n_off,HD
        V_all  = torch.stack(V_list, dim=3)
        scores = (q.unsqueeze(3) * K_all).sum(-1) * scale   # B,H,N,n_off
        scores = scores + self.pos_bias.T.unsqueeze(0).unsqueeze(2)  # [1,H,1,n_off]

        n_idx = torch.arange(N, device=x.device).unsqueeze(1)
        d_idx = self.offsets.unsqueeze(0)
        scores = scores.masked_fill(
            (n_idx < d_idx).unsqueeze(0).unsqueeze(0), float('-inf'))

        alpha = F.softmax(scores, dim=-1)
        out   = (alpha.unsqueeze(-1) * V_all).sum(dim=3)    # B,H,N,HD
        out   = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate  = torch.sigmoid(self.gate_proj(x))
        return self.out_proj(out * gate)


class DSQGBlockP(nn.Module):
    def __init__(self, D, H, FFN, seq_len=MAX_SEQ_LEN, offsets=None, interference=False, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        self.attn  = DSQGAttentionP(D, H, seq_len, offsets, dropout)
        self.ffn   = DSQGFFN(D, FFN, dropout)
        self.interference = interference
        if interference:
            self.inter_norm = nn.LayerNorm(D)
            self.inter_gate = nn.Linear(D, D)
            self.inter_pool = nn.Linear(D, D)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        if self.interference:
            xi   = self.inter_norm(x)
            pool = xi.cumsum(dim=1) / torch.arange(1, xi.size(1) + 1,
                   device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(-1)
            x = x + torch.sigmoid(self.inter_gate(xi)) * self.inter_pool(pool)
        x = x + self.ffn(self.norm2(x))
        return x


class CondPTransformer(nn.Module):
    """condP: pure DSQG with 74 offsets (dense-64 + dyadic), tied embeddings."""
    def __init__(self, D=256, H=8, FFN=1024, L=6, interf=3,
                 seq_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D)
        self.pos_embed = nn.Embedding(seq_len + 2, D)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            DSQGBlockP(D, H, FFN, seq_len, OFFSETS_P,
                       interference=(i % interf == interf - 1))
            for i in range(L)
        ])
        self.norm = nn.LayerNorm(D)
        self.out  = nn.Linear(D, VOCAB_SIZE, bias=False)
        self.out.weight = self.embedding.weight  # tied

    def forward(self, idx):
        B, N = idx.shape
        x = self.drop(self.embedding(idx) +
            self.pos_embed(torch.arange(N, device=idx.device).unsqueeze(0)))
        for block in self.blocks: x = block(x)
        return self.out(self.norm(x))

    def param_count(self): return sum(p.numel() for p in self.parameters())


# ─── Model builder ────────────────────────────────────────────────────────────

def build_model(cfg):
    arch = cfg['arch']
    D, H, FFN, L = cfg['D'], cfg['H'], cfg['FFN'], cfg['L']
    if arch == 'standard':
        return StandardTransformer(D=D, H=H, FFN=FFN, L=L)
    elif arch == 'condm':
        return CondMTransformer(D=D, H=H, FFN=FFN, L=L, full_layer=cfg['full_layer'])
    elif arch == 'condp':
        return CondPTransformer(D=D, H=H, FFN=FFN, L=L)
    elif arch == 'condm_periodic':
        CondMPeriodicTransformer = _import_condm_periodic()
        return CondMPeriodicTransformer(
            vocab_size=VOCAB_SIZE, D=D, L=L, H=H, FFN_dim=FFN,
            seq_len=MAX_SEQ_LEN,
            full_attn_layers=set(cfg.get('full_attn_layers', [L - 1])),
            interference_interval=cfg.get('interference', 3),
        )
    elif arch == 'condm_v2':
        CondMV2Transformer = _import_condm_v2()
        ffn_hidden = int(8 * D / 3)   # SwiGLU iso-parameter: 8D/3
        return CondMV2Transformer(
            vocab_size=VOCAB_SIZE, embedding_dim=D, num_layers=L, num_heads=H,
            ffn_hidden=ffn_hidden, seq_len=MAX_SEQ_LEN,
            interference_interval=cfg.get('interference', 3),
            full_attn_layer=cfg.get('full_layer', 5),
        )
    else:
        raise ValueError(f"Unknown arch: {arch}")


def load_model(cfg, device):
    ckpt_path = cfg['checkpoint']
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = build_model(cfg)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    elif isinstance(state, dict) and 'model' in state:
        state = state['model']
    # strict=False: standard_85m was trained with SDPA (no causal_mask buffer);
    # all trainable weights match — only the derived buffer is absent from ckpt.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected}")
    causal_only = [k for k in missing if "causal_mask" in k]
    other_missing = [k for k in missing if "causal_mask" not in k]
    if other_missing:
        raise RuntimeError(f"Missing non-buffer keys: {other_missing}")
    model = model.to(device)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {cfg['label']}: {n:,} params from {os.path.basename(ckpt_path)}")
    return model, n


# ─── Tokenizer ────────────────────────────────────────────────────────────────

class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)


def load_tokenizer():
    from tokenizers import Tokenizer
    return BPETokenizerWrapper(Tokenizer.from_file(TOKENIZER))


# ─── Sampling ─────────────────────────────────────────────────────────────────

def sample_top_p(probs, top_p=0.9):
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumprobs = sorted_probs.cumsum(dim=-1)
    mask     = (cumprobs - sorted_probs) >= top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum()
    idx = torch.multinomial(sorted_probs, 1)
    return sorted_idx[idx]


def generate_tokens(model, tokenizer, prompt, device, max_new=150,
                    temperature=1.0, top_p=0.9, greedy=False):
    ids = tokenizer.encode(prompt)
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    gen = []
    with torch.no_grad():
        for _ in range(max_new):
            inp   = ids[:, -MAX_SEQ_LEN:]
            logits = model(inp)[:, -1, :]
            if greedy:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probs   = F.softmax(logits / temperature, dim=-1)
                next_id = torch.tensor([[sample_top_p(probs[0], top_p).item()]], dtype=torch.long, device=device)
            ids = torch.cat([ids, next_id], dim=1)
            gen.append(next_id.item())
    return tokenizer.decode(gen)


# ─── 1. CALIBRATION ───────────────────────────────────────────────────────────

_CALIB_PROMPTS = [
    "The weather today is",
    "In the field of computer science,",
    "Once upon a time, there lived",
    "The president announced that",
    "Scientists have discovered that",
    "According to recent studies,",
    "The stock market experienced",
    "In the beginning, God created",
    "To be or not to be,",
    "The quick brown fox",
]

def eval_calibration(model, tokenizer, device, max_new=200, top_p=0.9):
    results = []
    with torch.no_grad():
        for prompt in _CALIB_PROMPTS:
            ids    = tokenizer.encode(prompt)
            ids    = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(ids)[:, -1, :]
            probs  = F.softmax(logits, dim=-1)[0]
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item() / math.log(2)
            top1_conf = probs.max().item()
            eff_vocab = (1.0 / (probs ** 2).sum()).item()
            results.append({
                'prompt':     prompt,
                'entropy':    entropy,
                'top1_conf':  top1_conf,
                'eff_vocab':  eff_vocab,
            })
    agg = {
        'mean_entropy':    sum(r['entropy']   for r in results) / len(results),
        'mean_top1_conf':  sum(r['top1_conf'] for r in results) / len(results),
        'mean_eff_vocab':  sum(r['eff_vocab'] for r in results) / len(results),
    }
    return {'per_prompt': results, 'aggregate': agg}


# ─── 2. DISTANCE-CONDITIONED LOSS (wikitext-103, 17 full chunks) ──────────────

DIST_BUCKETS = [
    ('0-16',    0,    17),
    ('17-64',   17,   65),
    ('65-256',  65,   257),
    ('257-512', 257,  513),
    ('513-1024',513,  1025),
    ('1025-2047',1025,2048),
]

def bucket_of(pos):
    for name, lo, hi in DIST_BUCKETS:
        if lo <= pos < hi: return name
    return '1025-2047'

def compute_per_token_loss(model, token_ids, device):
    ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(ids)
    shift_logits = logits[0, :-1, :]
    shift_labels = ids[0, 1:]
    losses = F.cross_entropy(shift_logits, shift_labels, reduction='none')
    return losses.cpu().tolist()

def eval_distance_loss(model, tokenizer, device):
    from datasets import load_dataset
    print('    Loading wikitext-103 test split...')
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
    all_ids = []
    for item in ds:
        t = item['text'].strip()
        if t:
            all_ids.extend(tokenizer.encode(t))

    chunk_size = MAX_SEQ_LEN
    chunks = [all_ids[i:i+chunk_size] for i in range(0, len(all_ids), chunk_size)
              if len(all_ids[i:i+chunk_size]) == chunk_size]
    n_chunks = min(17, len(chunks))
    print(f'    Using {n_chunks} full {chunk_size}-token chunks')

    bucket_losses = {b[0]: [] for b in DIST_BUCKETS}
    for chunk in chunks[:n_chunks]:
        per_tok = compute_per_token_loss(model, chunk, device)
        for pos, loss in enumerate(per_tok):
            bname = bucket_of(pos)
            bucket_losses[bname].append(loss)

    results = {}
    for bname, lo, hi in DIST_BUCKETS:
        losses = bucket_losses[bname]
        if losses:
            mean_loss = sum(losses) / len(losses)
            results[bname] = {
                'mean_loss': mean_loss,
                'ppl':       math.exp(mean_loss),
                'n_tokens':  len(losses),
            }
    return {'buckets': results, 'n_chunks': n_chunks}


# ─── 3. FEW-SHOT STRING COPY ──────────────────────────────────────────────────

_COPY_TESTS = [
    {'name': 'word_copy_at_distance',
     'template': 'The secret word is "{w}". Remember it.\n' * 3 + 'What is the secret word? The secret word is "',
     'words': ['apple', 'mountain', 'river', 'castle', 'thunder']},
    {'name': 'number_copy',
     'template': 'The secret number is {w}. Remember it.\n' * 3 + 'What is the secret number? The secret number is ',
     'words': ['42', '789', '13', '256', '99']},
    {'name': 'keyword_repeat',
     'template': '{w} {w} {w} {w} {w} {w} {w} {w} {w} {w} {w} {w} {w} {w} {w} ',
     'words': ['banana', 'elephant', 'library', 'gravity', 'horizon']},
]

def levenshtein(a, b):
    if len(a) > len(b): a, b = b, a
    row = list(range(len(a) + 1))
    for c2 in b:
        new_row = [row[0] + 1]
        for j, c1 in enumerate(a):
            new_row.append(min(row[j + 1] + 1, new_row[-1] + 1,
                               row[j] + (0 if c1 == c2 else 1)))
        row = new_row
    return row[-1]

def eval_few_shot_copy(model, tokenizer, device, temperature=0.1, top_p=0.9):
    results = []
    with torch.no_grad():
        for test in _COPY_TESTS:
            for word in test['words']:
                prompt = test['template'].format(w=word)
                gen    = generate_tokens(model, tokenizer, prompt, device,
                                         max_new=20, temperature=temperature,
                                         top_p=top_p)
                gen_lower  = gen.lower().strip()
                word_lower = word.lower()
                exact = gen_lower.startswith(word_lower)
                lev   = levenshtein(gen_lower[:len(word_lower)+5], word_lower)
                results.append({
                    'test':        test['name'],
                    'target':      word,
                    'generated':   gen[:50],
                    'exact_match': exact,
                    'levenshtein': lev,
                })
    total   = len(results)
    exact   = sum(r['exact_match'] for r in results)
    mean_lev = sum(r['levenshtein'] for r in results) / total
    return {
        'per_test':       results,
        'exact_match_n':  exact,
        'exact_match_pct': exact / total,
        'mean_levenshtein': mean_lev,
        'total_tests':    total,
    }


# ─── 4. PASSKEY RETRIEVAL ─────────────────────────────────────────────────────

_PASSKEY_WORDS   = ['apple', 'banana', 'orange', 'cherry', 'grape',
                    'mango', 'peach', 'plum', 'kiwi', 'melon']
_FILLER_SENTENCE = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE  = 'the secret word is {word} .'
_RETRIEVAL_CUE   = 'the secret word is'  # model predicts " <word>" next (space-prefixed)

def eval_passkey_at_distance(model, tokenizer, distance, device, n_trials=5):
    """Exact format from eval_condM_vs_condP27m.py (proven correct)."""
    filler_ids = tokenizer.encode(_FILLER_SENTENCE)
    cue_ids    = tokenizer.encode(_RETRIEVAL_CUE)

    results = []
    for i in range(n_trials):
        target_word = _PASSKEY_WORDS[i % len(_PASSKEY_WORDS)]
        other_words = [w for w in _PASSKEY_WORDS if w != target_word]
        intro_ids   = tokenizer.encode(_INTRO_TEMPLATE.format(word=target_word))

        available = MAX_SEQ_LEN - 1 - len(intro_ids) - len(cue_ids) - 1
        if distance > available:
            results.append({'distance': distance, 'target': target_word, 'skipped': True})
            continue

        filler = []
        while len(filler) < distance:
            filler.extend(filler_ids)
        filler = filler[:distance]

        full_seq = intro_ids + filler + cue_ids
        if len(full_seq) >= MAX_SEQ_LEN:
            results.append({'distance': distance, 'target': target_word, 'skipped': True})
            continue

        ids = torch.tensor([full_seq], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(ids)[:, -1, :]
        last = logits[0]

        # 10-way choice: target vs 9 distractors (all space-prefixed, first token)
        candidates = [target_word] + other_words[:9]
        candidate_ids = []
        for w in candidates:
            toks = tokenizer.encode(' ' + w)
            candidate_ids.append(toks[0] if toks else tokenizer.encode(w)[0])

        cand_scores = last[candidate_ids]
        pred        = candidates[cand_scores.argmax().item()]
        correct     = (pred == target_word)
        results.append({
            'distance': distance, 'target': target_word,
            'predicted': pred, 'correct': correct, 'skipped': False,
        })

    valid = [r for r in results if not r.get('skipped')]
    acc   = sum(r['correct'] for r in valid) / len(valid) if valid else 0.0
    return {'distance': distance, 'accuracy': acc, 'n_valid': len(valid), 'trials': results}


def eval_passkey(model, tokenizer, device):
    distances = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
    results   = []
    for d in distances:
        r = eval_passkey_at_distance(model, tokenizer, d, device)
        results.append(r)
        print(f'    d={d:5d}: accuracy={r["accuracy"]:.1%}')
    mean_acc = sum(r['accuracy'] for r in results) / len(results)
    return {
        'per_distance': results,
        'mean_accuracy': mean_acc,
        'distances': distances,
    }


# ─── Temperature sweep (generation quality) ───────────────────────────────────

_GEN_PROMPTS = [
    'It was a dark and stormy',
    'The length of the hypotenuse',
    'The President of the United',
    'Once upon a time there was',
    'The results indicate that',
]

def eval_generation(model, tokenizer, device):
    sweep = {}
    for label, greedy, temp, top_p in [
        ('greedy', True, 1.0, 0.9),
        ('T=0.7',  False, 0.7, 0.9),
    ]:
        sweep[label] = []
        for prompt in _GEN_PROMPTS:
            gen = generate_tokens(model, tokenizer, prompt, device,
                                  max_new=100, temperature=temp,
                                  top_p=top_p, greedy=greedy)
            sweep[label].append({'prompt': prompt, 'generated': gen})
    return sweep


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model to evaluate')
    parser.add_argument('--skip_distance', action='store_true',
                        help='Skip the (slow) distance-conditioned loss eval')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*60}')
    print(f'  eval_suite.py — {args.model}')
    print(f'  Device: {device}')
    print(f'{"="*60}\n')

    cfg   = MODEL_REGISTRY[args.model]
    print(f'  Model: {cfg["label"]}')

    # Load tokenizer + model
    print('\n  Loading tokenizer...')
    tokenizer = load_tokenizer()
    print('  Loading model...')
    model, n_params = load_model(cfg, device)

    ts      = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    outfile = os.path.join(LOGS_DIR, f'eval_suite_{args.model}_{ts}.json')
    results = {
        'model':      args.model,
        'label':      cfg['label'],
        'arch':       cfg['arch'],
        'D': cfg['D'], 'H': cfg['H'], 'FFN': cfg['FFN'], 'L': cfg['L'],
        'n_params':   n_params,
        'timestamp':  ts,
        'device':     str(device),
    }
    if 'full_layer' in cfg:
        results['full_layer'] = cfg['full_layer']

    # 1. Calibration
    print('\n  [1/4] Calibration...')
    t0 = time.time()
    results['calibration'] = eval_calibration(model, tokenizer, device)
    agg = results['calibration']['aggregate']
    print(f'    entropy={agg["mean_entropy"]:.3f} bits  top1={agg["mean_top1_conf"]:.3f}'
          f'  eff_vocab={agg["mean_eff_vocab"]:.1f}  ({time.time()-t0:.1f}s)')

    # 2. Distance-conditioned loss
    if not args.skip_distance:
        print('\n  [2/4] Distance-conditioned loss (wikitext-103)...')
        t0 = time.time()
        results['distance_loss'] = eval_distance_loss(model, tokenizer, device)
        print(f'    Buckets:')
        for bname, lo, hi in DIST_BUCKETS:
            b = results['distance_loss']['buckets'].get(bname, {})
            if b:
                print(f'    {bname:12s}  PPL={b["ppl"]:7.2f}  n={b["n_tokens"]:5d}')
        print(f'    ({time.time()-t0:.1f}s)')
    else:
        print('\n  [2/4] Distance-conditioned loss — SKIPPED')

    # 3. Few-shot copy
    print('\n  [3/4] Few-shot string copy...')
    t0 = time.time()
    results['few_shot_copy'] = eval_few_shot_copy(model, tokenizer, device)
    fsc = results['few_shot_copy']
    print(f'    exact={fsc["exact_match_n"]}/{fsc["total_tests"]}  '
          f'({fsc["exact_match_pct"]:.1%})  mean_lev={fsc["mean_levenshtein"]:.2f}'
          f'  ({time.time()-t0:.1f}s)')

    # 4. Passkey retrieval
    print('\n  [4/4] Passkey retrieval...')
    t0 = time.time()
    results['passkey'] = eval_passkey(model, tokenizer, device)
    print(f'    mean_accuracy={results["passkey"]["mean_accuracy"]:.1%}  ({time.time()-t0:.1f}s)')

    # Generation samples
    print('\n  Generation samples (greedy + T=0.7)...')
    results['generation'] = eval_generation(model, tokenizer, device)
    for label, samples in results['generation'].items():
        print(f'\n  [{label}]')
        for s in samples:
            print(f'    {s["prompt"]!r:40s} → {s["generated"][:80]!r}')

    # Save
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results saved to {outfile}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
