#!/usr/bin/env python3
"""
Comparative Evaluation: condM 13M (5:1 hybrid) vs condP 27M (pure DSQG)

Runs four evaluations:
  1. Calibration — logit entropy, top-1 confidence, effective vocabulary size
  2. Distance-conditioned loss — per-position CE loss bucketed by lookback distance
  3. Few-shot string copy — template-following / induction head capability
  4. Passkey retrieval — content-addressed memory at varying distances

Architectures:
  condM:     D=256, H=8, FFN=1024, L=6, FULL_ATTN_LAYER=5 (5 DSQG + 1 full attn)
             condN offsets (44: dense-32 + dyadic), ~14M params
  condP 27M: D=400, H=8, FFN=1600, L=6, pure DSQG
             condP offsets (74: dense-64 + dyadic), ~27M params

Both use the condI BPE tokenizer (vocab=32000).

Usage:
  cd /home/dlewis3/Desktop/AI/DWARF
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 benchmarks/eval_condM_vs_condP27m.py

Results logged to: benchmarks/logs/eval_condM_vs_condP27m_<timestamp>.json
"""

import json, math, os, sys, time, datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.dirname(SCRIPT_DIR)
TOKENIZER    = os.path.join(SCRIPT_DIR, 'results', '2048_condI_tokenizer.json')
CONDM_CKPT   = os.path.join(REPO_ROOT, 'checkpoints', '2048_condM_checkpoints', 'best.pt')
CONDP27_CKPT = os.path.join(REPO_ROOT, 'checkpoints', '27m_2048__condP_checkpoints', 'best.ptrom')
LOGS_DIR     = os.path.join(SCRIPT_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

MAX_SEQ_LEN  = 2048
VOCAB_SIZE   = 32000

# ─── condM offsets (condN: dense-32 + dyadic) ─────────────────────────────────
_COND_N_OFFSETS = sorted(
    set(range(0, 33)) | {48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536})
assert len(_COND_N_OFFSETS) == 44

# ─── condP offsets (dense-64 + dyadic) ────────────────────────────────────────
_COND_P_OFFSETS = sorted(
    set(range(0, 65)) | {96, 128, 192, 256, 384, 512, 768, 1024, 1536})
assert len(_COND_P_OFFSETS) == 74

# ─── condM architecture ───────────────────────────────────────────────────────

class DSQGAttentionN(nn.Module):
    """condN-style DSQG attention (44 offsets: dense-32 + dyadic)."""
    def __init__(self, embedding_dim, num_heads, seq_len=2048, offsets=None, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads
        self.seq_len       = seq_len
        if offsets is None:
            offsets = _COND_N_OFFSETS
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.long))
        self.n_offsets = len(offsets)
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 2.0)
        alphas = torch.linspace(0.2, 2.0, num_heads)
        delta_vals = torch.tensor([math.log(1.0 + d) for d in offsets], dtype=torch.float32)
        self.pos_bias = nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)
        scale = HD ** -0.5
        K_list, V_list = [], []
        for delta in self.offsets.tolist():
            if delta == 0:
                K_list.append(k); V_list.append(v)
            elif delta >= N:
                K_list.append(torch.zeros_like(k)); V_list.append(torch.zeros_like(v))
            else:
                pad = k.new_zeros(B, H, delta, HD)
                K_list.append(torch.cat([pad, k[:, :, :N-delta, :]], dim=2))
                V_list.append(torch.cat([pad, v[:, :, :N-delta, :]], dim=2))
        K_all  = torch.stack(K_list, dim=3)
        V_all  = torch.stack(V_list, dim=3)
        scores = (q.unsqueeze(3) * K_all).sum(-1) * scale
        scores = scores + self.pos_bias.T.unsqueeze(0).unsqueeze(2)
        n_idx  = torch.arange(N, device=x.device).unsqueeze(1)
        d_idx  = self.offsets.unsqueeze(0)
        scores = scores.masked_fill((n_idx < d_idx).unsqueeze(0).unsqueeze(0), float('-inf'))
        alpha  = F.softmax(scores, dim=-1)
        out    = (alpha.unsqueeze(-1) * V_all).sum(dim=3)
        flat   = out.permute(0, 2, 1, 3).reshape(B, N, D)
        return self.dropout(self.out_proj(flat * torch.sigmoid(self.gate_proj(x))))


class FullCausalAttention(nn.Module):
    """Standard O(N²) causal attention for the condM hybrid layer."""
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 2.0)
        self.dropout_p = dropout

    def forward(self, x):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True)
        flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        return F.dropout(self.out_proj(flat * torch.sigmoid(self.gate_proj(x))),
                         p=self.dropout_p, training=self.training)


class FFN_M(nn.Module):
    def __init__(self, d, ffn_d, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d, ffn_d); self.fc2 = nn.Linear(ffn_d, d)
        self.drop = nn.Dropout(dropout)
    def forward(self, x): return self.fc2(self.drop(F.gelu(self.fc1(x))))


class DSQGBlock_M(nn.Module):
    def __init__(self, d, h, ffn_d, seq_len, dropout=0.1, use_ckpt=True, interference=False):
        super().__init__()
        self.use_ckpt    = use_ckpt
        self.interference = interference
        self.norm1 = nn.LayerNorm(d); self.norm2 = nn.LayerNorm(d)
        self.attn  = DSQGAttentionN(d, h, seq_len=seq_len, dropout=dropout)
        self.ffn   = FFN_M(d, ffn_d, dropout)
        if interference:
            self.inter_norm = nn.LayerNorm(d)
            self.inter_gate = nn.Linear(d, d)
            self.inter_pool = nn.Linear(d, d)
    def _af(self, x): return self.attn(self.norm1(x))
    def forward(self, x):
        if self.use_ckpt:
            x = x + torch.utils.checkpoint.checkpoint(self._af, x, use_reentrant=False)
        else:
            x = x + self._af(x)
        if self.interference:
            xi = self.inter_norm(x)
            B, N, D = xi.shape
            counts = torch.arange(1, N+1, device=xi.device, dtype=xi.dtype).view(1, N, 1)
            x = x + torch.sigmoid(self.inter_gate(xi)) * self.inter_pool(xi.cumsum(1) / counts)
        return x + self.ffn(self.norm2(x))


class FullAttnBlock(nn.Module):
    def __init__(self, d, h, ffn_d, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d); self.norm2 = nn.LayerNorm(d)
        self.attn  = FullCausalAttention(d, h, dropout)
        self.ffn   = FFN_M(d, ffn_d, dropout)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class CondMTransformer(nn.Module):
    """condM: 5 DSQG layers (condN offsets) + 1 full causal attention at layer 5."""
    D          = 256
    NUM_HEADS  = 8
    FFN_DIM    = 1024
    NUM_LAYERS = 6
    FULL_LAYER = 5
    INTERF     = 3

    def __init__(self):
        super().__init__()
        D, H, FFN, L = self.D, self.NUM_HEADS, self.FFN_DIM, self.NUM_LAYERS
        self.embedding = nn.Embedding(VOCAB_SIZE, D)
        self.pos_embed = nn.Embedding(MAX_SEQ_LEN + 2, D)
        self.drop      = nn.Dropout(0.1)
        blocks = []
        for i in range(L):
            if i == self.FULL_LAYER:
                blocks.append(FullAttnBlock(D, H, FFN))
            else:
                blocks.append(DSQGBlock_M(
                    D, H, FFN, MAX_SEQ_LEN, use_ckpt=False,
                    interference=(i % self.INTERF == self.INTERF - 1)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(D)
        self.out  = nn.Linear(D, VOCAB_SIZE, bias=False)
        self.out.weight = self.embedding.weight

    def forward(self, idx):
        B, N = idx.shape
        x = self.drop(self.embedding(idx) + self.pos_embed(
            torch.arange(N, device=idx.device).unsqueeze(0)))
        for block in self.blocks: x = block(x)
        return self.out(self.norm(x))

    def param_count(self): return sum(p.numel() for p in self.parameters())


# ─── condP 27M architecture ───────────────────────────────────────────────────

class DSQGAttentionP(nn.Module):
    """condP-style DSQG attention (74 offsets: dense-64 + dyadic)."""
    def __init__(self, embedding_dim, num_heads, seq_len=2048, offsets=None, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads
        self.seq_len       = seq_len
        if offsets is None:
            offsets = _COND_P_OFFSETS
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.long))
        self.n_offsets = len(offsets)
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 2.0)
        alphas = torch.linspace(0.2, 2.0, num_heads)
        delta_vals = torch.tensor([math.log(1.0 + d) for d in offsets], dtype=torch.float32)
        self.pos_bias = nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)
        scale = HD ** -0.5
        K_list, V_list = [], []
        for delta in self.offsets.tolist():
            if delta == 0:
                K_list.append(k); V_list.append(v)
            elif delta >= N:
                K_list.append(torch.zeros_like(k)); V_list.append(torch.zeros_like(v))
            else:
                pad = k.new_zeros(B, H, delta, HD)
                K_list.append(torch.cat([pad, k[:, :, :N-delta, :]], dim=2))
                V_list.append(torch.cat([pad, v[:, :, :N-delta, :]], dim=2))
        K_all  = torch.stack(K_list, dim=3)
        V_all  = torch.stack(V_list, dim=3)
        scores = (q.unsqueeze(3) * K_all).sum(-1) * scale
        scores = scores + self.pos_bias.T.unsqueeze(0).unsqueeze(2)
        n_idx  = torch.arange(N, device=x.device).unsqueeze(1)
        d_idx  = self.offsets.unsqueeze(0)
        scores = scores.masked_fill((n_idx < d_idx).unsqueeze(0).unsqueeze(0), float('-inf'))
        alpha  = F.softmax(scores, dim=-1)
        out    = (alpha.unsqueeze(-1) * V_all).sum(dim=3)
        flat   = out.permute(0, 2, 1, 3).reshape(B, N, D)
        return self.dropout(self.out_proj(flat * torch.sigmoid(self.gate_proj(x))))


class FFN_P(nn.Module):
    def __init__(self, d, ffn_d, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d, ffn_d); self.fc2 = nn.Linear(ffn_d, d)
        self.drop = nn.Dropout(dropout)
    def forward(self, x): return self.fc2(self.drop(F.gelu(self.fc1(x))))


class DSQGBlock_P(nn.Module):
    def __init__(self, d, h, ffn_d, seq_len, dropout=0.1, use_ckpt=True, interference=False):
        super().__init__()
        self.use_ckpt    = use_ckpt
        self.interference = interference
        self.norm1 = nn.LayerNorm(d); self.norm2 = nn.LayerNorm(d)
        self.attn  = DSQGAttentionP(d, h, seq_len=seq_len, dropout=dropout)
        self.ffn   = FFN_P(d, ffn_d, dropout)
        if interference:
            self.inter_norm = nn.LayerNorm(d)
            self.inter_gate = nn.Linear(d, d)
            self.inter_pool = nn.Linear(d, d)
    def _af(self, x): return self.attn(self.norm1(x))
    def forward(self, x):
        if self.use_ckpt:
            x = x + torch.utils.checkpoint.checkpoint(self._af, x, use_reentrant=False)
        else:
            x = x + self._af(x)
        if self.interference:
            xi = self.inter_norm(x)
            B, N, D = xi.shape
            counts = torch.arange(1, N+1, device=xi.device, dtype=xi.dtype).view(1, N, 1)
            x = x + torch.sigmoid(self.inter_gate(xi)) * self.inter_pool(xi.cumsum(1) / counts)
        return x + self.ffn(self.norm2(x))


class CondP27MTransformer(nn.Module):
    """condP 27M: D=400, H=8, FFN=1600, L=6, pure DSQG (74 condP offsets)."""
    D          = 400
    NUM_HEADS  = 8
    FFN_DIM    = 1600
    NUM_LAYERS = 6
    INTERF     = 3

    def __init__(self):
        super().__init__()
        D, H, FFN, L = self.D, self.NUM_HEADS, self.FFN_DIM, self.NUM_LAYERS
        self.embedding = nn.Embedding(VOCAB_SIZE, D)
        self.pos_embed = nn.Embedding(MAX_SEQ_LEN + 2, D)
        self.drop      = nn.Dropout(0.1)
        self.blocks    = nn.ModuleList([
            DSQGBlock_P(D, H, FFN, MAX_SEQ_LEN, use_ckpt=False,
                        interference=(i % self.INTERF == self.INTERF - 1))
            for i in range(L)
        ])
        self.norm = nn.LayerNorm(D)
        self.out  = nn.Linear(D, VOCAB_SIZE, bias=False)
        self.out.weight = self.embedding.weight

    def forward(self, idx):
        B, N = idx.shape
        x = self.drop(self.embedding(idx) + self.pos_embed(
            torch.arange(N, device=idx.device).unsqueeze(0)))
        for block in self.blocks: x = block(x)
        return self.out(self.norm(x))

    def param_count(self): return sum(p.numel() for p in self.parameters())


# ─── Tokenizer ────────────────────────────────────────────────────────────────

class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)
    def vocab_size(self):   return self.tokenizer.get_vocab_size()


# ─── Sampling ─────────────────────────────────────────────────────────────────

def sample_top_p(probs, top_p=0.9):
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    mask   = cumsum - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum()
    return int(sorted_idx[torch.multinomial(sorted_probs, 1)])


def generate_tokens(model, tokenizer, prompt, device, max_new=200,
                    temperature=1.0, top_p=0.9):
    ids    = tokenizer.encode(prompt)
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(tensor[:, -MAX_SEQ_LEN:])
            last   = logits[0, -1]
            probs  = F.softmax(last / max(temperature, 0.01), dim=-1)
            nid    = sample_top_p(probs, top_p)
            tensor = torch.cat([tensor, torch.tensor([[nid]], device=device)], dim=1)
    return tensor[0, len(ids):].tolist()


# ─── 1. CALIBRATION ───────────────────────────────────────────────────────────

CALIB_PROMPTS = [
    'It was a dark and stormy night and the wind howled',
    'The length of the hypotenuse is determined by',
    'The President signed the executive order regarding',
    'Once upon a time there was a princess who lived in',
    'The experimental results clearly indicate that the proposed method',
    'Scientists have recently discovered a new species of',
    'The best way to learn programming is to start with',
    'In the year 2045, artificial intelligence had become',
]


def eval_calibration(model, tokenizer, device, max_new=200, top_p=0.9):
    model.eval()
    all_entropy, all_top1, all_top5, all_eff = [], [], [], []
    prompt_results = {}

    for prompt in CALIB_PROMPTS:
        ids    = tokenizer.encode(prompt)
        tensor = torch.tensor([ids], dtype=torch.long, device=device)
        step_stats = []

        with torch.no_grad():
            for _ in range(max_new):
                logits     = model(tensor[:, -MAX_SEQ_LEN:])
                last       = logits[0, -1]
                probs      = F.softmax(last, dim=-1)
                log_probs  = torch.log(probs + 1e-10)
                entropy    = float(-(probs * (log_probs / math.log(2))).sum())
                top1       = float(probs.max())
                top5_sum   = float(torch.topk(probs, 5).values.sum())
                eff_vocab  = 2 ** entropy
                step_stats.append({'entropy_bits': entropy, 'top1_prob': top1,
                                   'top5_mass': top5_sum, 'effective_vocab': eff_vocab})
                all_entropy.append(entropy); all_top1.append(top1)
                all_top5.append(top5_sum);   all_eff.append(eff_vocab)
                # Sample to continue generation
                probs_s = F.softmax(last, dim=-1)
                nid     = sample_top_p(probs_s, top_p)
                tensor  = torch.cat([tensor, torch.tensor([[nid]], device=device)], dim=1)

        me = sum(e['entropy_bits'] for e in step_stats) / len(step_stats)
        mt = sum(e['top1_prob']    for e in step_stats) / len(step_stats)
        prompt_results[prompt[:60]] = {'mean_entropy': me, 'mean_top1': mt,
                                        'n_steps': len(step_stats)}

    def pct(lst, p):
        s = sorted(lst); n = len(s)
        return s[max(0, min(n-1, int(n * p / 100)))]

    n = len(all_entropy)
    agg = {
        'n_steps':        n,
        'mean_entropy':   sum(all_entropy) / n,
        'mean_top1_prob': sum(all_top1)    / n,
        'mean_top5_mass': sum(all_top5)    / n,
        'mean_eff_vocab': sum(all_eff)     / n,
        'entropy_pct': {
            'p10': pct(all_entropy, 10), 'p25': pct(all_entropy, 25),
            'p50': pct(all_entropy, 50), 'p75': pct(all_entropy, 75),
            'p90': pct(all_entropy, 90),
        },
    }
    return {'per_prompt': prompt_results, 'aggregate': agg}


# ─── 2. DISTANCE-CONDITIONED LOSS ─────────────────────────────────────────────

BUCKETS = [
    ('0-16',      0,    16),
    ('17-64',     17,   64),
    ('65-256',    65,  256),
    ('257-512',  257,  512),
    ('513-1024', 513, 1024),
    ('1025-2047', 1025, 2047),
]

DIST_PARAGRAPHS = [
    # Quantum mechanics
    "Quantum mechanics is the branch of physics that describes the behavior of particles at the atomic and subatomic level. Unlike classical mechanics, which predicts deterministic outcomes, quantum mechanics operates on probabilities. The central equation of quantum mechanics is the Schrödinger equation, which describes how the quantum state of a physical system evolves over time. A quantum state is represented by a wave function, a mathematical object that encodes all possible outcomes of measurements on the system. When a measurement is made, the wave function collapses to a specific eigenstate corresponding to the measured value. This probabilistic nature led Einstein to famously object that 'God does not play dice,' though subsequent experiments confirmed quantum predictions with extraordinary precision.",
    # Roman Republic
    "The Roman Republic, which lasted from approximately 509 BCE to 27 BCE, was one of the most consequential political experiments in human history. Emerging from the overthrow of the Tarquin kings, the Republic established a system of government designed to prevent any single individual from accumulating too much power. The two consuls who served as chief executives were elected annually and each held the power of veto over the other. The Senate, originally an advisory body composed of former magistrates and aristocrats, gradually accumulated enormous influence over foreign policy, finances, and the appointment of provincial governors.",
    # Neural networks
    "Neural networks are computational systems loosely inspired by the structure of biological neural networks in animal brains. A neural network consists of layers of interconnected nodes, each of which performs a simple computation on its inputs and passes the result to subsequent layers. The power of neural networks arises from the composition of many such simple computations and from the adjustment of connection weights through training. Deep learning refers to neural networks with many hidden layers between the input and output. The depth allows the network to learn hierarchical representations of data, with early layers detecting simple features and later layers combining these into increasingly abstract representations.",
    # Climate
    "Climate change refers to long-term shifts in global temperatures and weather patterns. While some climate change is natural, since the 1800s human activities have been the main driver of climate change, primarily due to burning fossil fuels like coal, oil and gas. Burning fossil fuels generates greenhouse gas emissions that act like a blanket wrapped around the Earth, trapping the sun's heat and raising temperatures. Carbon dioxide and methane are among the greenhouse gases that are most responsible for climate change. These come from using gasoline for driving a car or coal for heating a building, for example.",
    # Economics
    "The gross domestic product measures the monetary value of final goods and services produced in a country in a given period of time. GDP is used to compare the economic output of countries across time and across national boundaries. While GDP is the most widely used measure of economic activity, it has well-known limitations: it does not capture income distribution, environmental sustainability, or unpaid work such as household labor. Economists have developed supplementary measures including GNP, which measures income earned by a country's residents regardless of where production occurs, and GNI, which adjusts for international transfers of income.",
]


def bucket_of(pos):
    for name, lo, hi in BUCKETS:
        if lo <= pos <= hi:
            return name
    return None


@torch.no_grad()
def compute_per_token_loss(model, token_ids, device):
    ids    = token_ids.unsqueeze(0).to(device)
    logits = model(ids)
    losses = F.cross_entropy(logits[0, :-1, :], ids[0, 1:], reduction='none')
    return losses.cpu().tolist()


def eval_distance_loss(model, tokenizer, device):
    model.eval()
    all_ids = []
    for para in DIST_PARAGRAPHS:
        all_ids.extend(tokenizer.encode(para))

    # Build chunks
    chunks = []
    for start in range(0, len(all_ids) - MAX_SEQ_LEN, MAX_SEQ_LEN):
        chunk = all_ids[start: start + MAX_SEQ_LEN]
        if len(chunk) == MAX_SEQ_LEN:
            chunks.append(torch.tensor(chunk, dtype=torch.long))
    if not chunks and len(all_ids) >= 64:
        chunks = [torch.tensor(all_ids[:min(len(all_ids), MAX_SEQ_LEN)], dtype=torch.long)]

    bucket_losses = {name: [] for name, _, _ in BUCKETS}

    for chunk in chunks:
        losses = compute_per_token_loss(model, chunk, device)
        for i, loss in enumerate(losses):
            bname = bucket_of(i)
            if bname:
                bucket_losses[bname].append(loss)

    results = {}
    for name, lo, hi in BUCKETS:
        bl = bucket_losses[name]
        if not bl:
            results[name] = {'mean_loss': None, 'mean_ppl': None, 'n': 0,
                             'range': (lo, hi)}
            continue
        ml = sum(bl) / len(bl)
        results[name] = {'mean_loss': ml, 'mean_ppl': math.exp(ml),
                         'n': len(bl), 'range': (lo, hi)}
    return results


# ─── 3. FEW-SHOT STRING COPY ──────────────────────────────────────────────────

FEW_SHOT_CASES = [
    {'name': 'alphanumeric_3shot',
     'prompt': 'Input: XKCD → Output: XKCD\nInput: 7829 → Output: 7829\nInput: QWRT → Output:',
     'expected': 'QWRT', 'max_new': 15},
    {'name': 'number_2shot',
     'prompt': '5 → 5\n12 → 12\n847 → ',
     'expected': '847', 'max_new': 10},
    {'name': 'word_copy_at_distance',
     'prompt': 'The password is ZEPHYR. What is the password? The password is',
     'expected': 'ZEPHYR', 'max_new': 10},
    {'name': 'uppercase_3shot',
     'prompt': 'alpha → ALPHA\nbanana → BANANA\ncherry → CHERRY\ndragon → ',
     'expected': 'DRAGON', 'max_new': 15},
    {'name': 'pipe_copy_2shot',
     'prompt': 'cat|cat\ndog|dog\nbird|',
     'expected': 'bird', 'max_new': 10},
    {'name': 'mixed_alphanum_2shot',
     'prompt': 'X1X → X1X\nY2Y → Y2Y\nZ3Z → ',
     'expected': 'Z3Z', 'max_new': 10},
    {'name': 'repeated_keyword_recall',
     'prompt': 'Remember this code: ALPHA7. The code is ALPHA7. Never forget: ALPHA7. What is the code?',
     'expected': 'ALPHA7', 'max_new': 15},
    {'name': 'equals_copy_2shot',
     'prompt': 'red=red, blue=blue, green=',
     'expected': 'green', 'max_new': 10},
    {'name': 'code_copy_4shot',
     'prompt': 'CODE: A3F → A3F\nCODE: B7X → B7X\nCODE: C2P → C2P\nCODE: D9K → D9K\nCODE: E5M → ',
     'expected': 'E5M', 'max_new': 10},
    {'name': 'position_indexed_copy',
     'prompt': '[1] apple\n[2] banana\n[3] cherry\n[1]',
     'expected': 'apple', 'max_new': 15},
]


def levenshtein(a, b):
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        new_dp = [i] + [0] * lb
        for j in range(1, lb + 1):
            new_dp[j] = dp[j-1] if a[i-1] == b[j-1] else 1 + min(dp[j], new_dp[j-1], dp[j-1])
        dp = new_dp
    return dp[lb]


def eval_few_shot_copy(model, tokenizer, device, temperature=0.1, top_p=0.9):
    model.eval()
    results = []
    for case in FEW_SHOT_CASES:
        gen_ids = generate_tokens(model, tokenizer, case['prompt'], device,
                                   max_new=case['max_new'],
                                   temperature=temperature, top_p=top_p)
        gen_text  = tokenizer.decode(gen_ids)
        # Extract first token
        clean = gen_text.lstrip(' \n\r\t')
        parts = clean.split()
        pred  = parts[0].rstrip('.,;:!?\n\r') if parts else ''
        exp   = case['expected']
        results.append({
            'name':         case['name'],
            'expected':     exp,
            'predicted':    pred,
            'verbatim':     gen_text[:150],
            'exact_match':  pred == exp,
            'prefix_match': pred[:len(exp)] == exp,
            'edit_dist':    levenshtein(pred, exp),
        })
    n     = len(results)
    exact = sum(r['exact_match']  for r in results)
    prefix= sum(r['prefix_match'] for r in results)
    return {
        'cases':          results,
        'exact_match':    exact,
        'prefix_match':   prefix,
        'total':          n,
        'exact_rate':     exact / n,
        'prefix_rate':    prefix / n,
        'mean_edit_dist': sum(r['edit_dist'] for r in results) / n,
    }


# ─── 4. PASSKEY RETRIEVAL ─────────────────────────────────────────────────────

PASSKEY_WORDS = [
    'apple', 'tiger', 'robot', 'ocean', 'piano',
    'eagle', 'storm', 'river', 'crown', 'flame',
]
FILLER_SENTENCE = 'the weather was mild and the air was still . '
INTRO_TEMPLATE  = 'the secret word is {word} .'
RETRIEVAL_CUE   = 'the secret word is'
TEST_DISTANCES  = [0, 5, 16, 32, 48, 50, 64, 96, 100, 128, 192, 200, 256,
                   300, 384, 500, 512, 750, 768, 1000, 1024, 1250, 1500, 1536, 1700]


@torch.no_grad()
def eval_passkey_at_distance(model, tokenizer, distance, device, n_trials=5):
    filler_ids = tokenizer.encode(FILLER_SENTENCE)
    cue_ids    = tokenizer.encode(RETRIEVAL_CUE)

    correct = 0
    scores  = []

    for i in range(n_trials):
        word     = PASSKEY_WORDS[i % len(PASSKEY_WORDS)]
        intro_ids = tokenizer.encode(INTRO_TEMPLATE.format(word=word))

        # Build filler of exact length
        available = MAX_SEQ_LEN - 1 - len(intro_ids) - len(cue_ids) - 1
        if distance > available:
            return None  # Doesn't fit

        filler = []
        while len(filler) < distance:
            filler.extend(filler_ids)
        filler = filler[:distance]

        full_seq = intro_ids + filler + cue_ids
        if len(full_seq) >= MAX_SEQ_LEN:
            return None

        x = torch.tensor([full_seq], dtype=torch.long, device=device)
        logits = model(x)
        last   = logits[0, -1]

        # First token of the passkey word (try with space prefix first)
        word_ids = tokenizer.encode(' ' + word) or tokenizer.encode(word)
        target   = word_ids[0]
        log_p    = F.log_softmax(last, dim=-1)

        # Distractor accuracy: 10-way choice (target word vs 9 other passkey words)
        distractors = [w for w in PASSKEY_WORDS if w != word][:9]
        all_words   = [word] + distractors
        logit_vals  = []
        for w in all_words:
            wids = tokenizer.encode(' ' + w) or tokenizer.encode(w)
            logit_vals.append(last[wids[0]].item())
        if logit_vals.index(max(logit_vals)) == 0:
            correct += 1
        scores.append(log_p[target].item())

    return {
        'distance':   distance,
        'accuracy':   correct / n_trials,
        'mean_log_p': sum(scores) / len(scores),
        'n_trials':   n_trials,
    }


def eval_passkey(model, tokenizer, device):
    model.eval()
    results = []
    for d in TEST_DISTANCES:
        r = eval_passkey_at_distance(model, tokenizer, d, device)
        if r is not None:
            results.append(r)

    accs = [r['accuracy'] for r in results]
    return {
        'per_distance':  results,
        'mean_accuracy': sum(accs) / len(accs) if accs else 0.0,
        'n_distances':   len(results),
    }


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(model_class, checkpoint_path, device):
    model = model_class().to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'condM checkpoint:    {CONDM_CKPT}')
    print(f'condP 27M checkpoint: {CONDP27_CKPT}')

    # Check checkpoints
    for p in [CONDM_CKPT, CONDP27_CKPT, TOKENIZER]:
        if not os.path.exists(p):
            print(f'ERROR: not found: {p}'); sys.exit(1)

    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(TOKENIZER))
    print(f'Tokenizer: {tokenizer.vocab_size()} tokens\n')

    # Load models
    print('Loading condM (13M, 5:1 hybrid)...')
    condm = load_model(CondMTransformer, CONDM_CKPT, device)
    print(f'  condM: {condm.param_count():,} params')

    print('Loading condP 27M (pure DSQG)...')
    condp27 = load_model(CondP27MTransformer, CONDP27_CKPT, device)
    print(f'  condP 27M: {condp27.param_count():,} params')

    results = {
        'meta': {
            'date': datetime.datetime.now().isoformat(),
            'condm_checkpoint':  CONDM_CKPT,
            'condp27_checkpoint': CONDP27_CKPT,
            'condm_params':  condm.param_count(),
            'condp27_params': condp27.param_count(),
            'condm_test_ppl':  54.529,
            'condp27_test_ppl': 52.756,
        }
    }

    # ── 1. Calibration ──
    print('\n' + '='*72)
    print('  1. CALIBRATION (logit entropy, top-1 confidence)')
    print('='*72)
    for name, model in [('condM', condm), ('condP_27M', condp27)]:
        print(f'\n  Running calibration: {name}...')
        t0 = time.time()
        r  = eval_calibration(model, tokenizer, device)
        elapsed = time.time() - t0
        agg = r['aggregate']
        print(f'  {name} ({elapsed:.1f}s):')
        print(f'    Entropy (bits): mean={agg["mean_entropy"]:.3f}')
        print(f'    Top-1 prob:     mean={agg["mean_top1_prob"]:.4f}')
        print(f'    Top-5 mass:     mean={agg["mean_top5_mass"]:.4f}')
        print(f'    Eff vocab:      mean={agg["mean_eff_vocab"]:.1f}')
        ep = agg['entropy_pct']
        print(f'    Entropy pct:    p10={ep["p10"]:.2f} p25={ep["p25"]:.2f} '
              f'p50={ep["p50"]:.2f} p75={ep["p75"]:.2f} p90={ep["p90"]:.2f}')
        results[f'calibration_{name}'] = r

    # Comparison
    cm  = results['calibration_condM']['aggregate']
    cp  = results['calibration_condP_27M']['aggregate']
    d_ent  = cm['mean_entropy']   - cp['mean_entropy']
    d_top1 = cm['mean_top1_prob'] - cp['mean_top1_prob']
    print(f'\n  ── Calibration delta (condM − condP27M) ──')
    print(f'    Entropy:   {d_ent:+.3f} bits  (positive = condM more uncertain)')
    print(f'    Top-1 prob:{d_top1:+.4f}      (negative = condM less peaked)')
    print(f'    Eff vocab: {cm["mean_eff_vocab"]-cp["mean_eff_vocab"]:+.1f}')

    # ── 2. Distance-conditioned loss ──
    print('\n' + '='*72)
    print('  2. DISTANCE-CONDITIONED LOSS (per-position CE bucketed by lookback)')
    print('='*72)
    for name, model in [('condM', condm), ('condP_27M', condp27)]:
        print(f'\n  Running distance loss: {name}...')
        t0 = time.time()
        r  = eval_distance_loss(model, tokenizer, device)
        elapsed = time.time() - t0
        print(f'  {name} ({elapsed:.1f}s):')
        print(f'    {"Bucket":<14} {"Mean PPL":>10} {"n tokens":>10}')
        for bname, _, _ in BUCKETS:
            br = r[bname]
            if br['mean_ppl'] is not None:
                print(f'    {bname:<14} {br["mean_ppl"]:>10.2f} {br["n"]:>10}')
        results[f'distance_{name}'] = r

    # Comparison
    print(f'\n  ── Distance-loss delta (condM − condP27M), PPL ──')
    print(f'    {"Bucket":<14} {"condM PPL":>10} {"condP27 PPL":>11} {"Δ PPL":>10}')
    for bname, _, _ in BUCKETS:
        cm_r  = results['distance_condM'][bname]
        cp_r  = results['distance_condP_27M'][bname]
        if cm_r['mean_ppl'] and cp_r['mean_ppl']:
            delta = cm_r['mean_ppl'] - cp_r['mean_ppl']
            print(f'    {bname:<14} {cm_r["mean_ppl"]:>10.2f} '
                  f'{cp_r["mean_ppl"]:>11.2f} {delta:>+10.2f}')

    # ── 3. Few-shot copy ──
    print('\n' + '='*72)
    print('  3. FEW-SHOT STRING COPY (template following / induction heads)')
    print('='*72)
    for name, model in [('condM', condm), ('condP_27M', condp27)]:
        print(f'\n  Running few-shot copy: {name}...')
        t0 = time.time()
        r  = eval_few_shot_copy(model, tokenizer, device)
        elapsed = time.time() - t0
        print(f'  {name} ({elapsed:.1f}s):  '
              f'exact={r["exact_match"]}/{r["total"]} ({r["exact_rate"]*100:.0f}%)  '
              f'prefix={r["prefix_match"]}/{r["total"]} ({r["prefix_rate"]*100:.0f}%)  '
              f'mean_edit={r["mean_edit_dist"]:.2f}')
        for case in r['cases']:
            em = '✓' if case['exact_match'] else '✗'
            print(f'    {em} {case["name"]:<35} exp={case["expected"]!r:<8} '
                  f'got={case["predicted"]!r}')
        results[f'few_shot_{name}'] = r

    # Comparison
    cm_fs = results['few_shot_condM']
    cp_fs = results['few_shot_condP_27M']
    print(f'\n  ── Few-shot summary ──')
    print(f'    condM:     {cm_fs["exact_match"]}/{cm_fs["total"]} exact  '
          f'({cm_fs["exact_rate"]*100:.0f}%)  mean_edit={cm_fs["mean_edit_dist"]:.2f}')
    print(f'    condP 27M: {cp_fs["exact_match"]}/{cp_fs["total"]} exact  '
          f'({cp_fs["exact_rate"]*100:.0f}%)  mean_edit={cp_fs["mean_edit_dist"]:.2f}')

    # ── 4. Passkey retrieval ──
    print('\n' + '='*72)
    print('  4. PASSKEY RETRIEVAL (content-addressed memory at distance)')
    print('='*72)
    for name, model in [('condM', condm), ('condP_27M', condp27)]:
        print(f'\n  Running passkey: {name}...')
        t0 = time.time()
        r  = eval_passkey(model, tokenizer, device)
        elapsed = time.time() - t0
        print(f'  {name} ({elapsed:.1f}s):  mean_accuracy={r["mean_accuracy"]*100:.1f}%  '
              f'over {r["n_distances"]} distances')
        print(f'    {"Distance":>10}  {"Accuracy":>10}  {"Mean logP":>10}')
        for pr in r['per_distance']:
            print(f'    {pr["distance"]:>10}  {pr["accuracy"]*100:>9.0f}%  '
                  f'{pr["mean_log_p"]:>10.3f}')
        results[f'passkey_{name}'] = r

    # Comparison
    cm_pk = results['passkey_condM']
    cp_pk = results['passkey_condP_27M']
    print(f'\n  ── Passkey summary ──')
    print(f'    condM:     mean_accuracy={cm_pk["mean_accuracy"]*100:.1f}%')
    print(f'    condP 27M: mean_accuracy={cp_pk["mean_accuracy"]*100:.1f}%')
    # Per-distance comparison
    cm_by_d  = {r['distance']: r for r in cm_pk['per_distance']}
    cp_by_d  = {r['distance']: r for r in cp_pk['per_distance']}
    print(f'\n    {"Dist":>6}  {"condM acc":>10}  {"condP27 acc":>12}')
    for d in sorted(set(cm_by_d) & set(cp_by_d)):
        print(f'    {d:>6}  {cm_by_d[d]["accuracy"]*100:>9.0f}%  '
              f'{cp_by_d[d]["accuracy"]*100:>11.0f}%')

    # ── Save ──
    ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(LOGS_DIR, f'eval_condM_vs_condP27m_{ts}.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n\nResults saved to: {out}')

    # ── Final summary ──
    print('\n' + '='*72)
    print('  FINAL COMPARISON SUMMARY')
    print('='*72)
    print(f'  {"Metric":<35}  {"condM (13M)":>14}  {"condP 27M":>12}')
    print(f'  {"-"*35}  {"-"*14}  {"-"*12}')
    print(f'  {"Test PPL":<35}  {"54.529":>14}  {"52.756":>12}')
    print(f'  {"Params":<35}  {condm.param_count():>14,}  {condp27.param_count():>12,}')
    print(f'  {"Calibration: mean entropy (bits)":<35}  '
          f'{cm["mean_entropy"]:>14.3f}  {cp["mean_entropy"]:>12.3f}')
    print(f'  {"Calibration: top-1 prob":<35}  '
          f'{cm["mean_top1_prob"]:>14.4f}  {cp["mean_top1_prob"]:>12.4f}')
    print(f'  {"Few-shot exact match":<35}  '
          f'{cm_fs["exact_match"]}/{cm_fs["total"]} ({cm_fs["exact_rate"]*100:.0f}%)'
          f'{"":>4}  '
          f'{cp_fs["exact_match"]}/{cp_fs["total"]} ({cp_fs["exact_rate"]*100:.0f}%)')
    print(f'  {"Passkey mean accuracy":<35}  '
          f'{cm_pk["mean_accuracy"]*100:>13.1f}%  '
          f'{cp_pk["mean_accuracy"]*100:>11.1f}%')


if __name__ == '__main__':
    main()
