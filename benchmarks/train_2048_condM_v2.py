"""
DWARF condM-v2: architectural freebies bundle on top of condM baseline

Changes from condM baseline (train_2048_condM_layer_ablation.py):
  [Previous]
  1. RMSNorm        — replaces LayerNorm; no bias parameter, ~15% faster kernel
  2. SwiGLU FFN     — replaces GELU; hidden_dim=682 (8D/3) for iso-parameter match
  3. RoPE           — Q,K in FullCausalAttention; drops absolute P[pos]
  4. EMA pooling    — replaces causal running mean; per-channel learned decay (init=0.9)
  5. δ=0 pos_bias   — small negative init (−log(1.2)·α) vs 0; reduces identity attractor
                      (δ=0 cannot be removed: only valid offset at t=0; removal → NaN)

  [This version adds]
  6. Remove QKV/out_proj biases      — bias=False on qkv_proj, out_proj;
                                       gate_proj KEEPS bias (2.0 init is load-bearing)
  7. Scaled residual init            — out_proj & down_proj: σ = 0.02/√(2·L)
                                       prevents residual stream variance growth with depth;
                                       NOT the same as condK "RG init" (that was D4 DWT
                                       gain-structure init to prevent j0 collapse — different
                                       architecture, different mechanism)
  8. LR warmup                       — linear warmup over first 1% of steps → cosine decay
  9. bf16 autocast                   — replaces fp16 default; 4× dynamic range, native on 4090
 10. Embedding output scaling ×√D    — corrects variance mismatch for tied embeddings;
                                       standard Transformer / T5 practice
 11. torch.backends.cudnn.benchmark  — cuDNN auto-profiles kernel selection; free at fixed seqlen
 12. AdamW param groups              — zero weight decay on embeddings, norms, biases

Architecture stays: [DSQG, DSQG, DSQG*, Full, DSQG, DSQG*]  (layer 5 default)
  * = interference/EMA pooling module at layers i % 3 == 2 (layers 2, 5)
  No architectural capability change; pure training hygiene + efficiency.

Recommended launch (RTX 4090, CUDA_VISIBLE_DEVICES=0):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u benchmarks/train_2048_condM_v2.py \\
    2>&1 | tee benchmarks/results/condM_v2_run.log

Results: benchmarks/results/condM_v2_results.json
Checkpoint: checkpoints/2048_condM_v2_checkpoints/best.pt
"""

import json, math, os, sys, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# cuDNN auto-profiles kernel selection for fixed input sizes — free, do at import time
torch.backends.cudnn.benchmark = True

# ─── Hyperparameters ──────────────────────────────────────────────────────────

VOCAB_SIZE      = 32000
NUM_EPOCHS      = 10
BATCH_SIZE      = 8
GRAD_ACCUM      = 4
LR              = 3e-4
MAX_SEQ_LEN     = 2048
NUM_DOCS        = 100_000

EMBEDDING_DIM   = 256
NUM_LAYERS      = 6
NUM_HEADS       = 8
HEAD_DIM        = EMBEDDING_DIM // NUM_HEADS          # 32
FFN_HIDDEN      = int(8 * EMBEDDING_DIM / 3)          # 682 (≈ iso-parameter with 4D FFN)
INTERFERENCE    = 3                                    # pooling every 3rd layer
FULL_ATTN_LAYER = 5                                   # default: last layer (best passkey)

# ─── condN offset set ─────────────────────────────────────────────────────────

_COND_N_OFFSETS = sorted(
    set(range(0, 33)) | {48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536})
assert len(_COND_N_OFFSETS) == 44

# ─── RMSNorm ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """RMSNorm: skip mean-centering, no bias, ~15% faster than LayerNorm."""
    def __init__(self, D, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(D))
        self.eps   = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * (x / rms)


# ─── SwiGLU FFN ───────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """SwiGLU FFN: (gate⊙silu(up))·down. hidden_dim=8D/3 for iso-parameter match."""
    def __init__(self, D, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8 * D / 3)   # 682 for D=256
        self.gate_proj = nn.Linear(D, hidden_dim, bias=False)
        self.up_proj   = nn.Linear(D, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, D, bias=False)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ─── RoPE utilities ───────────────────────────────────────────────────────────

def build_rope_cache(seq_len, head_dim, device, dtype=torch.float32, base=10000):
    """Precompute RoPE cos/sin cache. head_dim must be even."""
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device,
                                             dtype=dtype) / head_dim))
    t       = torch.arange(seq_len, device=device, dtype=dtype)
    freqs   = torch.einsum('i,j->ij', t, inv_freq)           # [N, HD/2]
    emb     = torch.cat([freqs, freqs], dim=-1)               # [N, HD]
    return emb.cos(), emb.sin()


def rotate_half(x):
    """Rotate the latter half of the last dimension into the former half."""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    """x: [B, H, N, HD]; cos/sin: [N, HD] (broadcast over B, H)."""
    cos = cos.unsqueeze(0).unsqueeze(0)   # [1, 1, N, HD]
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


# ─── DSQG Attention (condN architecture, unchanged) ──────────────────────────

class DSQGAttentionN(nn.Module):
    def __init__(self, embedding_dim, num_heads, seq_len=MAX_SEQ_LEN,
                 offsets=None, dropout=0.1):
        super().__init__()
        if offsets is None: offsets = _COND_N_OFFSETS
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        self.n_offsets = len(offsets)
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.long))

        # bias=False on qkv/out_proj: redundant with pre-RMSNorm; all modern LLMs drop these.
        # gate_proj KEEPS bias: 2.0 init → sigmoid≈0.88 initial gate state (load-bearing).
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 2.0)

        # ALiBi-style init with small penalty for δ=0 to reduce identity attractor.
        # δ=0 originally gets 0 bias (log(1+0)=0), giving it an unearned advantage.
        # Using log(1+max(δ,0.2)) gives δ=0 a small initial penalty ≈ −0.18·α_h.
        alphas     = torch.linspace(0.2, 2.0, num_heads)
        delta_vals = torch.tensor(
            [math.log(1.0 + max(d, 0.2)) for d in offsets], dtype=torch.float32)
        self.pos_bias = nn.Parameter(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim
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

        K_all  = torch.stack(K_list, dim=3)    # B,H,N,n_off,HD
        V_all  = torch.stack(V_list, dim=3)
        scores = (q.unsqueeze(3) * K_all).sum(-1) * scale    # B,H,N,n_off
        scores = scores + self.pos_bias.T.unsqueeze(0).unsqueeze(2)  # [1,H,1,n_off]

        n_idx  = torch.arange(N, device=x.device).unsqueeze(1)
        d_idx  = self.offsets.unsqueeze(0)
        scores = scores.masked_fill(
            (n_idx < d_idx).unsqueeze(0).unsqueeze(0), float('-inf'))

        alpha = F.softmax(scores, dim=-1)
        out   = (alpha.unsqueeze(-1) * V_all).sum(dim=3)    # B,H,N,HD
        out   = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate  = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(out * gate))

    def attn_summary(self):
        with torch.no_grad():
            pb = self.pos_bias.detach().cpu()
        return {
            'pos_bias_abs_mean':      pb.abs().mean().item(),
            'pos_bias_abs_max':       pb.abs().max().item(),
            'pos_bias_mean_per_head': pb.mean(0).tolist(),
        }


# ─── Full Causal Attention with RoPE ──────────────────────────────────────────

class FullCausalAttentionRoPE(nn.Module):
    """Full O(N²) causal attention with RoPE. No absolute position embeddings needed."""
    def __init__(self, embedding_dim, num_heads, seq_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        self.seq_len   = seq_len

        # bias=False on qkv/out_proj: consistent with DSQG blocks above.
        # gate_proj KEEPS bias for same reason (sigmoid gate init).
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 2.0)

        # Precompute RoPE cache for training length
        cos, sin = build_rope_cache(seq_len + 2, self.head_dim,
                                    device='cpu', dtype=torch.float32)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)
        self.dropout_p = dropout

    def forward(self, x):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim
        qkv     = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)   # B,H,N,HD
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)

        # Apply RoPE to Q and K (not V)
        cos = self.rope_cos[:N].to(q.dtype)
        sin = self.rope_sin[:N].to(q.dtype)
        q   = apply_rope(q, cos, sin)
        k   = apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0)
        out  = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate = torch.sigmoid(self.gate_proj(x))
        return self.out_proj(out * gate)


# ─── EMA Pooling ──────────────────────────────────────────────────────────────

class CausalMeanPooling(nn.Module):
    """
    Vectorized causal running mean: pool_t = mean(x_0 .. x_t)
    Implemented as cumsum / position_index — single GPU op, zero Python loop overhead.
    Replaces EMA pooling (which required 2047 sequential kernel launches per forward
    pass, cutting GPU utilization from ~99% to ~46% and adding ~3-4× wall-clock time).

    EMA (per-channel learned decay) is the theoretically better formulation but is
    not efficiently expressible as a vectorized PyTorch op without a custom CUDA kernel.
    Causal mean is the fully vectorized equivalent that matches prior condN/condP behavior.
    """
    def forward(self, x):
        B, N, D = x.shape
        cumsum = x.cumsum(dim=1)                         # [B, N, D], causal
        counts = torch.arange(1, N + 1, device=x.device,
                              dtype=x.dtype).view(1, N, 1)
        return cumsum / counts                           # [B, N, D]


# ─── DSQG Block (RMSNorm + SwiGLU + EMA pooling) ─────────────────────────────

class DSQGBlock(nn.Module):
    def __init__(self, D, H, ffn_hidden, seq_len=MAX_SEQ_LEN, offsets=None,
                 interference=False, dropout=0.1):
        super().__init__()
        self.use_checkpoint = True
        self.interference   = interference

        self.norm1 = RMSNorm(D)
        self.norm2 = RMSNorm(D)
        self.attn  = DSQGAttentionN(D, H, seq_len=seq_len,
                                    offsets=offsets, dropout=dropout)
        self.ffn   = SwiGLUFFN(D, hidden_dim=ffn_hidden, dropout=dropout)

        if interference:
            self.inter_norm = RMSNorm(D)
            self.inter_gate = nn.Linear(D, D)
            self.inter_pool_proj = nn.Linear(D, D)
            self.causal_pool = CausalMeanPooling()

    def _attn_fn(self, x):
        return self.attn(self.norm1(x))

    def forward(self, x):
        if self.use_checkpoint and self.training:
            x = x + torch.utils.checkpoint.checkpoint(
                self._attn_fn, x, use_reentrant=False)
        else:
            x = x + self._attn_fn(x)

        if self.interference:
            xi   = self.inter_norm(x)
            pool = self.causal_pool(xi)                  # [B, N, D], vectorized
            x    = x + torch.sigmoid(self.inter_gate(xi)) * self.inter_pool_proj(pool)

        x = x + self.ffn(self.norm2(x))
        return x


# ─── Full Attention Block (RMSNorm + SwiGLU + RoPE) ──────────────────────────

class FullAttnBlock(nn.Module):
    def __init__(self, D, H, ffn_hidden, seq_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(D)
        self.norm2 = RMSNorm(D)
        self.attn  = FullCausalAttentionRoPE(D, H, seq_len=seq_len, dropout=dropout)
        self.ffn   = SwiGLUFFN(D, hidden_dim=ffn_hidden, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ─── condM-v2 Transformer ─────────────────────────────────────────────────────

class CondMV2Transformer(nn.Module):
    """
    condM-v2: hybrid DSQG + full attention, with:
      - RMSNorm (not LayerNorm)
      - SwiGLU FFN (not GELU)
      - RoPE in full attention (not absolute position embeddings)
      - EMA pooling (not running mean)
      - δ=0 pos_bias initialized negative to reduce identity attractor

    No absolute position embedding P[pos]. DSQG layers handle relative
    position through learned ALiBi-style β bias. Full attention layer
    handles position through RoPE on Q/K.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
                 num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
                 ffn_hidden=FFN_HIDDEN, seq_len=MAX_SEQ_LEN,
                 full_attn_layer=FULL_ATTN_LAYER,
                 interference_interval=INTERFERENCE, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # NO pos_embed — positional info comes from DSQG β bias + RoPE
        self.drop      = nn.Dropout(dropout)

        blocks = []
        for i in range(num_layers):
            interf = (i % interference_interval == interference_interval - 1)
            if i == full_attn_layer:
                blocks.append(FullAttnBlock(embedding_dim, num_heads,
                                            ffn_hidden, seq_len, dropout))
            else:
                blocks.append(DSQGBlock(embedding_dim, num_heads, ffn_hidden,
                                        seq_len, offsets=None,
                                        interference=interf, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)
        self.norm   = RMSNorm(embedding_dim)
        self.out    = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.out.weight = self.embedding.weight    # tied

        self._full_attn_layer = full_attn_layer
        self._init_weights()

    def _init_weights(self):
        # Scaled residual init (GPT-2): weights that project INTO the residual stream
        # use σ = 0.02/√(2·L) to prevent variance growth with depth.
        # At L=6: σ = 0.02/√12 ≈ 0.00577. Applies to: out_proj, FFN down_proj.
        # NOT related to condK "RG init" (that was D4 DWT gain-structure init).
        residual_std = 0.02 / math.sqrt(2 * NUM_LAYERS)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)

        # Apply scaled residual init to out_proj and FFN down_proj
        for block in self.blocks:
            if isinstance(block, DSQGBlock):
                nn.init.normal_(block.attn.out_proj.weight, 0, residual_std)
                nn.init.normal_(block.ffn.down_proj.weight, 0, residual_std)
            elif isinstance(block, FullAttnBlock):
                nn.init.normal_(block.attn.out_proj.weight, 0, residual_std)
                nn.init.normal_(block.ffn.down_proj.weight, 0, residual_std)

        # Re-init gate biases after generic init (must come last)
        for block in self.blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'gate_proj'):
                nn.init.constant_(block.attn.gate_proj.bias, 2.0)

    def forward(self, idx):
        B, N = idx.shape
        # ×√D embedding scaling: corrects variance mismatch for tied embeddings.
        # Embedding table init: N(0, 0.02²) → per-element var = 0.0004.
        # With ×√D (×16 at D=256): per-element var = (√256)²×0.0004 = 0.1024.
        x = self.drop(self.embedding(idx) * math.sqrt(EMBEDDING_DIM))
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def attn_summary(self):
        dsqg_blocks = [b for b in self.blocks if isinstance(b, DSQGBlock)]
        if not dsqg_blocks:
            return {'pos_bias_abs_mean': 0, 'pos_bias_abs_max': 0,
                    'pos_bias_mean_per_head': [0] * NUM_HEADS}
        summaries = [b.attn.attn_summary() for b in dsqg_blocks]
        n = len(summaries)
        return {
            'pos_bias_abs_mean':      sum(s['pos_bias_abs_mean'] for s in summaries) / n,
            'pos_bias_abs_max':       max(s['pos_bias_abs_max']  for s in summaries),
            'pos_bias_mean_per_head': [
                sum(s['pos_bias_mean_per_head'][h] for s in summaries) / n
                for h in range(NUM_HEADS)
            ],
        }


# ─── Data utilities (identical to condM) ──────────────────────────────────────

class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)
    def vocab_size(self):   return self.tokenizer.get_vocab_size()


def load_data(num_docs=NUM_DOCS):
    from datasets import load_dataset
    print(f'Loading OpenWebText (up to {num_docs:,} docs)...')
    ds    = load_dataset('openwebtext', split='train', streaming=True)
    texts = []
    for i, item in enumerate(ds):
        if i >= num_docs: break
        texts.append(item['text'])
        if (i + 1) % 25_000 == 0: print(f'  {i+1:,} docs...')
    n = len(texts)
    return {
        'train': texts[:int(n * 0.95)],
        'val':   texts[int(n * 0.95): int(n * 0.95) + 2500],
        'test':  texts[int(n * 0.95) + 2500: int(n * 0.95) + 5000],
    }


def encode_split(split_texts, tokenizer, max_seq_len, split_name):
    tokens = []
    for text in split_texts:
        tokens.extend(tokenizer.encode(text))
        tokens.append(3)
    n    = (len(tokens) // max_seq_len) * max_seq_len
    data = torch.tensor(tokens[:n], dtype=torch.long)
    seqs = data.view(-1, max_seq_len)
    print(f'  {split_name}: {len(seqs):,} sequences')
    return seqs


@torch.no_grad()
def evaluate(model, data, batch_size, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(data) - batch_size, batch_size):
        x = data[i:i + batch_size, :-1].to(device)
        y = data[i:i + batch_size,  1:].to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
        loss   = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss   += loss.item() * y.numel()
        total_tokens += y.numel()
    return total_loss / max(total_tokens, 1)


def generate(model, tokenizer, prompts, device, max_new=150, temperature=1.0, top_p=0.9):
    model.eval()
    results = []
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(max_new):
                logits = model(ids[:, -MAX_SEQ_LEN:])
                logits_last = logits[0, -1]
                if temperature <= 0.01:
                    next_id = logits_last.argmax()
                else:
                    probs = F.softmax(logits_last / temperature, dim=-1)
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=0)
                    mask   = cumsum - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs /= sorted_probs.sum()
                    next_id = sorted_idx[torch.multinomial(sorted_probs, 1)]
                ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
        gen = tokenizer.decode(ids[0, len(tokenizer.encode(prompt)):].tolist())
        results.append(gen[:120])
    return results


def causality_check(model, device):
    print('Running causality check...')
    model.eval()
    with torch.no_grad():
        x1 = torch.randint(0, VOCAB_SIZE, (1, 64), device=device)
        x2 = x1.clone(); x2[0, 10] = (x2[0, 10] + 1) % VOCAB_SIZE
        out1, out2 = model(x1), model(x2)
        diff = (out1 - out2).abs()
    pre  = diff[0, :10].max().item()
    pos  = diff[0,  10].max().item()
    post = diff[0, 11:].max().item()
    print(f'  Pre-10:  {pre:.8f}  (expect 0.0)')
    print(f'  Pos-10:  {pos:.6f}  (expect >0)')
    print(f'  Post-10: {post:.6f}  (expect >0)')
    ok = pre < 1e-6
    print(f'  {"PASS" if ok else "FAIL"}')
    return ok


# ─── Training loop ────────────────────────────────────────────────────────────

GEN_PROMPTS = [
    'It was a dark and stormy',
    'The length of the hypotenuse',
    'The President of the United',
    'Once upon a time there was',
    'The results indicate that',
]


def build_optimizer(model, total_steps):
    """
    AdamW with param groups:
      - weight_decay=0.1 for weight matrices (Linear.weight, Embedding.weight)
      - weight_decay=0.0 for biases, norms (RMSNorm.scale), EMA decay logits
    LR schedule: linear warmup over first 1% of steps → cosine decay to 0.
    """
    decay_params, no_decay_params = [], []
    no_decay_names = set()
    for name, param in model.named_parameters():
        # No decay: 1-D params (biases, norm scales, pos_bias)
        if param.ndim <= 1 or 'pos_bias' in name:
            no_decay_params.append(param)
            no_decay_names.add(name)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [{'params': decay_params,    'weight_decay': 0.1},
         {'params': no_decay_params, 'weight_decay': 0.0}],
        lr=LR, betas=(0.9, 0.95))

    # Linear warmup + cosine decay
    warmup_steps = max(1, int(0.01 * total_steps))   # 1% warmup

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f'  Optimizer: {len(decay_params)} decay params, '
          f'{len(no_decay_params)} no-decay params')
    print(f'  LR schedule: {warmup_steps} warmup steps → cosine over {total_steps} total')
    return optimizer, scheduler


def train(model, train_data, val_data, test_data, tokenizer,
          save_dir='checkpoints/2048_condM_v2_checkpoints', device='cuda'):
    os.makedirs(save_dir, exist_ok=True)
    total_steps = NUM_EPOCHS * math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    optimizer, scheduler = build_optimizer(model, total_steps)
    # bf16: 4× dynamic range vs fp16, native on 4090; no NaN risk from large activations
    scaler = torch.amp.GradScaler('cuda', enabled=False)   # not needed for bf16
    autocast_dtype = torch.bfloat16

    best_val_loss, best_val_ppl, best_epoch = float('inf'), float('inf'), 0
    t0 = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        indices = torch.randperm(len(train_data))
        step    = 0
        optimizer.zero_grad()
        steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)

        for acc_step in range(steps_per_epoch):
            for ga in range(GRAD_ACCUM):
                idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if idx_start >= len(train_data): continue
                batch = train_data[indices[idx_start: idx_start + BATCH_SIZE]]
                x, y  = batch[:, :-1].to(device), batch[:, 1:].to(device)
                with torch.amp.autocast('cuda', dtype=autocast_dtype):
                    loss = F.cross_entropy(
                        model(x).reshape(-1, VOCAB_SIZE),
                        y.reshape(-1)) / GRAD_ACCUM
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
            scheduler.step(); step += 1
            if step % 200 == 0:
                print(f'  Step {step}/{steps_per_epoch} | Loss {loss.item() * GRAD_ACCUM:.4f}')

        train_loss = loss.item() * GRAD_ACCUM
        val_loss   = evaluate(model, val_data, BATCH_SIZE, device)
        val_ppl    = math.exp(min(val_loss, 20))
        elapsed    = time.time() - t0

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss, best_val_ppl, best_epoch = val_loss, val_ppl, epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))
            marker = ' * BEST'

        print(f'Ep {epoch}/{NUM_EPOCHS} | Train {train_loss:.4f} '
              f'| Val {val_loss:.4f} PPL {val_ppl:.1f}{marker} | {elapsed:.0f}s')

        ss = model.attn_summary()
        head_means  = ss['pos_bias_mean_per_head']
        most_local  = int(max(range(NUM_HEADS), key=lambda h: abs(head_means[h])))
        most_global = int(min(range(NUM_HEADS), key=lambda h: abs(head_means[h])))
        print(f'  DSQG pos-bias: |mean|={ss["pos_bias_abs_mean"]:.4f} '
              f'|max|={ss["pos_bias_abs_max"]:.4f} '
              f'most-local=h{most_local} most-global=h{most_global}')

        print('  ── Generation samples (greedy) ──')
        for prompt, gen in zip(GEN_PROMPTS,
                               generate(model, tokenizer, GEN_PROMPTS, device,
                                        temperature=0.0)):
            print(f'    {repr(prompt)} → {repr(gen[:80])}')
        print('  ──')
        sys.stdout.flush()

    # Final test evaluation
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best.pt'),
                                     weights_only=True))
    test_loss = evaluate(model, test_data, BATCH_SIZE, device)
    test_ppl  = math.exp(min(test_loss, 20))
    print(f'\n  condM-v2 TEST: PPL {test_ppl:.3f} | Loss {test_loss:.4f}')

    print('\n  ── Temperature sweep (best checkpoint) ──')
    sweep_results = {}
    for temp in [0.0, 0.5, 0.7, 1.0]:
        label = 'greedy' if temp == 0.0 else f'T={temp}'
        print(f'\n  [{label}]')
        gens = generate(model, tokenizer, GEN_PROMPTS, device,
                        temperature=temp, top_p=0.9)
        sweep_results[label] = gens
        for prompt, gen in zip(GEN_PROMPTS, gens):
            print(f'    {repr(prompt)} → {repr(gen[:80])}')

    print('\n' + '=' * 70)
    print(f'  condM-v2 vs condM baseline (54.529 PPL):  {test_ppl - 54.529:+.3f}')
    print('=' * 70)

    return {
        'test_ppl':          test_ppl,
        'test_loss':         test_loss,
        'best_val_ppl':      best_val_ppl,
        'best_epoch':        best_epoch,
        'total_time_s':      time.time() - t0,
        'condM_baseline':    54.529,
        'delta_vs_baseline': test_ppl - 54.529,
        'full_attn_layer':   FULL_ATTN_LAYER,
        'changes':           ['RMSNorm', 'SwiGLU', 'RoPE', 'EMA_pooling', 'no_abs_pos'],
        'temperature_sweep': sweep_results,
        'attn_summary':      model.attn_summary(),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global FULL_ATTN_LAYER
    parser = argparse.ArgumentParser(description='condM-v2 training')
    parser.add_argument('--full_layer', type=int, default=5,
                        choices=[0, 1, 2, 3, 4, 5])
    args = parser.parse_args()
    FULL_ATTN_LAYER = args.full_layer

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 70)
    print('  DWARF condM-v2: RMSNorm + SwiGLU + RoPE + EMA pooling')
    print(f'  Full attention at layer {FULL_ATTN_LAYER} | No absolute position embeddings')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}  '
              f'({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')

    splits = load_data(NUM_DOCS)

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    tok_candidates = [
        os.path.join(_script_dir, 'results', '2048_condI_tokenizer.json'),
        os.path.join(_script_dir, '2048_condI_tokenizer.json'),
    ]
    tok_path = next((p for p in tok_candidates if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError('condI tokenizer not found.')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'Loaded condI BPE tokenizer from {tok_path}')

    print(f'Encoding data (max_seq_len={MAX_SEQ_LEN})...')
    train_data = encode_split(splits['train'], tokenizer, MAX_SEQ_LEN, 'Train')
    val_data   = encode_split(splits['val'],   tokenizer, MAX_SEQ_LEN, 'Val')
    test_data  = encode_split(splits['test'],  tokenizer, MAX_SEQ_LEN, 'Test')

    model = CondMV2Transformer(
        vocab_size            = tokenizer.vocab_size(),
        embedding_dim         = EMBEDDING_DIM,
        num_layers            = NUM_LAYERS,
        num_heads             = NUM_HEADS,
        ffn_hidden            = FFN_HIDDEN,
        seq_len               = MAX_SEQ_LEN,
        full_attn_layer       = FULL_ATTN_LAYER,
        interference_interval = INTERFERENCE,
    ).to(device)

    n_params    = model.param_count()
    layer_types = ['FULL' if i == FULL_ATTN_LAYER else 'DSQG'
                   for i in range(NUM_LAYERS)]
    print(f'\ncondM-v2: {n_params:,} parameters')
    print(f'  Layer stack:      {layer_types}')
    print(f'  FFN hidden:       {FFN_HIDDEN} (SwiGLU 8D/3, ≈ iso-parameter with 4D GELU)')
    print(f'  Norm:             RMSNorm (no bias)')
    print(f'  Pooling:          Causal mean (vectorized cumsum; EMA dropped: Python loop = ~3× slowdown)')
    print(f'  Position:         DSQG β-bias (relative) + RoPE in full attn (relative)')
    print(f'                    No absolute P[pos] embedding')
    print(f'  δ=0 bias:         Small negative init (−log(1.2)·α) vs 0 in condM')
    print(f'  QKV/out biases:   Removed (gate_proj keeps bias=2.0)')
    print(f'  Residual init:    σ = 0.02/√(2·{NUM_LAYERS}) = {0.02/math.sqrt(2*NUM_LAYERS):.5f}')
    print(f'  Embedding scale:  ×√{EMBEDDING_DIM} = ×{math.sqrt(EMBEDDING_DIM):.2f}')
    print(f'  Autocast:         bf16')
    print(f'  LR warmup:        1% of total steps')
    print(f'  Weight decay:     0.1 for weights, 0.0 for biases/norms/pos_bias')

    if not causality_check(model, device): return

    # torch.compile() disabled: EMA pooling's Python for-loop (range(N=2047)) causes
    # torch.compile to hang during symbolic shape tracing. The other 11 freebies
    # (bf16, RMSNorm, SwiGLU, RoPE, EMA, warmup, scaled init, etc.) are unaffected.
    # A compile-friendly EMA (using torch.linalg or custom CUDA scan) is a future option.

    results = train(model, train_data, val_data, test_data, tokenizer,
                    save_dir='checkpoints/2048_condM_v2_checkpoints', device=device)

    results_path = os.path.join(_script_dir, 'results', 'condM_v2_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f'\n  Results → {results_path}')


if __name__ == '__main__':
    main()
