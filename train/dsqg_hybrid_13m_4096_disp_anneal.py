"""
Run B — DSQG Hybrid 13M @ N=4096, Dispersive Kernel + Near-Zero Init
======================================================================

What changes vs Run A (dsqg_hybrid_13m_4096_anneal.py):
  - Adds dispersive component to pos_bias:
      effective_pos_bias[j,h] = pos_bias[j,h] + A_h * cos(ω_h * log(1+δ_j))
  - Per-head fixed frequencies ω_h chosen from energy-conserving set
    (verified in verification/src/advanced_annealing.rs test 1a)
  - Per-head learned amplitude A_h, initialised small (0.1)
  - pos_bias still near-zero init (same as Run A)

Physics motivation:
  Pure damping (current): each head is a lowpass filter with different cutoffs.
  Dispersive kernel: adds oscillatory component A_h·cos(ω_h·log(1+δ)) — enables
  bandpass behaviour per head. Each head tunes to a different spatial frequency,
  attending to a preferred distance band rather than just "closer = more weight".

  From Rust tests (advanced_annealing.rs test 5):
    dispersive + power-law (near-zero init) = 1.97× more long-range gradient flow
    vs pure damping + cosine schedule. The synergy: the dispersive kernel creates
    non-zero weight at long range even at small pos_bias; near-zero init keeps
    pos_bias small long enough for that weight to drive gradient.

Energy-conserving ω selection:
  ω_k = k * π / log(1 + 4096)  [V4 max offset = 4096, log(4097) ≈ 8.318]
  Energy-conserving (near-zero mean over V4 offset set): k = 1, 4, 6
    k=1: ω = 0.378 — ~0.5 cycles over full log-range (broad bands)
    k=4: ω = 1.511 — ~2 cycles (medium frequency selectivity)
    k=6: ω = 2.267 — ~3 cycles (narrow bands, sharp distance tuning)
  Heads 0-1: ω=0 (pure damping — global heads kept unmodified)
  Heads 2-3: ω=k1 (low frequency)
  Heads 4-5: ω=k4 (medium frequency)
  Heads 6-7: ω=k6 (high frequency — most local)

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/dsqg_hybrid_13m_4096_disp_anneal.py \\
    2>&1 | tee logs/dsqg_hybrid_13m_4096_disp_anneal_run.log
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Hyperparameters (identical to Run A / baseline) ────────────────────────────

VOCAB_SIZE    = 32000
NUM_EPOCHS    = 10
BATCH_SIZE    = 8
GRAD_ACCUM    = 4
LR            = 3e-4
MAX_SEQ_LEN   = 4096
NUM_DOCS      = 100_000

EMBEDDING_DIM   = 256
NUM_LAYERS      = 6
NUM_HEADS       = 8
FFN_DIM         = 1024
INTERFERENCE    = 3
FULL_ATTN_LAYER = 5

NUM_OFFSETS     = 47
MAX_TRAIN_SEQS  = 26_358

ENCODED_CACHE = 'benchmarks/logs/fineweb_encoded_4096.pt'

PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072]
PASSKEY_TRIALS    = 5
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'

SAVE_DIR    = 'checkpoints/4096_dsqg_hybrid_13m_disp_anneal'
RESULT_FILE = 'logs/dsqg_hybrid_13m_4096_disp_anneal_results.json'

_DENSE_LOCAL_W     = 32
_DYADIC_LONG_RANGE = [48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536,
                      2048, 3072, 4096]
_COND_N_OFFSETS    = sorted(set(range(0, _DENSE_LOCAL_W + 1)) | set(_DYADIC_LONG_RANGE))
assert len(_COND_N_OFFSETS) == NUM_OFFSETS == 47

import pathlib as _pl
_kernel_dir = str(_pl.Path(__file__).parent.parent / 'kernels')
if _kernel_dir not in sys.path:
    sys.path.insert(0, _kernel_dir)
from dsqg_attention_v4 import dsqg_attention_v4

def dsqg_attention_backend(q, k, v, pos_bias, scale_embed):
    return dsqg_attention_v4(q, k, v, pos_bias, scale_embed)

# ── Dispersive kernel constants ────────────────────────────────────────────────
# Energy-conserving frequencies for V4 offset set (log(4097) ≈ 8.318)
_LOG_MAX_V4 = math.log(1 + 4096)
_OMEGA_K1   = 1 * math.pi / _LOG_MAX_V4   # 0.378 — 0.5 cycles
_OMEGA_K4   = 4 * math.pi / _LOG_MAX_V4   # 1.511 — 2 cycles
_OMEGA_K6   = 6 * math.pi / _LOG_MAX_V4   # 2.267 — 3 cycles

# Per-head frequency assignment:
#   h0, h1: ω=0 (pure damping — global heads kept clean)
#   h2, h3: low frequency (broad bands)
#   h4, h5: medium frequency
#   h6, h7: high frequency (most local, narrowest bands)
_HEAD_OMEGAS = [0.0, 0.0, _OMEGA_K1, _OMEGA_K1, _OMEGA_K4, _OMEGA_K4, _OMEGA_K6, _OMEGA_K6]
_DISP_INIT_AMP = 0.1   # small initial amplitude; learned from there


# ── DSQGAttentionQW — dispersive kernel + near-zero pos_bias init ──────────────

class DSQGAttentionQW(nn.Module):
    def __init__(self, embedding_dim, num_heads, seq_len=4096, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        HD             = self.head_dim

        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        # Near-zero pos_bias init (same as Run A)
        self.pos_bias    = nn.Parameter(torch.zeros(NUM_OFFSETS, num_heads) + 0.001)
        self.scale_embed = nn.Parameter(torch.zeros(NUM_OFFSETS, HD))
        self.if_gain     = nn.Parameter(torch.ones(num_heads))

        # ── Run B key addition: dispersive component ──────────────────────────
        # Fixed per-head frequencies ω_h (energy-conserving, not learned)
        # Learned per-head amplitudes A_h (start small, gradient shapes them)
        self.disp_amp = nn.Parameter(
            torch.full((num_heads,), _DISP_INIT_AMP))

        # Precompute cos kernel [J, H] — fixed throughout training
        log_deltas  = torch.tensor([math.log(1.0 + d) for d in _COND_N_OFFSETS])
        cos_kernel  = torch.zeros(NUM_OFFSETS, num_heads)
        for h_idx, omega in enumerate(_HEAD_OMEGAS):
            if omega > 0.0:
                cos_kernel[:, h_idx] = torch.cos(omega * log_deltas)
            # ω=0 heads: cos_kernel stays zero → no dispersive component
        self.register_buffer('disp_cos_kernel', cos_kernel)  # [J, H], not learned
        # ─────────────────────────────────────────────────────────────────────

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_inject=None):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim

        qkv     = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3).contiguous()

        if kv_inject is not None:
            k_delta, v_delta = kv_inject
            k = k + k_delta
            v = v + v_delta

        # Effective pos_bias = learned base + dispersive oscillatory component
        # disp_cos_kernel [J, H], disp_amp [H] → effective_pb [J, H]
        effective_pb = self.pos_bias + self.disp_cos_kernel * self.disp_amp.unsqueeze(0)

        out      = dsqg_attention_backend(q, k, v, effective_pb, self.scale_embed)
        out      = out * self.if_gain.view(1, H, 1, 1)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate     = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(out_flat * gate))

    def attn_summary(self):
        with torch.no_grad():
            pb   = self.pos_bias.detach().cpu()
            se   = self.scale_embed.detach().cpu()
            gain = self.if_gain.detach().cpu()
            amp  = self.disp_amp.detach().cpu()
            epb  = (self.pos_bias + self.disp_cos_kernel * self.disp_amp.unsqueeze(0)).detach().cpu()
        return {
            'pos_bias_abs_mean':       pb.abs().mean().item(),
            'pos_bias_abs_max':        pb.abs().max().item(),
            'pos_bias_mean_per_head':  pb.mean(0).tolist(),
            'effective_pb_abs_mean':   epb.abs().mean().item(),
            'effective_pb_abs_max':    epb.abs().max().item(),
            'scale_embed_abs_mean':    se.abs().mean().item(),
            'scale_embed_abs_max':     se.abs().max().item(),
            'if_gain':                 gain.tolist(),
            'disp_amp':                amp.tolist(),
        }


# ── FullCausalAttention (unchanged) ───────────────────────────────────────────

class FullCausalAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)
        self.dropout_p = dropout

    def forward(self, x):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim
        qkv     = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0, is_causal=True)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate     = torch.sigmoid(self.gate_proj(x))
        return F.dropout(self.out_proj(out_flat * gate),
                         p=self.dropout_p, training=self.training)

    def attn_summary(self):
        return {'pos_bias_abs_mean': 0.0, 'pos_bias_abs_max': 0.0,
                'effective_pb_abs_mean': 0.0, 'effective_pb_abs_max': 0.0,
                'pos_bias_mean_per_head': [0.0]*NUM_HEADS,
                'scale_embed_abs_mean': 0.0, 'scale_embed_abs_max': 0.0,
                'if_gain': [1.0]*NUM_HEADS, 'disp_amp': [0.0]*NUM_HEADS}


class FFN(nn.Module):
    def __init__(self, embedding_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(embedding_dim, ffn_dim)
        self.fc2  = nn.Linear(ffn_dim, embedding_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class DSQGBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 dropout=0.1, interference=False):
        super().__init__()
        self.interference = interference
        self.num_heads    = num_heads
        self.head_dim     = embedding_dim // num_heads
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = DSQGAttentionQW(embedding_dim, num_heads,
                                     seq_len=seq_len, dropout=dropout)
        self.ffn   = FFN(embedding_dim, ffn_dim, dropout)
        if interference:
            self.inter_norm   = nn.LayerNorm(embedding_dim)
            self.inter_gate   = nn.Linear(embedding_dim, embedding_dim)
            self.inter_k_proj = nn.Linear(embedding_dim, embedding_dim)
            self.inter_v_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        kv_inject = None
        if self.interference:
            xi      = self.inter_norm(x)
            B, N, D = xi.shape
            H, HD   = self.num_heads, self.head_dim
            counts  = torch.arange(1, N+1, device=xi.device, dtype=xi.dtype).view(1, N, 1)
            pool    = xi.cumsum(dim=1) / counts
            inter   = torch.sigmoid(self.inter_gate(xi)) * pool
            k_delta = self.inter_k_proj(inter).view(B,N,H,HD).permute(0,2,1,3).contiguous()
            v_delta = self.inter_v_proj(inter).view(B,N,H,HD).permute(0,2,1,3).contiguous()
            kv_inject = (k_delta, v_delta)
        x = x + self.attn(self.norm1(x), kv_inject=kv_inject)
        x = x + self.ffn(self.norm2(x))
        return x


class FullAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = FullCausalAttention(embedding_dim, num_heads, dropout)
        self.ffn   = FFN(embedding_dim, ffn_dim, dropout)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CondUTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len, full_attn_layer=FULL_ATTN_LAYER,
                 interference_interval=INTERFERENCE, dropout=0.1):
        super().__init__()
        self.embedding       = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed       = nn.Embedding(seq_len + 2, embedding_dim)
        self.drop            = nn.Dropout(dropout)
        self.full_attn_layer = full_attn_layer
        blocks = []
        for i in range(num_layers):
            if i == full_attn_layer:
                blocks.append(FullAttentionBlock(embedding_dim, num_heads, ffn_dim, dropout))
            else:
                blocks.append(DSQGBlock(
                    embedding_dim, num_heads, ffn_dim, seq_len, dropout=dropout,
                    interference=(i % interference_interval == interference_interval - 1)))
        self.blocks = nn.ModuleList(blocks)
        self.norm   = nn.LayerNorm(embedding_dim)
        self.out    = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.out.weight = self.embedding.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)
        for m in self.modules():
            if hasattr(m, 'gate_proj') and isinstance(m.gate_proj, nn.Linear):
                nn.init.constant_(m.gate_proj.bias, 0.0)

    def forward(self, idx):
        B, N = idx.shape
        pos  = torch.arange(N, device=idx.device).unsqueeze(0)
        x    = self.drop(self.embedding(idx) + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def attn_summary(self):
        dsqg_blocks = [b for b in self.blocks if isinstance(b, DSQGBlock)]
        if not dsqg_blocks:
            return {'pos_bias_abs_mean': 0.0, 'pos_bias_abs_max': 0.0,
                    'effective_pb_abs_mean': 0.0, 'effective_pb_abs_max': 0.0,
                    'pos_bias_mean_per_head': [0.0]*NUM_HEADS,
                    'scale_embed_abs_mean': 0.0, 'scale_embed_abs_max': 0.0,
                    'if_gain': [1.0]*NUM_HEADS, 'disp_amp': [0.0]*NUM_HEADS}
        summaries = [b.attn.attn_summary() for b in dsqg_blocks]
        n = len(summaries)
        def avg(key): return sum(s[key] for s in summaries) / n
        def mx(key):  return max(s[key] for s in summaries)
        return {
            'pos_bias_abs_mean':      avg('pos_bias_abs_mean'),
            'pos_bias_abs_max':       mx('pos_bias_abs_max'),
            'effective_pb_abs_mean':  avg('effective_pb_abs_mean'),
            'effective_pb_abs_max':   mx('effective_pb_abs_max'),
            'pos_bias_mean_per_head': [sum(s['pos_bias_mean_per_head'][h] for s in summaries)/n
                                       for h in range(NUM_HEADS)],
            'scale_embed_abs_mean':   avg('scale_embed_abs_mean'),
            'scale_embed_abs_max':    mx('scale_embed_abs_max'),
            'if_gain':               [sum(s['if_gain'][h] for s in summaries)/n
                                      for h in range(NUM_HEADS)],
            'disp_amp':              [sum(s['disp_amp'][h] for s in summaries)/n
                                      for h in range(NUM_HEADS)],
        }


# ── Utilities (identical to Run A) ────────────────────────────────────────────

class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)
    def vocab_size(self):   return self.tokenizer.get_vocab_size()


@torch.no_grad()
def evaluate(model, data, batch_size, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(data) - batch_size + 1, batch_size):
        x = data[i:i+batch_size, :-1].to(device)
        y = data[i:i+batch_size,  1:].to(device)
        with torch.amp.autocast('cuda'):
            logits = model(x)
            loss   = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss   += loss.item() * y.numel()
        total_tokens += y.numel()
    return total_loss / max(total_tokens, 1)


def generate(model, tokenizer, prompts, device, max_new=150, temperature=0.0):
    model.eval()
    results = []
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(max_new):
                with torch.amp.autocast('cuda'):
                    logits = model(ids[:, -MAX_SEQ_LEN:])
                next_id = logits[0, -1].argmax()
                ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
        results.append(tokenizer.decode(ids[0, len(tokenizer.encode(prompt)):].tolist())[:120])
    return results


def passkey_accuracy(model, tokenizer, device):
    model.eval()
    filler_ids = tokenizer.encode(_FILLER_SENTENCE)
    cue_ids    = tokenizer.encode(_RETRIEVAL_CUE)
    results    = {}
    for d in PASSKEY_DISTANCES:
        correct = 0; n_valid = 0
        for i in range(PASSKEY_TRIALS):
            target    = _PASSKEY_WORDS[i % len(_PASSKEY_WORDS)]
            others    = [w for w in _PASSKEY_WORDS if w != target]
            intro_ids = tokenizer.encode(_INTRO_TEMPLATE.format(word=target))
            available = MAX_SEQ_LEN - 1 - len(intro_ids) - len(cue_ids) - 1
            if d > available: continue
            filler    = []
            while len(filler) < d: filler.extend(filler_ids)
            full_seq  = intro_ids + filler[:d] + cue_ids
            if len(full_seq) >= MAX_SEQ_LEN: continue
            ids    = torch.tensor([full_seq], dtype=torch.long, device=device)
            with torch.amp.autocast('cuda'):
                logits = model(ids)[:, -1, :]
            cand_ids = [(tokenizer.encode(' ' + w) or tokenizer.encode(w))[0]
                        for w in [target] + others[:9]]
            correct  += int(([target]+others[:9])[logits[0][cand_ids].argmax().item()] == target)
            n_valid  += 1
        results[d] = correct / n_valid if n_valid else 0.0
    return results


GEN_PROMPTS = [
    'It was a dark and stormy',
    'The length of the hypotenuse',
    'The President of the United',
    'Once upon a time there was',
    'The results indicate that',
]


def train(model, train_data, val_data, test_data, tokenizer, device='cuda'):
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = NUM_EPOCHS * math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler      = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    t0            = time.time()
    per_epoch     = []
    tokens_per_epoch = len(train_data) * (MAX_SEQ_LEN - 1)
    chin_tokens      = 20 * model.param_count()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        indices         = torch.randperm(len(train_data))
        step            = 0
        optimizer.zero_grad()
        steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
        running_loss    = 0.0

        for acc_step in range(steps_per_epoch):
            for ga in range(GRAD_ACCUM):
                idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if idx_start >= len(train_data): continue
                batch = train_data[indices[idx_start: idx_start + BATCH_SIZE]]
                x, y  = batch[:, :-1].to(device), batch[:, 1:].to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(x)
                    loss   = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), y.reshape(-1)) / GRAD_ACCUM
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            scheduler.step(); step += 1
            running_loss += loss.item() * GRAD_ACCUM
            if step % 200 == 0:
                print(f'  Step {step}/{steps_per_epoch} | Loss {loss.item()*GRAD_ACCUM:.4f}')

        train_loss = running_loss / max(step, 1)
        val_loss   = evaluate(model, val_data, BATCH_SIZE, device)
        val_ppl    = math.exp(min(val_loss, 20))
        elapsed    = time.time() - t0
        chin_pct   = epoch * tokens_per_epoch / chin_tokens * 100

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best.pt'))
            marker = ' * BEST'
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'val_ppl': val_ppl}, os.path.join(SAVE_DIR, f'epoch_{epoch:02d}.pt'))

        print(f'Ep {epoch}/{NUM_EPOCHS} | Train {train_loss:.4f} '
              f'| Val {val_loss:.4f} PPL {val_ppl:.1f}{marker} '
              f'| {elapsed:.0f}s ({chin_pct:.0f}%C)')

        ss = model.attn_summary()
        hm = ss['pos_bias_mean_per_head']
        ml = int(max(range(NUM_HEADS), key=lambda h: abs(hm[h])))
        mg = int(min(range(NUM_HEADS), key=lambda h: abs(hm[h])))
        print(f'  DSQG pos-bias: |mean|={ss["pos_bias_abs_mean"]:.4f} '
              f'|max|={ss["pos_bias_abs_max"]:.4f} most-local=h{ml} most-global=h{mg}')
        print(f'  Effective pb:  |mean|={ss["effective_pb_abs_mean"]:.4f} '
              f'|max|={ss["effective_pb_abs_max"]:.4f}')
        print(f'  scale_embed:   |mean|={ss["scale_embed_abs_mean"]:.4f} '
              f'|max|={ss["scale_embed_abs_max"]:.4f}')
        amp_str  = '  '.join(f'h{h}:{ss["disp_amp"][h]:.3f}' for h in range(NUM_HEADS))
        gain_str = '  '.join(f'h{h}:{ss["if_gain"][h]:.3f}' for h in range(NUM_HEADS))
        print(f'  disp_amp: {amp_str}')
        print(f'  IF gains: {gain_str}')

        print('  -- Generation samples (greedy) --')
        for p, g in zip(GEN_PROMPTS, generate(model, tokenizer, GEN_PROMPTS, device)):
            print(f"    {repr(p)} -> {repr(g[:80])}")
        print('  --')

        print('  Passkey...')
        pk      = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        above50 = sum(1 for v in pk.values() if v >= 0.5)
        print(f'  mean={pk_mean*100:.1f}%  ({above50}/{len(pk)} distances >50%)')
        print('  ' + '  '.join(f'd={d}:{int(pk[d]*100)}%' for d in PASSKEY_DISTANCES))

        per_epoch.append({'epoch': epoch, 'val_ppl': val_ppl, 'train_loss': train_loss,
                          'chinchilla_pct': chin_pct, 'elapsed_s': elapsed,
                          'passkey_mean': pk_mean,
                          'passkey_by_d': {str(d): v for d, v in pk.items()},
                          'if_gain': ss['if_gain'], 'disp_amp': ss['disp_amp']})
        sys.stdout.flush()

    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best.pt'), weights_only=True))
    test_loss = evaluate(model, test_data, BATCH_SIZE, device)
    test_ppl  = math.exp(min(test_loss, 20))
    print(f'\n  Run B (disp+anneal) TEST PPL: {test_ppl:.3f}')
    pk_final = passkey_accuracy(model, tokenizer, device)
    pk_mean  = sum(pk_final.values()) / len(pk_final)
    print(f'  Final passkey: mean={pk_mean*100:.1f}%')
    print('  ' + '  '.join(f'd={d}:{int(pk_final[d]*100)}%' for d in PASSKEY_DISTANCES))

    ss = model.attn_summary()
    print(f'  Final disp_amp: {[f"{a:.3f}" for a in ss["disp_amp"]]}')

    results = {
        'experiment':     'run_b_dsqg_hybrid_13m_4096_disp_anneal',
        'description':    'Dispersive kernel + near-zero pos_bias init; N=4096; V4 kernel',
        'disp_omegas':    _HEAD_OMEGAS,
        'disp_init_amp':  _DISP_INIT_AMP,
        'final_test_ppl': test_ppl,
        'final_passkey_mean': pk_mean,
        'final_passkey_by_d': {str(d): v for d, v in pk_final.items()},
        'per_epoch': per_epoch,
        'refs': {'condu_13m_2048_ppl': 52.206, 'condu_13m_2048_pk': 0.433},
    }
    with open(RESULT_FILE, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f'\n  Results -> {RESULT_FILE}')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 70)
    print('  Run B — DSQG Hybrid 13M @ N=4096 — Dispersive Kernel + Anneal')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, L={NUM_LAYERS}, N={MAX_SEQ_LEN}')
    print(f'  Dispersive ωs: {[f"{w:.3f}" for w in _HEAD_OMEGAS]}')
    print(f'  Init amp: {_DISP_INIT_AMP}  (learned per head)')
    print('=' * 70)

    torch.set_float32_matmul_precision('high')
    os.makedirs('logs', exist_ok=True)

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    tok_path    = os.path.join(_script_dir, '..', 'benchmarks', 'results',
                               '2048_condI_tokenizer.json')
    if not os.path.exists(tok_path):
        tok_path = os.path.join(_script_dir, '..', 'benchmarks', '2048_condI_tokenizer.json')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))

    if os.path.exists(ENCODED_CACHE):
        _cache     = torch.load(ENCODED_CACHE, weights_only=True)
        train_data = _cache['train']
        val_data   = _cache['val']
        test_data  = _cache['test']
    else:
        raise FileNotFoundError(f'{ENCODED_CACHE} not found.')

    if len(train_data) > MAX_TRAIN_SEQS:
        idx        = torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]
        train_data = train_data[idx]
    print(f'  train: {len(train_data):,}  val: {len(val_data):,}  test: {len(test_data):,}')

    model = CondUTransformer(
        vocab_size=tokenizer.vocab_size(), embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ffn_dim=FFN_DIM,
        seq_len=MAX_SEQ_LEN, full_attn_layer=FULL_ATTN_LAYER,
        interference_interval=INTERFERENCE,
    ).to(device)
    print(f'  Parameters: {model.param_count():,}')

    # Quick causality check
    model.eval()
    with torch.no_grad():
        x1 = torch.randint(0, VOCAB_SIZE, (1, 64), device=device)
        x2 = x1.clone(); x2[0, 10] = (x2[0, 10] + 1) % VOCAB_SIZE
        d  = (model(x1) - model(x2)).abs()
    ok = d[0, :10].max().item() < 1e-6
    print(f'  Causality: {"PASS" if ok else "FAIL"}')
    if not ok: return

    print('  Compiling model (torch.compile mode=default)...')
    model = torch.compile(model, mode='default')
    train(model, train_data, val_data, test_data, tokenizer, device=device)


if __name__ == '__main__':
    main()
