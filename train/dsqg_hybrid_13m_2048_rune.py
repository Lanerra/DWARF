"""
Run E — DSQG Hybrid 13M @ N=2048, Dense 1-39 + Octave Blocks at 64-512
================================================================

What changes vs Run D (55.488 PPL, 35.0% passkey):
  - Offset set: dense 1-39 (restores V3 short-range NL power + covers δ_eff
    for d=1,2,4,8,16,32) + 4 octave blocks at 64-512 (J=79)
  - pos_bias init: NEAR-ZERO (zeros) — same as Run D
  - All other hyperparameters identical to Run D

Offset design (J=79):
  Base:        {0}                                (1)
  Dense 1-39:  {1,2,...,39}                      (39) — NL + δ_eff for d=1-32
  Gap:         {48}                               (1)
  Block d≈64:  {64-71}                            (8) — δ_eff for d=64
  Gap:         {96}                               (1)
  Block d≈128: {128-135}                          (8) — δ_eff for d=128
  Gap:         {192}                              (1)
  Block d≈256: {256-263}                          (8) — δ_eff for d=256
  Gap:         {384}                              (1)
  Block d≈512: {512-519}                          (8) — δ_eff for d=512
  Tail sparse: {768, 1024, 1536}                  (3)

Motivation (from Run C, J=40, linspace init):
  Run C confirmed: dense coverage at δ=32-48 breaks d=32 passkey wall (0%→40%
  by ep5). But short-range density (δ=0-15) was redundant — language teaches
  d=1-16 regardless. Run D replaces short-range density with scaffolding at
  each octave (32→64→128→256→512), extending the retrieval ladder.
  Near-zero init used to get the early nudge from ep1 (vs ep3 in Run C).

Hypothesis: near-zero + octave scaffolding = simultaneous unlocking of multiple
  octaves at ep1-2, rather than sequential discovery over epochs 3-5+.

Kernel: kernels/dsqg_attention_v3_rune.py (J=79, max δ=1536 unchanged)

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/dsqg_hybrid_13m_2048_rune.py \\
    2>&1 | tee benchmarks/logs/dsqg_hybrid_13m_2048_rune_run.log

Before running: smoke-test the kernel:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 kernels/dsqg_attention_v3_rune.py
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Hyperparameters ────────────────────────────────────────────────────────────

VOCAB_SIZE    = 32000
NUM_EPOCHS    = 10
BATCH_SIZE    = 8
GRAD_ACCUM    = 4
LR            = 3e-4
MAX_SEQ_LEN   = 2048
NUM_DOCS      = 100_000

EMBEDDING_DIM   = 256
NUM_LAYERS      = 6
NUM_HEADS       = 8
FFN_DIM         = 1024
INTERFERENCE    = 3
FULL_ATTN_LAYER = 5

# J=79 dense-1-39 + octave blocks at 64-512
_BASE           = [0]
_DENSE          = list(range(1, 40))     # dense 1-39: NL + δ_eff for d=1-32
_D64            = list(range(64, 72))    # octave block 1: δ_eff for d=64
_D128           = list(range(128, 136))  # octave block 2: δ_eff for d=128
_D256           = list(range(256, 264))  # octave block 3: δ_eff for d=256
_D512           = list(range(512, 520))  # octave block 4: δ_eff for d=512
_TAIL           = [768, 1024, 1536]
_RUNE_OFFSETS   = sorted(_BASE + _DENSE + [48] + _D64 + [96] +
                          _D128 + [192] + _D256 + [384] + _D512 + _TAIL)
NUM_OFFSETS     = len(_RUNE_OFFSETS)     # 79
assert NUM_OFFSETS == 79

MAX_TRAIN_SEQS = 52_716   # same as condU 13M N=2048 baseline (iso-compute)

ENCODED_CACHE  = 'benchmarks/logs/fineweb_encoded_2048.pt'
FW_CACHE_FILE  = 'benchmarks/logs/condm_fineweb_edu_doc_cache.json'

PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 5
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'

SAVE_DIR    = 'checkpoints/2048_dsqg_hybrid_13m_rune'
RESULT_FILE = 'benchmarks/logs/dsqg_hybrid_13m_2048_rune_results.json'

import pathlib as _pl
_kernel_dir = str(_pl.Path(__file__).parent.parent / 'kernels')
if _kernel_dir not in sys.path:
    sys.path.insert(0, _kernel_dir)
from dsqg_attention_v3_rune import dsqg_attention_v3_rune

def dsqg_attention_backend(q, k, v, pos_bias, scale_embed):
    return dsqg_attention_v3_rune(q, k, v, pos_bias, scale_embed)


# ── DSQGAttentionQW — near-zero pos_bias init ────────────────────────────────────

class DSQGAttentionQW(nn.Module):
    def __init__(self, embedding_dim, num_heads, seq_len=2048, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        HD             = self.head_dim

        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        # Run D: near-zero init — all heads start equal; global heads compete from ep1.
        # Avoids linspace locality prior that delays retrieval nudge to ep3+ (as in Run C).
        self.pos_bias    = nn.Parameter(torch.zeros(NUM_OFFSETS, num_heads))
        self.scale_embed = nn.Parameter(torch.zeros(NUM_OFFSETS, HD))
        self.if_gain     = nn.Parameter(torch.ones(num_heads))
        self.dropout     = nn.Dropout(dropout)

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

        out      = dsqg_attention_backend(q, k, v, self.pos_bias, self.scale_embed)
        out      = out * self.if_gain.view(1, H, 1, 1)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate     = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(out_flat * gate))

    def attn_summary(self):
        with torch.no_grad():
            pb   = self.pos_bias.detach().cpu()
            se   = self.scale_embed.detach().cpu()
            gain = self.if_gain.detach().cpu()
        return {
            'pos_bias_abs_mean':      pb.abs().mean().item(),
            'pos_bias_abs_max':       pb.abs().max().item(),
            'pos_bias_mean_per_head': pb.mean(0).tolist(),
            'scale_embed_abs_mean':   se.abs().mean().item(),
            'scale_embed_abs_max':    se.abs().max().item(),
            'if_gain':                gain.tolist(),
        }


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
                'pos_bias_mean_per_head': [0.0]*NUM_HEADS,
                'scale_embed_abs_mean': 0.0, 'scale_embed_abs_max': 0.0,
                'if_gain': [1.0]*NUM_HEADS}


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
        for block in self.blocks: x = block(x)
        return self.out(self.norm(x))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def attn_summary(self):
        dsqg_blocks = [b for b in self.blocks if isinstance(b, DSQGBlock)]
        if not dsqg_blocks:
            return {'pos_bias_abs_mean': 0.0, 'pos_bias_abs_max': 0.0,
                    'pos_bias_mean_per_head': [0.0]*NUM_HEADS,
                    'scale_embed_abs_mean': 0.0, 'scale_embed_abs_max': 0.0,
                    'if_gain': [1.0]*NUM_HEADS}
        summaries = [b.attn.attn_summary() for b in dsqg_blocks]
        n = len(summaries)
        return {
            'pos_bias_abs_mean':      sum(s['pos_bias_abs_mean']    for s in summaries) / n,
            'pos_bias_abs_max':       max(s['pos_bias_abs_max']     for s in summaries),
            'pos_bias_mean_per_head': [sum(s['pos_bias_mean_per_head'][h] for s in summaries)/n
                                       for h in range(NUM_HEADS)],
            'scale_embed_abs_mean':   sum(s['scale_embed_abs_mean'] for s in summaries) / n,
            'scale_embed_abs_max':    max(s['scale_embed_abs_max']  for s in summaries),
            'if_gain':               [sum(s['if_gain'][h] for s in summaries)/n
                                      for h in range(NUM_HEADS)],
        }


# ── Utilities ──────────────────────────────────────────────────────────────────

class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)
    def vocab_size(self):   return self.tokenizer.get_vocab_size()


def encode_split(split_texts, tokenizer, split_name):
    tokens = []
    for text in split_texts:
        tokens.extend(tokenizer.encode(text))
        tokens.append(3)
    n    = (len(tokens) // MAX_SEQ_LEN) * MAX_SEQ_LEN
    data = torch.tensor(tokens[:n], dtype=torch.long).view(-1, MAX_SEQ_LEN)
    print(f'  {split_name}: {len(data):,} sequences')
    return data


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


def generate(model, tokenizer, prompts, device, max_new=80, temperature=0.0):
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
        results.append(tokenizer.decode(ids[0, len(tokenizer.encode(prompt)):].tolist())[:80])
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

    best_val_loss    = float('inf')
    t0               = time.time()
    per_epoch        = []
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
        print(f'  scale_embed:   |mean|={ss["scale_embed_abs_mean"]:.4f} '
              f'|max|={ss["scale_embed_abs_max"]:.4f}')
        gains    = ss['if_gain']
        gain_str = '  '.join(f'h{h}:{gains[h]:.3f}' for h in range(NUM_HEADS))
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
                          'if_gain': ss['if_gain']})
        sys.stdout.flush()

    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best.pt'), weights_only=True))
    test_loss = evaluate(model, test_data, BATCH_SIZE, device)
    test_ppl  = math.exp(min(test_loss, 20))
    print(f'\n  Run D (octave scaffolding) TEST PPL: {test_ppl:.3f}')
    print(f'  condU 13M 2048 baseline:             52.206 PPL  delta={test_ppl-52.206:+.3f}')
    print(f'  Run C (interleaved, linspace):       ~52 PPL est (in progress)')

    pk_final = passkey_accuracy(model, tokenizer, device)
    pk_mean  = sum(pk_final.values()) / len(pk_final)
    print(f'  Final passkey: mean={pk_mean*100:.1f}%')
    print('  ' + '  '.join(f'd={d}:{int(pk_final[d]*100)}%' for d in PASSKEY_DISTANCES))
    print(f'  condU 13M 2048 baseline pk: 43.3%')

    results = {
        'experiment':      'run_e_dsqg_hybrid_13m_2048_rune',
        'description':     'Dense 1-39 + octave blocks 64-512 (J=79); N=2048; near-zero init',
        'offset_set':      _RUNE_OFFSETS,
        'offset_design':   'sparse 0-31 | dense 32-39 | 48 | dense 64-71 | 96 | dense 128-135 | 192 | dense 256-263 | 384 | dense 512-519 | 768 1024 1536',
        'change_vs_base':  'dense 1-39 + octave blocks 64-512 (J=79) + near-zero pos_bias init',
        'num_offsets':     NUM_OFFSETS,
        'final_test_ppl':  test_ppl,
        'final_passkey_mean': pk_mean,
        'final_passkey_by_d': {str(d): v for d, v in pk_final.items()},
        'per_epoch': per_epoch,
        'refs': {
            'condu_13m_2048_ppl':    52.206,
            'condu_13m_2048_pk':     0.433,
            'run_c_interleaved_j40': 'in progress (linspace init)',
            'v3_offsets':            'range(0,33) + [48,64,96,128,192,256,384,512,768,1024,1536]',
        },
    }
    with open(RESULT_FILE, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f'\n  Results -> {RESULT_FILE}')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 70)
    print('  Run E — DSQG Hybrid 13M @ N=2048 — Dense 1-39 + Octave Blocks (J=79)')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, L={NUM_LAYERS}, N={MAX_SEQ_LEN}')
    print(f'  Offsets ({NUM_OFFSETS}): sparse 0-31 | dense 32-39,64-71,128-135,256-263,512-519 | sparse tail')
    print(f'  Init: near-zero pos_bias (global heads compete from ep1)')
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
        print(f'Loading encoded dataset from {ENCODED_CACHE}...')
        _cache     = torch.load(ENCODED_CACHE, weights_only=True)
        train_data = _cache['train']
        val_data   = _cache['val']
        test_data  = _cache['test']
    else:
        import json as _json
        print(f'Encoding from JSON cache {FW_CACHE_FILE}...')
        with open(FW_CACHE_FILE) as fp:
            texts = _json.load(fp)
        n = len(texts)
        train_data = encode_split(texts[:int(n*0.95)],                    tokenizer, 'Train')
        val_data   = encode_split(texts[int(n*0.95): int(n*0.95)+2500],  tokenizer, 'Val')
        test_data  = encode_split(texts[int(n*0.95)+2500: int(n*0.95)+5000], tokenizer, 'Test')
        torch.save({'train': train_data, 'val': val_data, 'test': test_data}, ENCODED_CACHE)

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

    # Causality check
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
