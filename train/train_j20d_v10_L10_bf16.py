"""
🔬 DWARF Autoresearch — J20D V10 L10 Depth Experiment

Builds on J26D/J24D work with three architectural advances from Deep Research
session (March 15, 2026):

  1. V9 kernel (J=20 Frobenius-optimal set):
     Frobenius co-optimization search found J=20 achieves equal passkey coverage
     to J=24 with better redundancy. V10 kernel rewrites from scratch for coalesced
     HBM access: BLOCK_M=16 output tiles, contiguous [BLOCK_M x HD] KV loads,
     num_stages=3 software pipelining. 2.87x faster than V8 (1.50ms vs 4.30ms).
     No MOVT/QK-OVT/NPCI in kernel (pos_bias + scale_embed only). dQ computed
     analytically in PyTorch; dK/dV via tiled Triton kernel.

  2. KdV removal:
     KdV soliton correction (pool += α·pool·Δpool) was confirmed null throughout
     all j24d/j26d runs (kdv_alpha → near-zero). Deep Research confirms this is
     theoretically expected: KdV addresses dispersion (frequency-dependent speed)
     in spatial continua; discrete token sequences have no such continuum.
     Attenuation (EMA) is the only relevant mechanism. Cited: MEGA (Ma et al.,
     ICLR 2023), forget gate literature (Van der Westhuizen & Lasenby, 2018).

  3. NoPE in full attention layer:
     Deep Research finding: HypeNet achieves 128K+ context via NoPE in non-attention
     layers (Kazemnejad et al., NeurIPS 2023: NoPE outperforms all explicit position
     encodings on length generalization). DWARF's DSQG layers already use relative
     position via pos_bias (effectively NoPE-like). The full attention layer at L5
     is the only layer with absolute position information (learned pos_embed).
     Removing it from full attention → position-invariant relay-chain + retrieval.
     Implementation: FullCausalAttention receives x WITHOUT position embedding added.
     The shared pos_embed is still used by DSQG layers; full attention just doesn't
     get the positional injection.

  4. EMA dead-zone fix (from j26d):
     EMA_INIT=0.00035 (near-zero, prevents Adam momentum overshoot past floor).
     EMA_FLOOR=0.00001 (nearly unconstrained, lets model find natural attractor).
     Combined with exact Triton scan EMA (causal_ema_scan.py) — no K=256 truncation.

Offsets (J=20, V9 kernel):
  OFFSETS = [1,2,3,4,5,6,7,8,9,11,13,15,16,23,32,64,128,256,512,1024]
  J_SMALL=14 (δ≤23, no MOVT), J_LARGE=6 (δ≥32, MOVT applied)

Run (from repo root):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_j20d_v9_bf16.py \
    > logs/run_j20d_v9.log 2>&1 &
"""

# =============================================================================
# EXPERIMENT KNOBS
# =============================================================================

OFFSETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 23, 32, 64, 128, 256, 512, 1024]

EMBEDDING_DIM    = 512
NUM_HEADS        = 8
FFN_DIM          = 2048
NUM_LAYERS       = 10
INTERFERENCE     = 2
FULL_ATTN_LAYER  = 9

MAX_TRAIN_SEQS      = 121_232   # match d41_35m / fulldata confirmed winner
SCALE_EMBED_INIT_VAL = 0.1      # confirmed working from fulldata run
SCALE_EMBED_LR_MULT  = 15.0

# Kalman-EMA init — CRITICAL. 0.003 → window ≈ 333t at init.
# Prior d41-J16D used 0.03 (33t) and immediately collapsed to 10t.
# Lower init gives the relay gradient signal time to keep α from rising.
EMA_INIT  = 0.00035  # near-zero init — lets optimizer find natural attractor
                     # (prev 0.003 caused Adam momentum to overshoot floor → dead zone)
EMA_FLOOR = 0.00001  # nearly unconstrained — model can settle at true preferred window
                     # (prev 0.0003 still caused dead zone when raw param went negative)

LR            = 3e-4
SCREEN_EPOCHS = 3

# =============================================================================

import json, math, os, subprocess, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE     = 32000
BATCH_SIZE     = 8    # 4090: CE OOM at BS>8 (V=32K); H100 can use BS=32
GRAD_ACCUM     = 4    # effective batch = 32
MAX_SEQ_LEN    = 2048
NUM_DOCS       = 100_000
MAX_VAL_SEQS   = 5_582

FW_CACHE_FILE = 'benchmarks/logs/condm_fineweb_edu_doc_cache.json'
TOKENIZER_CANDIDATES = [
    'benchmarks/logs/condm_tokenizer.json',
    'results/2048_condI_tokenizer.json',
]
PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 20
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'
CHECKPOINT_DIR    = 'autoresearch/checkpoints'

# ── Kernel import ─────────────────────────────────────────────────────────────

import pathlib as _pl
_project_root = str(_pl.Path(__file__).resolve().parent.parent)
_kernel_dir   = os.path.join(_project_root, 'kernels')
for _d in [_kernel_dir, _project_root]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from dsqg_attention_v10 import DSQGAttentionV10 as DSQGAttentionV6

assert len(OFFSETS) == 20
assert FULL_ATTN_LAYER == NUM_LAYERS - 1

# ── condV physics helpers ─────────────────────────────────────────────────────

from causal_ema_scan import causal_ema_scan as _causal_ema_scan

def _causal_ema(xi: torch.Tensor, ema_factor: torch.Tensor,
                floor: float = EMA_FLOOR) -> torch.Tensor:
    """Causal EMA — Triton scan (O(B·N·D) memory vs O(B·D·N·K) conv)."""
    return _causal_ema_scan(xi, ema_factor, floor=floor)


# KdV removed: confirmed null result (kdv_alpha → 0 throughout all runs).
# Deep Research: KdV addresses dispersion in spatial continua; discrete token
# sequences have no such continuum. Attenuation (EMA) is the only mechanism.
# Cited: MEGA (Ma et al. ICLR 2023), JANET (Van der Westhuizen 2018).


def _agc_normalize(pool: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """AGC: normalise to unit RMS per token. No learnable params."""
    D   = pool.shape[-1]
    rms = pool.norm(dim=-1, keepdim=True) / (D ** 0.5)
    return pool / (rms + eps)


# ── Model ─────────────────────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, d, ffn, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(d, ffn)
        self.fc2  = nn.Linear(ffn, d)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class DSQGBlockV6Physics(nn.Module):
    """V8 DSQG attention + condV interference (EMA + KdV + AGC)."""
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 dropout=0.1, interference=False):
        super().__init__()
        self.interference = interference
        self.num_heads    = num_heads
        self.head_dim     = embedding_dim // num_heads
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = DSQGAttentionV6(embedding_dim, num_heads,
                                     seq_len=seq_len, dropout=dropout)
        self.ffn   = FFN(embedding_dim, ffn_dim, dropout)

        if interference:
            self.inter_norm   = nn.LayerNorm(embedding_dim)
            self.inter_gate   = nn.Linear(embedding_dim, embedding_dim)
            self.inter_k_proj = nn.Linear(embedding_dim, embedding_dim)
            self.inter_v_proj = nn.Linear(embedding_dim, embedding_dim)
            # KEY: α₀=0.003 → window ≈ 333t (was 0.03 → 33t → collapsed)
            self.ema_factor = nn.Parameter(torch.full((1,), EMA_INIT))
            # KdV: zero-init → pure EMA at start, activates as needed
            # kdv_alpha removed (null result confirmed)

    def forward(self, x):
        kv_inject = None
        if self.interference:
            xi = self.inter_norm(x)
            B, N, D = xi.shape
            H, HD   = self.num_heads, self.head_dim

            pool = _causal_ema(xi, self.ema_factor, floor=EMA_FLOOR)
            pool = _agc_normalize(pool)

            inter   = torch.sigmoid(self.inter_gate(xi)) * pool
            k_delta = (self.inter_k_proj(inter)
                       .view(B, N, H, HD).permute(0, 2, 1, 3).contiguous())
            v_delta = (self.inter_v_proj(inter)
                       .view(B, N, H, HD).permute(0, 2, 1, 3).contiguous())
            kv_inject = (k_delta, v_delta)

        x = x + self.attn(self.norm1(x), kv_inject=kv_inject)
        x = x + self.ffn(self.norm2(x))
        return x


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
        q, k, v = self.qkv_proj(x).split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True)
        out_flat = out.permute(0, 2, 1, 3).reshape(B, N, D)
        return F.dropout(self.out_proj(out_flat * torch.sigmoid(self.gate_proj(x))),
                         p=self.dropout_p, training=self.training)


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


class AutoresearchTransformerPhysics(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len, full_attn_layer, interference_interval,
                 scale_embed_init_val=0.0, dropout=0.1):
        super().__init__()
        self.embedding       = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed       = nn.Embedding(seq_len + 2, embedding_dim)
        self.drop            = nn.Dropout(dropout)
        self.full_attn_layer = full_attn_layer

        blocks = []
        for i in range(num_layers):
            if i == full_attn_layer:
                blocks.append(FullAttentionBlock(
                    embedding_dim, num_heads, ffn_dim, dropout))
            else:
                has_if = (interference_interval is not None and
                          i % interference_interval == interference_interval - 1)
                blocks.append(DSQGBlockV6Physics(
                    embedding_dim, num_heads, ffn_dim, seq_len,
                    dropout=dropout, interference=has_if))
        self.blocks = nn.ModuleList(blocks)
        self.norm   = nn.LayerNorm(embedding_dim)
        self.out    = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.out.weight = self.embedding.weight
        self._init_weights(scale_embed_init_val)

    def _init_weights(self, scale_embed_init_val):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)
        for m in self.modules():
            if hasattr(m, 'gate_proj') and isinstance(m.gate_proj, nn.Linear):
                nn.init.constant_(m.gate_proj.bias, 0.0)
        for m in self.modules():
            if isinstance(m, DSQGAttentionV6):
                # V10: no phase_base/gain/query_probes/key_probes (removed in V10)
                if scale_embed_init_val != 0.0:
                    nn.init.constant_(m.scale_embed, scale_embed_init_val)

    def forward(self, idx):
        B, N = idx.shape
        pos  = torch.arange(N, device=idx.device).unsqueeze(0)
        x    = self.drop(self.embedding(idx) + self.pos_embed(pos))
        # NoPE for full attention: our full attention already has no RoPE and no
        # per-layer positional re-injection. pos_emb is added only once at the
        # token embedding layer; by L5 the residual stream's position info is
        # entirely implicit in the DSQG-transformed representations. Passing x
        # directly to full attention IS the NoPE configuration — no subtraction
        # needed (subtracting the original pos_emb from a nonlinearly-transformed
        # x adds noise, not position removal).
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def scale_embed_parameters(self):
        for m in self.modules():
            if isinstance(m, DSQGAttentionV6):
                yield m.scale_embed

    def non_scale_embed_parameters(self):
        se_ids = {id(p) for p in self.scale_embed_parameters()}
        for p in self.parameters():
            if id(p) not in se_ids:
                yield p

    def physics_summary(self):
        """Log EMA state for all interference blocks (KdV removed)."""
        entries = []
        for i, block in enumerate(self.blocks):
            if isinstance(block, DSQGBlockV6Physics) and block.interference:
                alpha = block.ema_factor.item()
                win   = round(1.0 / max(abs(alpha), EMA_FLOOR))
                entries.append(f'b{i}: α={alpha:.4f}(w≈{win}t)')
        return '  '.join(entries)


# ── Data utilities ────────────────────────────────────────────────────────────

class BPETokenizerWrapper:
    def __init__(self, tok):
        self.tokenizer = tok
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    def decode(self, ids):
        return self.tokenizer.decode(ids)
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()


def load_data():
    if os.path.exists(FW_CACHE_FILE):
        print(f'Loading FineWeb-Edu from cache: {FW_CACHE_FILE}')
        with open(FW_CACHE_FILE) as fp:
            texts = json.load(fp)
        print(f'  Loaded {len(texts):,} docs from cache')
    else:
        from datasets import load_dataset
        ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                          split='train', streaming=True)
        texts = []
        for item in ds:
            if len(item['text']) < 5_000:
                continue
            texts.append(item['text'])
            if len(texts) >= NUM_DOCS:
                break
        os.makedirs(os.path.dirname(FW_CACHE_FILE), exist_ok=True)
        with open(FW_CACHE_FILE, 'w') as fp:
            json.dump(texts, fp)
    n = len(texts)
    return {'train': texts[:int(n * 0.95)],
            'val':   texts[int(n * 0.95):int(n * 0.95) + 2500]}


def encode_split(split_texts, tokenizer, split_name):
    tokens = []
    for text in split_texts:
        tokens.extend(tokenizer.encode(text))
        tokens.append(3)
    n    = (len(tokens) // MAX_SEQ_LEN) * MAX_SEQ_LEN
    data = torch.tensor(tokens[:n], dtype=torch.long)
    seqs = data.view(-1, MAX_SEQ_LEN)
    print(f'  {split_name}: {len(seqs):,} sequences')
    return seqs


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(data) - BATCH_SIZE + 1, BATCH_SIZE):
        x = data[i:i+BATCH_SIZE, :-1].to(device)
        y = data[i:i+BATCH_SIZE,  1:].to(device)
        logits = model(x)
        loss   = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss   += loss.item() * y.numel()
        total_tokens += y.numel()
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def passkey_accuracy(model, tokenizer, device):
    model.eval()
    filler_ids = tokenizer.encode(_FILLER_SENTENCE)
    cue_ids    = tokenizer.encode(_RETRIEVAL_CUE)
    results    = {}
    for d in PASSKEY_DISTANCES:
        correct, n_valid = 0, 0
        for i in range(PASSKEY_TRIALS):
            target    = _PASSKEY_WORDS[i % len(_PASSKEY_WORDS)]
            others    = [w for w in _PASSKEY_WORDS if w != target]
            intro_ids = tokenizer.encode(_INTRO_TEMPLATE.format(word=target))
            available = MAX_SEQ_LEN - 1 - len(intro_ids) - len(cue_ids) - 1
            if d > available:
                continue
            filler   = []
            while len(filler) < d:
                filler.extend(filler_ids)
            full_seq = intro_ids + filler[:d] + cue_ids
            if len(full_seq) >= MAX_SEQ_LEN:
                continue
            ids    = torch.tensor([full_seq], dtype=torch.long, device=device)
            logits = model(ids)[:, -1, :]
            cand_ids = [(tokenizer.encode(' ' + w) or tokenizer.encode(w))[0]
                        for w in [target] + others[:9]]
            correct  += int(([target] + others[:9])[
                            logits[0][cand_ids].argmax().item()] == target)
            n_valid  += 1
        results[d] = correct / n_valid if n_valid else 0.0
    return results


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    git_hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    print('=' * 70)
    print('  🔬 DWARF Autoresearch — J20D V10 L10 Depth Experiment')
    print('  V10 kernel (J=20) + condV: Kalman-EMA + AGC  [tiled coalesced, 2.87x faster]')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  D={EMBEDDING_DIM}, H={NUM_HEADS}, L={NUM_LAYERS}, FFN={FFN_DIM}')
    print(f'  IF interval={INTERFERENCE}, Full attn layer={FULL_ATTN_LAYER}')
    print(f'  scale_embed init={SCALE_EMBED_INIT_VAL}, LR mult={SCALE_EMBED_LR_MULT}')
    print(f'  EMA α₀={EMA_INIT} (window≈{round(1/EMA_INIT)}t), floor={EMA_FLOOR}')
    print(f'  MAX_TRAIN_SEQS={MAX_TRAIN_SEQS}, LR={LR}, Epochs={SCREEN_EPOCHS}')
    print(f'  git={git_hash}')

    splits = load_data()
    tok_path = next((p for p in TOKENIZER_CANDIDATES if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError(f'Tokenizer not found.')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'Loaded tokenizer from {tok_path}')

    _encoded_cache = 'logs/fineweb_encoded_2048.pt'
    if os.path.exists(_encoded_cache):
        print(f'Loading pre-encoded dataset from {_encoded_cache}')
        _cache     = torch.load(_encoded_cache, weights_only=True)
        train_data = _cache['train']
        val_data   = _cache['val']
    else:
        train_data = encode_split(splits['train'], tokenizer, 'Train')
        val_data   = encode_split(splits['val'],   tokenizer, 'Val')

    if len(train_data) > MAX_TRAIN_SEQS:
        train_data = train_data[torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]]
    if len(val_data) > MAX_VAL_SEQS:
        val_data = val_data[:MAX_VAL_SEQS]
    print(f'  train: {len(train_data):,}  val: {len(val_data):,} seqs')

    model = AutoresearchTransformerPhysics(
        vocab_size=tokenizer.vocab_size(), embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ffn_dim=FFN_DIM,
        seq_len=MAX_SEQ_LEN, full_attn_layer=FULL_ATTN_LAYER,
        interference_interval=INTERFERENCE,
        scale_embed_init_val=SCALE_EMBED_INIT_VAL,
    ).to(device)

    try:
        for i, block in enumerate(model.blocks):
            if type(block).__name__ == "FullAttentionBlock":
                model.blocks[i] = torch.compile(block, fullgraph=False)
                print(f"  compiled FullAttnBlock at layer {i}")
        print("  torch.compile applied to FullAttnBlock(s)")
    except Exception as e:
        print(f"  torch.compile skipped: {e}")

    n_params = model.param_count()
    print(f'Parameters: {n_params:,} ({n_params / 1e6:.1f}M)')

    scale_embed_params     = list(model.scale_embed_parameters())
    non_scale_embed_params = list(model.non_scale_embed_parameters())
    optimizer = torch.optim.AdamW([
        {'params': non_scale_embed_params, 'lr': LR},
        {'params': scale_embed_params,     'lr': LR * SCALE_EMBED_LR_MULT},
    ], weight_decay=0.1, betas=(0.9, 0.95))

    total_steps = SCREEN_EPOCHS * math.ceil(
        len(train_data) / BATCH_SIZE / GRAD_ACCUM)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps)

    best_val_loss   = float('inf')
    passkey_results = {}
    ppl_results     = {}
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(1, SCREEN_EPOCHS + 1):
        model.train()
        indices         = torch.randperm(len(train_data))
        step            = 0
        optimizer.zero_grad()
        steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)

        for acc_step in range(steps_per_epoch):
            for ga in range(GRAD_ACCUM):
                idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
                if idx_start >= len(train_data):
                    continue
                batch = train_data[indices[idx_start:idx_start + BATCH_SIZE]]
                x, y  = batch[:, :-1].to(device), batch[:, 1:].to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x)
                    loss   = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1)) / GRAD_ACCUM
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1

            if step % 200 == 0:
                print(f'  Step {step}/{steps_per_epoch} '
                      f'| Loss {loss.item() * GRAD_ACCUM:.4f}')

        val_loss = evaluate(model, val_data, device)
        val_ppl  = math.exp(min(val_loss, 20))
        ppl_results[epoch] = val_ppl

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR,
                                    f'{git_hash}_j20d_v10_L10_best.pt'))
            marker = ' *'

        print(f'Ep {epoch}/{SCREEN_EPOCHS} | Val PPL {val_ppl:.2f}{marker}')

        se_vals = []
        for m in model.modules():
            if isinstance(m, DSQGAttentionV6):
                se_vals.append(m.scale_embed.detach().abs())
        if se_vals:
            se_all = torch.cat(se_vals)
            print(f'  scale_embed |mean|={se_all.mean():.4f} '
                  f'|max|={se_all.max():.4f}')

        print(f'  Physics: {model.physics_summary()}')

        pk      = passkey_accuracy(model, tokenizer, device)
        pk_mean = sum(pk.values()) / len(pk)
        passkey_results[epoch] = pk_mean * 100
        print(f'  Passkey mean={pk_mean * 100:.1f}%')
        parts = [f'd={d}:{int(pk[d] * 100)}%' for d in PASSKEY_DISTANCES]
        print('  ' + '  '.join(parts))
        sys.stdout.flush()

    elapsed_s     = time.time() - t_start
    memory_mb     = torch.cuda.max_memory_allocated() / 1e6
    passkey_final = passkey_results.get(SCREEN_EPOCHS, 0.0)
    ppl_final     = ppl_results.get(SCREEN_EPOCHS, 999.0)
    PPL_BASELINE     = 61.75
    PASSKEY_BASELINE = 18.3
    ar_score = (passkey_final - PASSKEY_BASELINE) - max(0, ppl_final - PPL_BASELINE) * 0.5

    print('\n---')
    for ep in range(1, SCREEN_EPOCHS + 1):
        print(f'passkey_ep{ep}:    {passkey_results.get(ep, 0.0):.1f}')
    for ep in range(1, SCREEN_EPOCHS + 1):
        print(f'ppl_ep{ep}:        {ppl_results.get(ep, 999.0):.2f}')
    print(f'ar_score:       {ar_score:.2f}')
    print(f'memory_mb:      {memory_mb:.1f}')
    print(f'elapsed_s:      {elapsed_s:.1f}')
    print(f'num_params_M:   {n_params / 1e6:.1f}')
    print(f'num_layers:     {NUM_LAYERS}')
    print(f'num_offsets:    20')
    print(f'scale_embed_lr_mult: {SCALE_EMBED_LR_MULT}')
    print(f'ema_init:       {EMA_INIT}')
    print(f'description:    J20D V10 L10 depth — {n_params/1e6:.1f}M, V8+condV, '
          f'J=20 V9 Frobenius-optimal, KdV removed, NoPE full-attn, EMA_init={EMA_INIT} floor={EMA_FLOOR}, '
          f'121232 seqs, scale_embed_init={SCALE_EMBED_INIT_VAL}')


if __name__ == '__main__':
    train()
