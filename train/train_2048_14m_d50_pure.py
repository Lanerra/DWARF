"""
d50_pure_14m — Pure DSQG with two-θ initialization, no hard masking.

Architecture changes from d49:

  1. d50 offset set (J=44):
     Dense local W=40 (δ=0..40), sparse [128, 384, 1536].
     No bridge offsets — clean geometry.

  2. Two-θ pos_bias initialization (regime_decoupled_wavelet_potential: +22.6%):
     Heads 0-6 (local): pos_bias decay scaled by θ_local=1.5 — steeper penalty
                         at large offsets; drives strong local specialization.
     Head 7 (distal):   pos_bias decay scaled by θ_distal=1.0 — shallower penalty;
                         naturally more open to δ=128/384/1536 relative to local heads.

  3. No hard masking — masking was the root cause of d49 catastrophic failures.
     Specialization emerges from initialization asymmetry + training dynamics.

  4. Removed FullCausalAttention — pure DSQG throughout all layers.

All other components retained from d49:
  - Q-weighted scale gains (V3 kernel, d50 offsets)
  - IF amplifier (per-head gain)
  - Huygens K/V injection
  - Kalman-EMA + KdV soliton + AGC interference
  - Scale_embed soft L2 reg (λ=1e-4), pos_bias clamp (|max|=10.0)

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_2048_14m_d50_pure.py \\
    2>&1 | tee benchmarks/logs/d50_pure_14m.log
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# -- Hyperparameters -----------------------------------------------------------

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
FFN_DIM         = 1024
INTERFERENCE    = 3

# -- FineWeb-Edu dataset config ------------------------------------------------

FW_DATASET_NAME = 'HuggingFaceFW/fineweb-edu'
FW_SUBSET       = 'sample-10BT'
FW_MIN_CHARS    = 5_000
FW_CACHE_FILE   = 'benchmarks/logs/condm_fineweb_edu_doc_cache.json'
MAX_TRAIN_SEQS  = 52_716

# -- Passkey eval config -------------------------------------------------------

PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
PASSKEY_TRIALS    = 5
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'

# -- Save paths ----------------------------------------------------------------

EXPERIMENT_NAME = 'd50_pure_14m'
SAVE_DIR        = 'checkpoints/d50_pure_14m'
RESULT_FILE     = 'benchmarks/logs/d50_pure_14m_results.json'

# -- d50 offset set (J=44) ----------------------------------------------------
#
#   Dense   δ=0..40   (41 offsets)
#   Sparse            [128, 384, 1536]
#
_DENSE_LOCAL_W  = 40
_SPARSE_LIST    = [128, 384, 1536]
_COND_N_OFFSETS = list(range(_DENSE_LOCAL_W + 1)) + _SPARSE_LIST
assert len(_COND_N_OFFSETS) == 44, f"Expected J=44, got {len(_COND_N_OFFSETS)}"

# -- Two-θ design (regime_decoupled_wavelet_potential: +22.6% theoretical gain) -
_THETA_LOCAL  = 1.5
_THETA_DISTAL = 1.0
_WAVELET_COEFFS_LOCAL  = [0.3536, 0.7071, 0.6124, -0.1768]
_WAVELET_COEFFS_DISTAL = [0.4903, 0.8345, 0.2168, -0.1274]

# -- Scale_embed regularization ------------------------------------------------

SCALE_EMBED_REG_LAMBDA = 1e-4
SCALE_EMBED_MAX_ABS    = 1.0

# -- pos_bias regularization ---------------------------------------------------

POS_BIAS_MAX_ABS = 10.0

# -- Triton kernel (d50 offsets, V3 Q-weighted scale gains) --------------------

import pathlib as _pl
_kernel_dir = str(_pl.Path(__file__).parent.parent / 'kernels')
if _kernel_dir not in sys.path:
    sys.path.insert(0, _kernel_dir)

from dsqg_d50_pure import dsqg_attention_v3


# ==============================================================================
#  Model Components
# ==============================================================================

class _KalmanEMA(nn.Module):
    """Kalman-filtered EMA — smoothed recurrent context for interference layers."""
    _EMA_KERNEL_LEN = 256

    def __init__(self, dim):
        super().__init__()
        self.alpha  = nn.Parameter(torch.full((dim,), 0.9))
        self.gain   = nn.Parameter(torch.ones(dim))
        self.dim    = dim

    def forward(self, x):
        B, T, D   = x.shape
        k_len     = min(self._EMA_KERNEL_LEN, T)
        alpha_c   = torch.sigmoid(self.alpha).clamp(0.01, 0.99)
        t         = torch.arange(k_len, device=x.device, dtype=x.dtype)
        kernel    = alpha_c.unsqueeze(1) ** t.unsqueeze(0)
        kernel    = kernel / kernel.sum(dim=1, keepdim=True)
        x_t       = x.transpose(1, 2)
        pad       = F.pad(x_t, (k_len - 1, 0))
        ema       = F.conv1d(pad, kernel.unsqueeze(1), groups=D)
        return ema.transpose(1, 2) * self.gain


class DSQGAttention(nn.Module):
    """
    DSQG attention with d50 geometry and two-θ pos_bias initialization.

    Two-θ design (regime_decoupled_wavelet_potential: +22.6% theoretical gain):
      Heads 0-6 (local): pos_bias decay scaled by θ_local=1.5 — steeper penalty
                          at large offsets; drives strong local specialization.
      Head 7 (distal):   pos_bias decay scaled by θ_distal=1.0 — shallower penalty;
                          naturally more open to δ=128/384/1536 relative to local heads.

    No hard masking — masking was the root cause of d49 catastrophic failures.
    Specialization emerges from initialization asymmetry + training dynamics.
    """

    def __init__(self, embedding_dim, num_heads, seq_len=2048, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        HD             = self.head_dim
        J              = len(_COND_N_OFFSETS)

        self.qkv        = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.out_proj   = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout    = nn.Dropout(dropout)

        # Two-θ pos_bias initialization:
        #   Local heads (0..6): steeper distance penalty (θ_local=1.5)
        #   Distal head (7):    shallower distance penalty (θ_distal=1.0)
        # pos_bias[j, h] = -log(1+δⱼ) * alpha_h
        delta_vals    = torch.tensor([math.log(1.0 + d)
                                      for d in _COND_N_OFFSETS], dtype=torch.float32)
        n_local       = num_heads - 1
        alphas_local  = torch.linspace(0.2, 2.0, n_local) * _THETA_LOCAL
        alpha_distal  = torch.tensor([1.0 * _THETA_DISTAL])
        alphas        = torch.cat([alphas_local, alpha_distal])
        pos_bias_init = -delta_vals.unsqueeze(1) * alphas.unsqueeze(0)
        self.pos_bias   = nn.Parameter(pos_bias_init)

        self.scale_embed = nn.Parameter(torch.zeros(J, HD))

        self.if_gain     = nn.Parameter(torch.ones(num_heads))

    def forward(self, x, return_stats=False):
        B, T, C = x.shape
        H, HD   = self.num_heads, self.head_dim

        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, H, HD).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, T, H, HD).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, T, H, HD).permute(0, 2, 1, 3).contiguous()

        out = dsqg_attention_v3(q, k, v, self.pos_bias, self.scale_embed)
        out = out * self.if_gain.view(1, H, 1, 1)
        out = out.permute(0, 2, 1, 3).reshape(B, T, C)
        out = self.dropout(self.out_proj(out))

        if return_stats:
            with torch.no_grad():
                pb  = self.pos_bias.detach().cpu()
                se  = self.scale_embed.detach().cpu()
                ig  = self.if_gain.detach().cpu()
                stats = {
                    "scale_embed_abs_mean": se.abs().mean().item(),
                    "scale_embed_abs_max":  se.abs().max().item(),
                    "pos_bias_abs_mean":    pb.abs().mean().item(),
                    "pos_bias_abs_max":     pb.abs().max().item(),
                    "pos_bias_mean_per_head": pb.mean(0).tolist(),
                    "if_gain":              ig.tolist(),
                }
            return out, stats
        return out


class InterferenceLayer(nn.Module):
    """Kalman-EMA + KdV soliton + AGC interference pooling."""

    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len, dropout=0.1):
        super().__init__()
        self.attn  = DSQGAttention(embedding_dim, num_heads, seq_len, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        self.ema   = _KalmanEMA(embedding_dim)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta  = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))

        H, HD = num_heads, embedding_dim // num_heads
        J     = len(_COND_N_OFFSETS)
        self.huygens_k = nn.Parameter(torch.zeros(J, HD))
        self.huygens_v = nn.Parameter(torch.zeros(J, HD))
        self.huygens_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, return_stats=False):
        hscale = torch.tanh(self.huygens_scale)

        res = self.norm1(x)
        if return_stats:
            attn_out, stats = self.attn(res, return_stats=True)
        else:
            attn_out = self.attn(res)
            stats    = {}

        x = x + attn_out

        kdv = torch.tanh(self.alpha) * (x * x.roll(1, dims=1))
        x   = x + kdv

        agc_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x        = x + torch.tanh(self.beta) * x / agc_norm

        ema_ctx = self.ema(x)
        x       = x + torch.tanh(self.gamma) * ema_ctx

        x = x + self.ffn(self.norm2(x))
        return (x, stats) if return_stats else x


class TransformerBlock(nn.Module):
    """Standard transformer block (non-interference layers)."""

    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attn  = DSQGAttention(embedding_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_stats=False):
        res = self.norm1(x)
        if return_stats:
            attn_out, stats = self.attn(res, return_stats=True)
        else:
            attn_out = self.attn(res)
            stats    = {}
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return (x, stats) if return_stats else x


class DSQGTransformer(nn.Module):

    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads,
                 ffn_dim, seq_len=2048, dropout=0.1, interference=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_enc   = nn.Embedding(seq_len, embedding_dim)
        self.drop      = nn.Dropout(dropout)

        layers = []
        for i in range(num_layers):
            if (i + 1) % interference == 0:
                layers.append(InterferenceLayer(
                    embedding_dim, num_heads, ffn_dim, seq_len,
                    dropout=dropout))
            else:
                layers.append(TransformerBlock(
                    embedding_dim, num_heads, ffn_dim, dropout=dropout))
        self.layers  = nn.ModuleList(layers)
        self.norm    = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.02)

    def forward(self, input_ids, return_stats=False):
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device)
        x    = self.drop(self.embedding(input_ids) + self.pos_enc(pos))

        all_stats = []
        for layer in self.layers:
            if return_stats:
                x, s = layer(x, return_stats=True)
                all_stats.append(s)
            else:
                x = layer(x)

        x      = self.norm(x)
        logits = self.lm_head(x)
        return (logits, all_stats) if return_stats else logits


# ==============================================================================
#  Dataset & Tokenizer
# ==============================================================================

class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)
    def vocab_size(self):   return self.tokenizer.get_vocab_size()


def load_tokenizer():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates  = [
        os.path.join(_script_dir, '..', 'results', '2048_condI_tokenizer.json'),
        os.path.join(_script_dir, 'results', '2048_condI_tokenizer.json'),
        os.path.join(_script_dir, '2048_condI_tokenizer.json'),
    ]
    tok_path = next((p for p in candidates if os.path.exists(p)), None)
    if tok_path is None:
        raise FileNotFoundError('condI BPE tokenizer not found in results/')
    from tokenizers import Tokenizer
    tok = BPETokenizerWrapper(Tokenizer.from_file(tok_path))
    print(f'Loaded tokenizer from {tok_path}  (vocab={tok.vocab_size()})')
    return tok


def load_dataset(tokenizer):
    """Load pre-encoded seqs from cache, or build cache from raw data."""
    encoded_cache = 'logs/fineweb_encoded_2048.pt'
    if os.path.exists(encoded_cache):
        print(f'Loading pre-encoded dataset from {encoded_cache} …')
        cache      = torch.load(encoded_cache, weights_only=True)
        train_data = cache['train']
        val_data   = cache['val']
        test_data  = cache['test']
    else:
        import datasets
        print('Streaming FineWeb-Edu (no cache found) …')
        ds   = datasets.load_dataset(FW_DATASET_NAME, FW_SUBSET, split='train',
                                     streaming=True, trust_remote_code=True)
        all_ids = []
        n_docs  = 0
        for ex in ds:
            if len(ex['text']) < FW_MIN_CHARS:
                continue
            all_ids.extend(tokenizer.encode(ex['text']))
            n_docs += 1
            if n_docs >= NUM_DOCS:
                break
        seqs = []
        for i in range(0, len(all_ids) - MAX_SEQ_LEN, MAX_SEQ_LEN):
            seqs.append(all_ids[i:i + MAX_SEQ_LEN + 1])
        import random; random.shuffle(seqs)
        n_val  = min(2000, len(seqs) // 10)
        n_test = min(2000, len(seqs) // 10)
        val_t   = torch.tensor(seqs[:n_val],             dtype=torch.long)
        test_t  = torch.tensor(seqs[n_val:n_val+n_test], dtype=torch.long)
        train_t = torch.tensor(seqs[n_val+n_test:],      dtype=torch.long)
        torch.save({'train': train_t, 'val': val_t, 'test': test_t}, encoded_cache)
        train_data, val_data, test_data = train_t, val_t, test_t

    if len(train_data) > MAX_TRAIN_SEQS:
        idx        = torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]
        train_data = train_data[idx]
    print(f'  train: {len(train_data):,}  val: {len(val_data):,}  '
          f'test: {len(test_data):,} seqs')
    return train_data, val_data, test_data


# ==============================================================================
#  Training
# ==============================================================================

def run_passkey_eval(model, tokenizer, device):
    model.eval()
    results = []
    with torch.no_grad():
        for dist in PASSKEY_DISTANCES:
            correct = 0
            n_valid = 0
            trials  = []
            for word in _PASSKEY_WORDS[:PASSKEY_TRIALS]:
                intro   = _INTRO_TEMPLATE.format(word=word)
                fillers = (_FILLER_SENTENCE * (dist * 10))[:dist * 80]
                prompt  = intro + ' ' + fillers + ' ' + _RETRIEVAL_CUE
                ids     = tokenizer.encode(prompt)
                if len(ids) >= MAX_SEQ_LEN:
                    trials.append({'distance': dist, 'target': word,
                                   'predicted': None, 'correct': False, 'skipped': True})
                    continue
                inp     = torch.tensor([ids], device=device)
                logits  = model(inp)
                pred_id = logits[0, -1].argmax().item()
                pred    = tokenizer.decode([pred_id]).strip()
                ok      = (pred == word)
                correct += int(ok)
                n_valid += 1
                trials.append({'distance': dist, 'target': word,
                               'predicted': pred, 'correct': ok, 'skipped': False})
            acc = correct / n_valid if n_valid > 0 else 0.0
            results.append({'distance': dist, 'accuracy': acc,
                            'n_valid': n_valid, 'trials': trials})
    return results


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Offsets ({len(_COND_N_OFFSETS)}): {_COND_N_OFFSETS}')
    print(f'Two-θ design: θ_local={_THETA_LOCAL} (heads 0-6), θ_distal={_THETA_DISTAL} (head 7)')
    print(f'Wavelet coefficients (provenance only):')
    print(f'  local:  {[round(c,4) for c in _WAVELET_COEFFS_LOCAL]}')
    print(f'  distal: {[round(c,4) for c in _WAVELET_COEFFS_DISTAL]}')

    tokenizer                    = load_tokenizer()
    train_data, val_data, test_data = load_dataset(tokenizer)

    model = DSQGTransformer(
        vocab_size      = VOCAB_SIZE,
        embedding_dim   = EMBEDDING_DIM,
        num_layers      = NUM_LAYERS,
        num_heads       = NUM_HEADS,
        ffn_dim         = FFN_DIM,
        seq_len         = MAX_SEQ_LEN,
        interference    = INTERFERENCE,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    total_steps = (len(train_data) // (BATCH_SIZE * GRAD_ACCUM)) * NUM_EPOCHS
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=LR * 0.1)

    os.makedirs(SAVE_DIR, exist_ok=True)

    all_results = []
    best_val    = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Training ──────────────────────────────────────────────────────
        model.train()
        perm        = torch.randperm(len(train_data))
        total_loss  = 0.0
        total_tokens = 0
        t0 = time.time()

        optimizer.zero_grad()
        step_count = 0

        for batch_idx in range(0, len(train_data) - BATCH_SIZE + 1, BATCH_SIZE):
            idxs  = perm[batch_idx:batch_idx + BATCH_SIZE]
            batch = train_data[idxs].to(device)
            inp   = batch[:, :-1]
            tgt   = batch[:, 1:]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(inp)
                loss   = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))

                se_reg = torch.tensor(0.0, device=device)
                for layer in model.layers:
                    attn = getattr(layer, 'attn', None)
                    if attn is not None and hasattr(attn, 'scale_embed'):
                        se_norm = attn.scale_embed.norm()
                        se_reg  = se_reg + se_norm * se_norm
                loss = (loss + SCALE_EMBED_REG_LAMBDA * se_reg) / GRAD_ACCUM

            loss.backward()
            total_loss   += loss.item() * GRAD_ACCUM
            total_tokens += tgt.numel()
            step_count   += 1

            if step_count % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                with torch.no_grad():
                    for layer in model.layers:
                        attn = getattr(layer, 'attn', None)
                        if attn is not None and hasattr(attn, 'pos_bias'):
                            attn.pos_bias.clamp_(-POS_BIAS_MAX_ABS, POS_BIAS_MAX_ABS)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        epoch_loss = total_loss / step_count
        elapsed    = time.time() - t0
        print(f'[Ep {epoch}] train_loss={epoch_loss:.4f} '
              f'({elapsed:.0f}s, {total_tokens/elapsed:.0f} tok/s)')

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        n_val_use  = min(512, len(val_data))
        val_loss   = 0.0
        val_tokens = 0
        attn_stats = []

        with torch.no_grad():
            for i in range(0, n_val_use, BATCH_SIZE):
                batch = val_data[i:i + BATCH_SIZE].to(device)
                inp   = batch[:, :-1]
                tgt   = batch[:, 1:]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    if i == 0:
                        logits, stats = model(inp, return_stats=True)
                        attn_stats = stats
                    else:
                        logits = model(inp)
                loss      = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE),
                                            tgt.reshape(-1))
                val_loss  += loss.item() * tgt.numel()
                val_tokens += tgt.numel()

        val_ppl = math.exp(val_loss / val_tokens)
        print(f'[Ep {epoch}] val_ppl={val_ppl:.3f}')

        for li, s in enumerate(attn_stats):
            se_max = s.get('scale_embed_abs_max', 0)
            pb_max = s.get('pos_bias_abs_max', 0)
            ig     = s.get('if_gain', [])
            if se_max > 0 or pb_max > 0:
                print(f'  L{li}: se_max={se_max:.4f} pb_max={pb_max:.4f} '
                      f'if_gain={[round(g,3) for g in ig]}')

        # ── Passkey eval ──────────────────────────────────────────────────
        passkey = run_passkey_eval(model, tokenizer, device)
        acc_per_dist = {r['distance']: r['accuracy'] for r in passkey}
        mean_acc     = sum(r['accuracy'] for r in passkey
                          if not all(t['skipped'] for t in r['trials'])) / max(
                          1, sum(1 for r in passkey
                                 if not all(t['skipped'] for t in r['trials'])))
        per_dist_str = [f"d{r['distance']}:{r['accuracy']:.0%}" for r in passkey]
        print(f'[Ep {epoch}] passkey_mean={mean_acc:.1%}  per_dist={per_dist_str}')

        # ── Save ──────────────────────────────────────────────────────────
        result = {
            'epoch':       epoch,
            'val_ppl':     val_ppl,
            'train_loss':  epoch_loss,
            'passkey':     passkey,
            'passkey_mean': mean_acc,
            'attn_stats':  attn_stats,
        }
        all_results.append(result)

        ckpt_path = os.path.join(SAVE_DIR, f'epoch_{epoch:02d}.pt')
        torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_ppl': val_ppl}, ckpt_path)

        if val_ppl < best_val:
            best_val = val_ppl
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_ppl': val_ppl},
                       os.path.join(SAVE_DIR, 'best.pt'))
            print(f'[Ep {epoch}] ✓ new best: {val_ppl:.3f}')

        with open(RESULT_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    train()
