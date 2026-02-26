"""
Wave Field V4D — Dispersion Ablation at 2048 Tokens
=====================================================
Compares two 13.5M Wave Field models head-to-head:

  Condition A: V4 improvements only (no dispersion)
  Condition B: V4 improvements + learned dispersion (V4D)

V4 improvements adopted (from 4A Lab fork analysis):
  - Log-scale frequency init: logspace(0.1, 10.0) — octave coverage
  - Better damping init: linspace(-2.0, 0.0) — less aggressive decay
  - Full 2π phase coverage: linspace(0, 2π·(H-1)/H) — head diversity
  - Float32 scatter accumulation — AMP numerical stability
  - +0.05 damping floor — stability guard
  
V4 improvements NOT adopted:
  - Softmax kernel normalization — eliminates negative kernel values,
    removes wave oscillation character, conflicts with dispersion
  - V4 parameter-change caching — no benefit during training
    (params change every step); explicit rebuild is correct for training
  - Sinusoidal PE — adds confounding variable

V4D additions (ours):
  - Dynamic stride: actual_stride = (G-1)/(N-1) computed at forward time
    (critical for 2048 tokens — V4 fork still has this bug)
  - Learned β per head: wave_dispersion init to ones, data-driven
    K_disp(f) = K(f) · exp(i·β·(f/G)²·2π)
    β=0 → non-dispersive; β>0 → normal; β<0 → anomalous

What this ablation measures:
  - Does dispersion improve PPL beyond V4 improvements alone?
  - Do β values differentiate across heads (multi-scale self-organization)?
  - Does convergence speed change (schedule effect)?

Reference:
  - V3.5 baseline (current run): ~PPL 95-115 (in progress)
  - Standard 13.5M @ 2048: PPL 64.5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import json
import gc

# ======================================================================
# WAVE FIELD ATTENTION V4D
# ======================================================================

class WaveFieldAttentionV4D(nn.Module):
    """
    Wave Field Attention with V4 stability improvements + optional dispersion.

    V4 improvements (from 4A Lab fork):
      - Log-scale frequency initialization
      - Better damping / phase initialization
      - Float32 scatter accumulation for AMP stability
      - Minimum damping floor (0.05)

    V4D additions (ours):
      - Dynamic stride: positions computed from actual N, not max_seq_len
      - Learned dispersion coefficient β per head (if use_dispersion=True)
    """

    def __init__(self, embedding_dim, num_heads, field_size=4096,
                 max_seq_len=2048, dropout=0.1, use_dispersion=True):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads      = num_heads
        self.head_dim       = embedding_dim // num_heads
        self.field_size     = field_size
        self.max_seq_len    = max_seq_len
        self.use_dispersion = use_dispersion

        assert embedding_dim % num_heads == 0

        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim)

        # ── V4: log-scale frequency init ─────────────────────────────────────
        # Geometric spacing: each head gets a genuine octave rather than a
        # linear increment. Low-freq heads → long-range structure; high-freq
        # heads → local syntax.
        self.wave_frequency = nn.Parameter(
            torch.logspace(math.log10(0.1), math.log10(10.0), num_heads)
        )

        # ── V4: less aggressive damping init ─────────────────────────────────
        # V3.5 used linspace(-3.0, 0.5); heavy negative values meant waves
        # decayed extremely fast.  linspace(-2.0, 0.0) allows longer reach.
        self.wave_damping = nn.Parameter(
            torch.linspace(-2.0, 0.0, num_heads)
        )

        # ── V4: full 2π phase coverage ────────────────────────────────────────
        # V3.5 used linspace(0, π) — only half-cycle coverage.
        self.wave_phase = nn.Parameter(
            torch.linspace(0.0, 2.0 * math.pi * (num_heads - 1) / num_heads, num_heads)
        )

        # ── V4D: per-head dispersion coefficient β ────────────────────────────
        # Initialised to one (active dispersion start).  Training will push β
        # toward values the network finds useful; β=0 would make the dispersion
        # branch unreachable (guard at forward skips computation when |β|<1e-8,
        # killing gradients entirely — confirmed empirically in first ablation).
        # β > 0 → high-freq components travel faster (normal dispersion)
        # β < 0 → low-freq components travel faster (anomalous dispersion)
        if use_dispersion:
            self.wave_dispersion = nn.Parameter(torch.ones(num_heads))
        else:
            # Fixed at zero — not a learnable parameter
            self.register_buffer('wave_dispersion', torch.zeros(num_heads))

        # Content-dependent gating (bias=2.0 → starts open)
        self.gate_proj = nn.Linear(embedding_dim, embedding_dim)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.0)

        # Cross-head field coupling
        self.field_coupling = nn.Parameter(
            torch.eye(num_heads) + torch.randn(num_heads, num_heads) * 0.01
        )

        self.scale = math.sqrt(self.head_dim)

    def _build_wave_kernels(self, device):
        """
        Build left-aligned causal wave kernels with V4 stability improvements
        and optional dispersive phase rotation.

        Kernel construction (time domain, then FFT):
          k(t) = exp(-(α+0.05)·t) · cos(ω·t + φ)   [V4: floor on α]
          Normalised by L1 sum (NOT softmax — preserves negative values
          needed for wave oscillation and dispersion)

        Dispersive phase (frequency domain, applied after FFT):
          K_disp(f) = K(f) · exp(i · β · (f/G)² · 2π)
          β=0 → identical to baseline; β≠0 → frequency-dependent phase velocity
        """
        G = self.field_size
        H = self.num_heads

        t = torch.arange(G, dtype=torch.float32, device=device)

        # V4: |ω| to ensure positive frequency; α floor at 0.05
        alpha = (F.softplus(self.wave_damping) + 0.05).unsqueeze(1)   # [H, 1]
        omega = self.wave_frequency.abs().unsqueeze(1)                  # [H, 1]
        phi   = self.wave_phase.unsqueeze(1)                            # [H, 1]

        # Time-domain damped cosine kernel
        kernels = torch.exp(-alpha * t.unsqueeze(0)) * torch.cos(omega * t.unsqueeze(0) + phi)

        # L1 normalisation (preserves sign — critical for wave character)
        kernel_sum = kernels.abs().sum(dim=1, keepdim=True).clamp(min=1e-8)
        kernels = kernels / kernel_sum

        # FFT (zero-padded for linear convolution)
        kernel_fft = torch.fft.rfft(kernels, n=2 * G)  # [H, G+1] complex

        # ── Dispersive phase rotation (V4D) ───────────────────────────────────
        # Applied in frequency domain after the base FFT.
        # f_norm = f / G ranges [0, 1]; quadratic phase = chirp in position space.
        # Because β is in wave_dispersion (an nn.Parameter or buffer),
        # gradients flow through this operation during training.
        beta = self.wave_dispersion.to(device)  # [H]
        if beta.abs().max() > 1e-8:
            f_indices = torch.arange(G + 1, dtype=torch.float32, device=device)  # [G+1]
            f_norm    = f_indices / G                                              # [G+1]
            disp_phase = beta.unsqueeze(1) * f_norm.unsqueeze(0).pow(2) * 2.0 * math.pi
            # exp(i·θ) = cos(θ) + i·sin(θ)
            disp_rot = torch.complex(torch.cos(disp_phase), torch.sin(disp_phase))
            kernel_fft = kernel_fft * disp_rot

            # ── Causal enforcement ────────────────────────────────────────────
            # Dispersion rotates phase in frequency domain, which can create
            # negative-time (acausal) components in the time-domain kernel —
            # energy at indices G..2G-1 in the 2G-point DFT, corresponding to
            # t < 0.  When the wave convolution runs, those components reach
            # future field positions (field[g+|t|]), breaking causality.
            # β=0 (no dispersion) is safe; any β≠0 requires this correction.
            # Fix: convert to time domain, zero the acausal tail, convert back.
            kernel_time = torch.fft.irfft(kernel_fft, n=2 * G)  # [H, 2G]
            kernel_time[:, G:] = 0.0                             # zero t < 0
            kernel_fft = torch.fft.rfft(kernel_time, n=2 * G)   # back to freq

        return kernel_fft

    def _wave_convolve(self, field, kernel_fft):
        """Linear convolution via zero-padded FFT. O(G log G) per head."""
        B, H, G, D = field.shape
        pad_size = 2 * G

        # Permute to (B*D, H, G) for batched FFT
        field_t   = field.permute(0, 3, 1, 2).reshape(B * D, H, G)
        field_fft = torch.fft.rfft(field_t, n=pad_size)
        conv_fft  = field_fft * kernel_fft.unsqueeze(0)
        convolved = torch.fft.irfft(conv_fft, n=pad_size)[:, :, :G]

        return convolved.reshape(B, D, H, G).permute(0, 2, 3, 1)

    def _bilinear_scatter(self, values, field_pos_float, B, H, G, head_dim, device):
        """
        Scatter values onto field using bilinear interpolation.

        V4: accumulate in float32 regardless of input dtype — prevents
        precision loss when many float16 values accumulate in scatter_add_.
        """
        N = field_pos_float.shape[0]

        idx_lo = field_pos_float.long().clamp(0, G - 2)
        idx_hi = idx_lo + 1
        frac   = (field_pos_float - idx_lo.float()).clamp(0, 1)
        w_lo   = (1.0 - frac).view(1, 1, N, 1)
        w_hi   = frac.view(1, 1, N, 1)

        # Float32 accumulation (V4 improvement)
        field = torch.zeros(B, H, G, head_dim, device=device, dtype=torch.float32)

        idx_lo_exp = idx_lo.view(1, 1, N, 1).expand(B, H, -1, head_dim)
        idx_hi_exp = idx_hi.view(1, 1, N, 1).expand(B, H, -1, head_dim)

        field.scatter_add_(2, idx_lo_exp, (values * w_lo).float())
        field.scatter_add_(2, idx_hi_exp, (values * w_hi).float())

        # Convert back to input dtype
        return field.to(values.dtype)

    def _bilinear_gather(self, field, field_pos_float):
        """Read from field using bilinear interpolation."""
        B, H, G, D = field.shape
        N = field_pos_float.shape[0]

        idx_lo = field_pos_float.long().clamp(0, G - 2)
        idx_hi = idx_lo + 1
        frac   = (field_pos_float - idx_lo.float()).clamp(0, 1)
        w_lo   = (1.0 - frac).view(1, 1, N, 1)
        w_hi   = frac.view(1, 1, N, 1)

        idx_lo_exp = idx_lo.view(1, 1, N, 1).expand(B, H, -1, D)
        idx_hi_exp = idx_hi.view(1, 1, N, 1).expand(B, H, -1, D)

        return torch.gather(field, 2, idx_lo_exp) * w_lo \
             + torch.gather(field, 2, idx_hi_exp) * w_hi

    def _apply_field_coupling(self, field):
        """Static cross-head field coupling."""
        B, H, G, D = field.shape
        coupling   = F.softmax(self.field_coupling, dim=-1)
        field_flat = field.reshape(B, H, G * D)
        coupled    = torch.bmm(coupling.unsqueeze(0).expand(B, -1, -1), field_flat)
        return coupled.reshape(B, H, G, D)

    def forward(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B, N, D = x.shape
        G       = self.field_size
        H       = self.num_heads
        hd      = self.head_dim

        # QKV
        qkv  = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, H, hd).transpose(1, 2)
        k = k.view(B, N, H, hd).transpose(1, 2)
        v = v.view(B, N, H, hd).transpose(1, 2)

        # V4D: dynamic stride — positions always span [0, G-1] regardless of N
        actual_stride  = (G - 1) / max(N - 1, 1)
        field_pos      = (torch.arange(N, device=x.device, dtype=torch.float32)
                          * actual_stride).clamp(0, G - 2)

        # Deposit = V * K_mag
        k_mag   = k.norm(dim=-1, keepdim=True)
        deposit = v * k_mag

        # Scatter → Wave → Couple → Gather
        field    = self._bilinear_scatter(deposit, field_pos, B, H, G, hd, x.device)
        kfft     = self._build_wave_kernels(x.device)
        field    = self._wave_convolve(field, kfft)
        field    = self._apply_field_coupling(field)

        # Gating
        gate     = torch.sigmoid(self.gate_proj(x))
        gate     = gate.view(B, N, H, hd).transpose(1, 2)
        gathered = self._bilinear_gather(field, field_pos)
        output   = (gathered * gate).transpose(1, 2).reshape(B, N, D)
        output   = self.out_proj(output)

        return output.squeeze(0) if squeeze else output


# ======================================================================
# TRANSFORMER USING V4D ATTENTION
# ======================================================================

class WaveFieldTransformerV4D(nn.Module):
    """Drop-in V4D transformer for the ablation."""

    def __init__(self, vocab_size, embedding_dim=256, num_layers=6,
                 num_heads=8, ffn_dim=1024, field_size=4096,
                 max_seq_len=2050, dropout=0.1, use_dispersion=True,
                 use_checkpoint=False, interference_interval=3):
        super().__init__()
        self.vocab_size       = vocab_size
        self.use_checkpoint   = use_checkpoint
        self.interference_interval = interference_interval

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout         = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            attn = WaveFieldAttentionV4D(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                field_size=field_size,
                max_seq_len=max_seq_len,
                dropout=dropout,
                use_dispersion=use_dispersion,
            )
            ffn = nn.Sequential(
                nn.Linear(embedding_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embedding_dim),
                nn.Dropout(dropout),
            )
            self.layers.append(nn.ModuleDict({'attn': attn, 'ffn': ffn}))

        self.norms  = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_layers)])
        self.final_norm         = nn.LayerNorm(embedding_dim)
        self.output_projection  = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, mask=None):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        x = self.dropout(self.token_embedding(input_ids))

        for i, layer in enumerate(self.layers):
            if self.use_checkpoint:
                x = x + torch.utils.checkpoint.checkpoint(
                    layer['attn'], self.norms[i](x))
            else:
                x = x + layer['attn'](self.norms[i](x))
            x = x + layer['ffn'](self.norms2[i](x))

        logits = self.output_projection(self.final_norm(x))
        loss   = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def dispersion_summary(self):
        """Return per-layer, per-head β values and summary statistics."""
        betas = []
        for i, layer in enumerate(self.layers):
            attn = layer['attn']
            if attn.use_dispersion:
                b = attn.wave_dispersion.detach().cpu()
                betas.append(b)
        if not betas:
            return None
        return torch.stack(betas)  # [num_layers, num_heads]


# ======================================================================
# TOKENIZER + DATA  (reused from train_2048_benchmark.py)
# ======================================================================

def train_bpe_tokenizer(train_texts, vocab_size=32000):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder       = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=['<pad>', '<unk>', '<bos>', '<eos>'],
        min_frequency=2,
    )
    tokenizer.train_from_iterator(train_texts, trainer=trainer)
    return tokenizer


class BPEWrapper:
    def __init__(self, tok):
        self.tokenizer = tok
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    def decode(self, ids):
        return self.tokenizer.decode(ids)
    def vocab_size_actual(self):
        return self.tokenizer.get_vocab_size()


def load_openwebtext(max_docs=100000):
    from datasets import load_dataset
    print(f'Loading OpenWebText (up to {max_docs:,} docs)...')
    ds = load_dataset('openwebtext', split='train', streaming=True)
    texts = []
    for i, item in enumerate(ds):
        if i >= max_docs: break
        t = item['text'].strip()
        if len(t) > 50: texts.append(t)
        if (i + 1) % 25000 == 0:
            print(f'  {i+1:,} docs...')
    n  = len(texts)
    t1 = int(n * 0.95); t2 = int(n * 0.975)
    print(f'  {n:,} docs | train {t1:,} | val {t2-t1:,} | test {n-t2:,}')
    return {'train': texts[:t1], 'valid': texts[t1:t2], 'test': texts[t2:]}


def encode_lines(lines, tok, max_seq_len):
    data = []
    for line in lines:
        ids = tok.encode(line)
        if len(ids) < 2: continue
        for s in range(0, len(ids) - 1, max_seq_len):
            chunk = ids[s:s + max_seq_len + 1]
            if len(chunk) >= 2:
                data.append((torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])))
    return data


def create_batches(data, batch_size, device, shuffle=True):
    idx = torch.randperm(len(data)).tolist() if shuffle else list(range(len(data)))
    batches = []
    for s in range(0, len(idx), batch_size):
        bi = idx[s:s + batch_size]
        bx = [data[i][0] for i in bi]; by = [data[i][1] for i in bi]
        ml = max(x.size(0) for x in bx)
        px = torch.zeros(len(bx), ml, dtype=torch.long, device=device)
        py = torch.full((len(by), ml), -100, dtype=torch.long, device=device)
        for i, (x, y) in enumerate(zip(bx, by)):
            px[i, :x.size(0)] = x; py[i, :y.size(0)] = y
        batches.append((px, py))
    return batches


# ======================================================================
# TRAINING
# ======================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup    = warmup_steps
        self.total     = total_steps
        self.min_lr    = min_lr
        self.base_lr   = optimizer.param_groups[0]['lr']
        self.step_n    = 0

    def step(self):
        self.step_n += 1
        if self.step_n <= self.warmup:
            lr = self.base_lr * self.step_n / self.warmup
        else:
            p  = (self.step_n - self.warmup) / max(1, self.total - self.warmup)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * p))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr


@torch.no_grad()
def evaluate(model, batches, vocab_size, device, use_amp=False):
    model.eval()
    tl, tc, tt, n = 0, 0, 0, 0
    for x, y in batches:
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1),
                                   ignore_index=-100)
        tl += loss.item(); n += 1
        mask = y != -100
        tc += (logits.argmax(-1)[mask] == y[mask]).sum().item()
        tt += mask.sum().item()
    model.train()
    al = tl / max(n, 1)
    return al, math.exp(min(al, 20)), tc / max(tt, 1) * 100


def train_model(model, train_data, val_data, tok, vocab_size, device,
                name, num_epochs=10, batch_size=8, grad_accum=4,
                peak_lr=3e-4, use_amp=True, save_dir='checkpoints',
                resume_from=None, start_epoch=1, prev_best_vl=float('inf')):
    os.makedirs(save_dir, exist_ok=True)
    params  = sum(p.numel() for p in model.parameters())
    eff_bs  = batch_size * grad_accum
    print(f'\n  {name}: {params:,} params | batch {batch_size}×{grad_accum}={eff_bs} | lr {peak_lr}')

    opt  = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                              weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8)
    spe  = math.ceil(len(train_data) / batch_size)
    sch  = WarmupCosineScheduler(opt, spe * 1, spe * num_epochs, min_lr=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if resume_from is not None and os.path.exists(resume_from):
        model.load_state_dict(torch.load(resume_from, weights_only=True))
        # Fast-forward scheduler to the correct position (approximate per-epoch steps)
        opt_steps_per_epoch = math.ceil(spe / grad_accum)
        steps_to_skip = (start_epoch - 1) * opt_steps_per_epoch
        for _ in range(steps_to_skip):
            sch.step()
        print(f'  Resumed from {resume_from}')
        print(f'  Scheduler fast-forwarded {steps_to_skip} steps ({start_epoch-1} epochs)')

    best_vl = prev_best_vl; best_vp = float('inf')
    best_va = 0; best_ep = start_epoch - 1
    t0 = time.time()

    for epoch in range(start_epoch, num_epochs + 1):
        et = time.time()
        model.train()
        batches = create_batches(train_data, batch_size, device)
        tl, nb  = 0, 0
        opt.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(batches):
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1),
                                       ignore_index=-100) / grad_accum
            if torch.isnan(loss) or torch.isinf(loss):
                opt.zero_grad(set_to_none=True); continue
            scaler.scale(loss).backward()
            tl += loss.item() * grad_accum; nb += 1

            if (step + 1) % grad_accum == 0 or (step + 1) == len(batches):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sch.step()
                opt.zero_grad(set_to_none=True)

            if (step + 1) % 200 == 0:
                print(f'    Step {step+1}/{len(batches)} | Loss {tl/nb:.4f}', flush=True)

        al = tl / max(nb, 1); et = time.time() - et
        vb = create_batches(val_data, batch_size, device, shuffle=False)
        vl, vp, va = evaluate(model, vb, vocab_size, device, use_amp)

        mk = ''
        if vl < best_vl:
            best_vl, best_vp, best_va, best_ep = vl, vp, va, epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))
            mk = ' * BEST'
        print(f'  Ep {epoch:3d}/{num_epochs} | Train {al:.4f} | Val {vl:.4f} '
              f'PPL {vp:.1f} Acc {va:.1f}% | {et:.0f}s{mk}')

    model.load_state_dict(torch.load(os.path.join(save_dir, 'best.pt'), weights_only=True))
    return {'model_name': name, 'params': params, 'best_ppl': best_vp,
            'best_acc': best_va, 'best_epoch': best_ep,
            'total_time': time.time() - t0}


# ======================================================================
# DISPERSION ANALYSIS
# ======================================================================

def print_dispersion_analysis(model, name):
    """Print per-head β values and their distribution."""
    betas = model.dispersion_summary()
    if betas is None:
        print(f'\n  {name}: dispersion disabled')
        return

    L, H = betas.shape
    print(f'\n  ── {name} — learned β values ──')
    print(f'  {"Layer":<8} ' + ' '.join(f'h{h:<5}' for h in range(H)))
    print(f'  {"─"*8} ' + '─' * (H * 7))
    for i in range(L):
        row = '  '.join(f'{betas[i, h].item():+.3f}' for h in range(H))
        print(f'  Layer {i:<2}  {row}')

    all_b = betas.flatten()
    print(f'\n  Summary: mean={all_b.mean():.3f}  std={all_b.std():.3f}  '
          f'min={all_b.min():.3f}  max={all_b.max():.3f}')

    # Count heads by dispersion direction
    near_zero  = (all_b.abs() < 0.1).sum().item()
    normal     = (all_b >  0.1).sum().item()
    anomalous  = (all_b < -0.1).sum().item()
    total      = L * H
    print(f'  Head roles: {near_zero}/{total} near-zero  |  '
          f'{normal}/{total} normal (β>0.1)  |  '
          f'{anomalous}/{total} anomalous (β<-0.1)')

    if all_b.std() > 0.1:
        print(f'  ✓ Heads DIFFERENTIATED — multi-scale self-organization detected')
    else:
        print(f'  ○ Heads relatively uniform — dispersion may need more epochs or data')


# ======================================================================
# MAIN
# ======================================================================

def main():
    max_seq_len = 2048

    print('=' * 70)
    print('  WAVE FIELD V4D — DISPERSION ABLATION @ 2048 TOKENS')
    print('  Condition A: V4 improvements (no dispersion)')
    print('  Condition B: V4 improvements + learned dispersion (V4D)')
    print('=' * 70)
    print('\n  V4 improvements: log-scale freq init, better damping init,')
    print('  full 2π phase, float32 scatter accumulation, +0.05 damping floor')
    print('  V4D addition: dynamic stride + learned β per head')

    splits = load_openwebtext(max_docs=100000)

    bpe_vocab = 32000
    print(f'\nTraining BPE tokenizer (vocab={bpe_vocab})...')
    raw_tok = train_bpe_tokenizer(splits['train'][:50000], vocab_size=bpe_vocab)
    tok     = BPEWrapper(raw_tok)
    vocab_size = tok.vocab_size_actual()
    print(f'  BPE vocab: {vocab_size} tokens')

    print(f'\n  Encoding data (max_seq_len={max_seq_len})...')
    train_data = encode_lines(splits['train'], tok, max_seq_len)
    val_data   = encode_lines(splits['valid'], tok, max_seq_len)
    test_data  = encode_lines(splits['test'],  tok, max_seq_len)
    print(f'  Train {len(train_data):,} | Val {len(val_data):,} | Test {len(test_data):,}')

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f'  Device: {device} | AMP: {use_amp}')

    # Shared model config (matches 13.5M run for direct comparison)
    cfg = dict(
        vocab_size=vocab_size, embedding_dim=256, num_layers=6,
        num_heads=8, ffn_dim=1024, field_size=4096,
        max_seq_len=max_seq_len + 2, dropout=0.1,
        use_checkpoint=True,
    )
    train_cfg = dict(
        num_epochs=10, batch_size=8, grad_accum=4, peak_lr=3e-4,
        use_amp=use_amp,
    )

    results = {}

    # ── Condition A: skipped — results loaded from prior completed run ────────
    print(f'\n{"="*70}')
    print('  CONDITION A — skipped (using saved results from prior run)')
    print(f'{"="*70}')
    res_a = {
        'model_name': 'Wave V4 (no dispersion)',
        'params': 13326352,
        'best_ppl': 87.20659856085048,
        'best_acc': 26.250515601918195,
        'best_epoch': 10,
        'total_time': 5242.925741672516,
        'test_ppl': 86.79913927291058,
        'test_acc': 26.286280589779736,
    }
    print(f'  Condition A TEST: PPL {res_a["test_ppl"]:.1f} | Acc {res_a["test_acc"]:.1f}%  (prior run)')
    results['A_v4_no_dispersion'] = res_a

    # ── Condition B: V4 improvements + dispersion ─────────────────────────────
    print(f'\n{"="*70}')
    print('  CONDITION B — V4 improvements + learned dispersion (V4D)')
    print(f'{"="*70}')

    model_b = WaveFieldTransformerV4D(**cfg, use_dispersion=True).to(device)
    res_b   = train_model(model_b, train_data, val_data, tok, vocab_size, device,
                          'Wave V4D (with dispersion)', **train_cfg,
                          save_dir='2048_v4d_dispersion_checkpoints')

    tb = create_batches(test_data, train_cfg['batch_size'], device, shuffle=False)
    _, tp_b, ta_b = evaluate(model_b, tb, vocab_size, device, use_amp)
    res_b['test_ppl'] = tp_b; res_b['test_acc'] = ta_b
    print(f'\n  Condition B TEST: PPL {tp_b:.1f} | Acc {ta_b:.1f}%')
    print_dispersion_analysis(model_b, 'Condition B')

    del model_b; gc.collect(); torch.cuda.empty_cache()
    results['B_v4d_with_dispersion'] = res_b

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print('  ABLATION RESULTS')
    print(f'{"="*70}')

    ref_std = 64.5  # Standard 13.5M @ 2048 (from earlier run)
    tp_a    = res_a['test_ppl']   # hardcoded from prior run

    print(f'\n  {"Model":<35} {"Test PPL":>10} {"vs Standard":>12} {"vs V3.5 base*":>14}')
    print(f'  {"─"*35} {"─"*10} {"─"*12} {"─"*14}')
    print(f'  {"[Standard 13.5M @ 2048]":<35} {ref_std:>10.1f} {"—":>12} {"—":>14}')
    print(f'  {"[Wave V3.5 13.5M @ 2048]*":<35} {"~100-115":>10} {"~+55-78%":>12} {"—":>14}')
    gap_a = (tp_a - ref_std) / ref_std * 100
    gap_b = (tp_b - ref_std) / ref_std * 100
    disp_gain = (tp_a - tp_b) / tp_a * 100 if tp_a > 0 else 0
    print(f'  {"Wave V4 (no dispersion)":<35} {tp_a:>10.1f} {gap_a:>+11.1f}% {"baseline":>14}')
    print(f'  {"Wave V4D (with dispersion)":<35} {tp_b:>10.1f} {gap_b:>+11.1f}% {disp_gain:>+13.1f}%')

    print(f'\n  * V3.5 baseline currently training; estimates from epoch trajectory')

    if tp_b < tp_a:
        print(f'\n  ✓ DISPERSION HELPS: {disp_gain:.1f}% PPL improvement over V4 base')
        if tp_b < ref_std * 1.3:
            print(f'  ✓ STRONG: within 30% of Standard transformer')
        elif tp_b < ref_std * 1.5:
            print(f'  ✓ COMPETITIVE: within 50% of Standard transformer')
    else:
        pct = (tp_b - tp_a) / tp_a * 100
        print(f'\n  ✗ Dispersion did not help at this scale (+{pct:.1f}%)')
        print(f'    Check: did β values differentiate? (see analysis above)')

    with open('2048_dispersion_ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results → 2048_dispersion_ablation_results.json')


if __name__ == '__main__':
    main()
