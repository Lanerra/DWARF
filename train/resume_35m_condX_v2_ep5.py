"""
Resume condX-v2 35M from epoch 5 checkpoint — runs epochs 6-10 then final eval.

Note: optimizer moments not saved, so AdamW restarts with cold moments.
LR is correctly positioned via scheduler fast-forward (≈1.5e-4 at ep5/10 of cosine).
Appends to existing logs/condX_35m_v2_run.log.
"""
import sys, os, math, time, torch, torch.nn.functional as F
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kernels')))

# ── Load original module (classes, constants, helpers) ───────────────────────
import importlib.util
_orig = os.path.join(os.path.dirname(__file__), 'train_2048_35m_condX_v2.py')
spec  = importlib.util.spec_from_file_location('condx_v2_35m', _orig)
T     = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

RESUME_EP    = 5
START_EP     = 6
NUM_EPOCHS   = T.NUM_EPOCHS        # 10
BATCH_SIZE   = T.BATCH_SIZE
GRAD_ACCUM   = T.GRAD_ACCUM
LR           = T.LR
SAVE_DIR     = T.SAVE_DIR
LOG_FILE     = os.path.join(os.path.dirname(__file__), '..', 'logs', 'condX_35m_v2_run.log')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}')

# ── Paths + tokenizer ─────────────────────────────────────────────────────────
_root     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_tok_path = os.path.join(_root, 'results', '2048_condI_tokenizer.json')

# ── Data (pre-encoded tensors — subset to match original condX-v2 run) ────────
# Original condX-v2 35M used ~52,716 train seqs (same as condU 35M, confirmed by
# 14%C at ep1: 0.14 × 700M Chinchilla / 2047 ≈ 48K seqs from 52,716-seq split).
# The full cache has 212,458 train seqs; we must use the same split to keep the
# comparison with condU 35M fair.
_encoded_cache = os.path.join(_root, 'logs', 'fineweb_encoded_2048.pt')
print(f'Loading pre-encoded dataset from {_encoded_cache} ...')
_cache          = torch.load(_encoded_cache, weights_only=True)
_train_full     = _cache['train']   # 212,458 seqs
# Match the 52,716 / 5,582 / 5,473 split used in all condU/d52/d53/d54 runs
N_TRAIN, N_VAL, N_TEST = 52716, 5582, 5473
train_data = _train_full[:N_TRAIN]
val_data   = _cache['val'][:N_VAL]
test_data  = _cache['test'][:N_TEST]
print(f'  train: {len(train_data):,}  val: {len(val_data):,}  test: {len(test_data):,} seqs  '
      f'(subset of {len(_train_full):,} — matching original condX-v2 run)')

steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE / GRAD_ACCUM)
total_steps     = NUM_EPOCHS * steps_per_epoch
steps_done      = RESUME_EP * steps_per_epoch
from tokenizers import Tokenizer
tokenizer = T.BPETokenizerWrapper(Tokenizer.from_file(_tok_path))

# ── Model + load checkpoint ───────────────────────────────────────────────────
model = T.CondXTransformer(
    T.VOCAB_SIZE, T.EMBEDDING_DIM, T.NUM_LAYERS, T.NUM_HEADS,
    T.FFN_DIM, seq_len=T.MAX_SEQ_LEN,
    interference_interval=T.INTERFERENCE,
).to(device)

ckpt_path = os.path.join(SAVE_DIR, f'epoch_{RESUME_EP:02d}.pt')
ckpt      = torch.load(ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
print(f'Loaded ep{RESUME_EP} weights  val_ppl={ckpt["val_ppl"]:.3f}')
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Parameters: {n_params:,}')

# ── Optimizer + scheduler fast-forwarded to step `steps_done` ────────────────
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps)
for _ in range(steps_done):
    scheduler.step()
print(f'LR after fast-forward ({steps_done} steps): {scheduler.get_last_lr()[0]:.2e}')

scaler         = torch.amp.GradScaler('cuda')
best_val_loss  = math.log(ckpt['val_ppl'])
best_val_ppl   = ckpt['val_ppl']
best_epoch     = RESUME_EP
t0_wall        = time.time()
T0_OFFSET      = 9374   # seconds elapsed at end of ep5 (for cumulative timing)

GEN_PROMPTS = [
    'It was a dark and stormy',
    'The length of the hypotenuse',
    'The President of the United',
    'Once upon a time there was',
    'The results indicate that',
]

per_epoch_results = []

# ── Log tee: append to existing run log ──────────────────────────────────────
import builtins
_orig_print = builtins.print
_logf = open(LOG_FILE, 'a')

def _tee(*args, **kwargs):
    _orig_print(*args, **kwargs)
    kwargs.pop('file', None)
    _orig_print(*args, file=_logf, **kwargs)
    _logf.flush()

builtins.print = _tee

print(f'\n{"="*60}')
print(f'RESUMED ep{RESUME_EP} → running epochs {START_EP}–{NUM_EPOCHS}')
print(f'LR at resume: {scheduler.get_last_lr()[0]:.2e}  (cosine from 3e-4)')
print(f'{"="*60}\n')

# ── Training loop ─────────────────────────────────────────────────────────────
for epoch in range(START_EP, NUM_EPOCHS + 1):
    model.train()
    indices      = torch.randperm(len(train_data))
    step         = 0
    optimizer.zero_grad()
    running_loss = 0.0

    for acc_step in range(steps_per_epoch):
        for ga in range(GRAD_ACCUM):
            idx_start = (acc_step * GRAD_ACCUM + ga) * BATCH_SIZE
            if idx_start >= len(train_data): continue
            batch = train_data[indices[idx_start: idx_start + BATCH_SIZE]]
            x, y  = batch[:, :-1].to(device), batch[:, 1:].to(device)
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss   = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)) / GRAD_ACCUM
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        scheduler.step(); step += 1
        running_loss += loss.item() * GRAD_ACCUM
        if step % 200 == 0:
            print(f'  Step {step}/{steps_per_epoch} | Loss {loss.item() * GRAD_ACCUM:.4f}')

    train_loss = running_loss / max(step, 1)
    val_loss   = T.evaluate(model, val_data, BATCH_SIZE, device)
    val_ppl    = math.exp(min(val_loss, 20))
    elapsed    = (time.time() - t0_wall) + T0_OFFSET
    chin_pct   = epoch * len(train_data) * (T.MAX_SEQ_LEN - 1) / (20 * n_params) * 100

    marker = ''
    if val_loss < best_val_loss:
        best_val_loss, best_val_ppl, best_epoch = val_loss, val_ppl, epoch
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best.pt'))
        marker = ' * BEST'

    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_ppl': val_ppl, 'chinchilla_pct': chin_pct},
               os.path.join(SAVE_DIR, f'epoch_{epoch:02d}.pt'))

    print(f'Ep {epoch}/{NUM_EPOCHS} | Train {train_loss:.4f} '
          f'| Val {val_loss:.4f} PPL {val_ppl:.1f}{marker} '
          f'| {elapsed:.0f}s ({chin_pct:.0f}%C)')

    # ── Per-epoch diagnostics (mirrors original script) ───────────────────────
    ss = model.attn_summary()
    head_means  = ss['pos_bias_mean_per_head']
    most_local  = int(max(range(T.NUM_HEADS), key=lambda h: abs(head_means[h])))
    most_global = int(min(range(T.NUM_HEADS), key=lambda h: abs(head_means[h])))
    print(f'  DSQG pos-bias: |mean|={ss["pos_bias_abs_mean"]:.4f} '
          f'|max|={ss["pos_bias_abs_max"]:.4f} '
          f'most-local=h{most_local} most-global=h{most_global}')
    print(f'  scale_embed:   |mean|={ss["scale_embed_abs_mean"]:.4f} '
          f'|max|={ss["scale_embed_abs_max"]:.4f}')
    gains    = ss['if_gain']
    gain_str = '  '.join(f'h{h}:{gains[h]:.2f}' for h in range(T.NUM_HEADS))
    print(f'  IF gains:      {gain_str}')
    if 'bypass_alpha' in ss:
        print(f'  Bypass gate:   alpha={ss["bypass_alpha"]:.4f}  '
              f'gate={ss["bypass_gate"]:.6f}  '
              f'(gate→0=condV-like, gate→1=fully-clean-Q)')
        print(f'  Bypass diag:   clean_norm={ss["clean_norm"]:.4f}  '
              f'full_norm={ss["full_norm"]:.4f}  '
              f'q_cosim={ss["q_cosim"]:.4f}  '
              f'q_delta_norm={ss["q_delta_norm"]:.4f}')
    if ss['ema_factors']:
        print(f'  EMA factors:   ' +
              '  '.join(f'b{i}:{v:.4f}' for i, v in enumerate(ss['ema_factors'])))
        print(f'  KdV alphas:    ' +
              '  '.join(f'b{i}:{v:.4f}' for i, v in enumerate(ss['kdv_alphas'])))

    print('  -- Generation (T=0.7) --')
    for prompt, gen in zip(GEN_PROMPTS,
                           T.generate(model, tokenizer, GEN_PROMPTS, device, temperature=0.7)):
        print(f'    {repr(prompt)} -> {repr(gen[:80])}')
    print('  --')

    print('  Passkey...')
    pk      = T.passkey_accuracy(model, tokenizer, device)
    pk_mean = sum(pk.values()) / len(pk)
    above50 = sum(1 for v in pk.values() if v >= 0.5)
    print(f'  mean={pk_mean*100:.1f}%  ({above50}/{len(pk)} distances >50%)')
    print('  ' + '  '.join(f'd={d}:{int(pk[d]*100)}%' for d in T.PASSKEY_DISTANCES))

    per_epoch_results.append({
        'epoch': epoch, 'val_ppl': val_ppl, 'train_loss': train_loss,
        'chinchilla_pct': chin_pct, 'passkey_mean': pk_mean,
        'passkey_by_d': {str(d): v for d, v in pk.items()},
        'bypass_alpha': ss.get('bypass_alpha'), 'bypass_gate': ss.get('bypass_gate'),
    })
    sys.stdout.flush()

# ── Final test eval ───────────────────────────────────────────────────────────
print(f'\n  Best checkpoint: ep{best_epoch}  val_ppl={best_val_ppl:.3f}')
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best.pt'),
                                  map_location=device, weights_only=True))
test_loss = T.evaluate(model, test_data, BATCH_SIZE, device)
test_ppl  = math.exp(min(test_loss, 20))
print(f'\n  condX-v2 35M TEST PPL: {test_ppl:.3f}')
print(f'  condU 35M hybrid:      38.293 PPL / 90.0% passkey  |  delta PPL = {test_ppl - 38.293:+.3f}')
print(f'  condV (all physics):   52.207 PPL                   |  delta PPL = {test_ppl - 52.207:+.3f}')
print(f'  condU 35M pure:        48.271 PPL / 33.3% passkey  |  delta PPL = {test_ppl - 48.271:+.3f}')

import json
result = {
    'experiment': 'condX_v2_35m', 'resumed_from_epoch': RESUME_EP,
    'test_ppl': test_ppl, 'best_epoch': best_epoch, 'best_val_ppl': best_val_ppl,
    'per_epoch': per_epoch_results,
}
out_path = os.path.join(_root, 'benchmarks', 'logs', 'condX_v2_35m_results.json')
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f'  Results saved: {out_path}')

builtins.print = _orig_print
_logf.close()
