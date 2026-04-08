"""
MDMT D=512 Smoke Test — Multiresolution Dyadic Memory Tree

300-step smoke test to validate:
  - No NaN
  - Loss descends below 6.0
  - Step time < 200ms
  - Passkey accuracy at d=16,64,256,512

Architecture:
  D=512, H=8, HD=64, L=8, FFN=1024, vocab=32000
  FA at layer 2 (0-indexed), all others DyadicTreeBlock (local W=16 + tree summaries)
  No EMA, no scale_embed, no DSQG offsets — clean slate.

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -u train/train_2048_mdmt_d512_smoke.py \\
    2>&1 | tee logs/mdmt_d512_smoke_run.log
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Hyperparameters ──────────────────────────────────────────────────────────

VOCAB_SIZE      = 32000
BATCH_SIZE      = 8
GRAD_ACCUM      = 8
LR              = 3e-4
MAX_SEQ_LEN     = 2048
MAX_STEPS       = 300
LOG_EVERY       = 50
PASSKEY_EVERY   = 100
WARMUP_STEPS    = 30

EMBEDDING_DIM   = 512
NUM_LAYERS      = 8
NUM_HEADS       = 8
FFN_DIM         = 1024
WINDOW_SIZE     = 16
FA_LAYER        = 2

# ── Dataset ──────────────────────────────────────────────────────────────────

ENCODED_CACHE   = 'logs/fineweb_edu_encoded_2048_v2.pt'
MAX_TRAIN_SEQS  = 50_000

# ── Save paths ───────────────────────────────────────────────────────────────

SAVE_DIR        = 'autoresearch/checkpoints'
BEST_CKPT       = os.path.join(SAVE_DIR, 'mdmt_d512_smoke_best.pt')
RESULT_FILE     = 'logs/mdmt_d512_smoke_results.json'

# ── Passkey ──────────────────────────────────────────────────────────────────

PASSKEY_DISTANCES = [16, 64, 256, 512]
PASSKEY_TRIALS    = 5
_PASSKEY_WORDS    = ['apple', 'banana', 'orange', 'cherry', 'grape',
                     'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE  = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE   = 'the secret word is {word} .'
_RETRIEVAL_CUE    = 'the secret word is'

# ── Model import ─────────────────────────────────────────────────────────────

import pathlib as _pl
_model_dir = str(_pl.Path(__file__).parent.parent / 'models')
if _model_dir not in sys.path:
    sys.path.insert(0, _model_dir)

from mdmt import MDMTTransformer

# ── Tokenizer ────────────────────────────────────────────────────────────────

class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)
    def vocab_size(self):   return self.tokenizer.get_vocab_size()


def load_tokenizer():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_directory, '..', 'results', '2048_condI_tokenizer.json'),
        os.path.join(script_directory, 'results', '2048_condI_tokenizer.json'),
        os.path.join(script_directory, '2048_condI_tokenizer.json'),
    ]
    tokenizer_path = next((p for p in candidates if os.path.exists(p)), None)
    if tokenizer_path is None:
        raise FileNotFoundError('condI BPE tokenizer not found in results/')
    from tokenizers import Tokenizer
    tok = BPETokenizerWrapper(Tokenizer.from_file(tokenizer_path))
    print(f'Loaded tokenizer from {tokenizer_path}  (vocab={tok.vocab_size()})')
    return tok


def load_dataset():
    assert os.path.exists(ENCODED_CACHE), f'Dataset not found: {ENCODED_CACHE}'
    print(f'Loading pre-encoded dataset from {ENCODED_CACHE} …')
    cache = torch.load(ENCODED_CACHE, weights_only=True)
    train_data = cache['train']
    val_data = cache['val']
    if len(train_data) > MAX_TRAIN_SEQS:
        train_data = train_data[:MAX_TRAIN_SEQS]
    print(f'  train: {len(train_data):,}  val: {len(val_data):,} seqs')
    return train_data, val_data


# ── Passkey eval ─────────────────────────────────────────────────────────────

def run_passkey_eval(model, tokenizer, device):
    model.eval()
    results = []
    with torch.no_grad():
        for distance in PASSKEY_DISTANCES:
            correct = 0
            number_valid = 0
            for word in _PASSKEY_WORDS[:PASSKEY_TRIALS]:
                intro = _INTRO_TEMPLATE.format(word=word)
                fillers = (_FILLER_SENTENCE * (distance * 10))[:distance * 80]
                prompt = intro + ' ' + fillers + ' ' + _RETRIEVAL_CUE
                ids = tokenizer.encode(prompt)
                if len(ids) >= MAX_SEQ_LEN:
                    continue
                inp = torch.tensor([ids], device=device)
                logits = model(inp)
                predicted_id = logits[0, -1].argmax().item()
                predicted = tokenizer.decode([predicted_id]).strip()
                if predicted == word:
                    correct += 1
                number_valid += 1
            accuracy = correct / number_valid if number_valid > 0 else 0.0
            results.append({'distance': distance, 'accuracy': accuracy, 'number_valid': number_valid})
    return results


# ── MFU estimate ─────────────────────────────────────────────────────────────

def estimate_mfu(number_of_parameters, batch_size, sequence_length, step_time_seconds):
    flops_per_token = 6 * number_of_parameters
    tokens_per_step = batch_size * GRAD_ACCUM * sequence_length
    flops_per_step = flops_per_token * tokens_per_step
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''
    if 'H100' in gpu_name:
        peak_flops = 989e12
    elif '4090' in gpu_name:
        peak_flops = 330e12
    elif 'A100' in gpu_name:
        peak_flops = 312e12
    else:
        peak_flops = 330e12
    return flops_per_step / (step_time_seconds * peak_flops)


# ── Training ─────────────────────────────────────────────────────────────────

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    tokenizer = load_tokenizer()
    train_data, val_data = load_dataset()

    model = MDMTTransformer(
        vocab_size=VOCAB_SIZE,
        embedding_dimension=EMBEDDING_DIM,
        number_of_layers=NUM_LAYERS,
        number_of_heads=NUM_HEADS,
        feedforward_dimension=FFN_DIM,
        sequence_length=MAX_SEQ_LEN,
        window_size=WINDOW_SIZE,
        full_attention_layer=FA_LAYER,
    ).to(device)

    number_of_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {number_of_parameters:,}')
    print(f'Architecture: D={EMBEDDING_DIM} H={NUM_HEADS} L={NUM_LAYERS} '
          f'FFN={FFN_DIM} W={WINDOW_SIZE} FA@L{FA_LAYER}')
    print(f'Effective batch: {BATCH_SIZE * GRAD_ACCUM}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

    def learning_rate_schedule(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_schedule)

    os.makedirs(SAVE_DIR, exist_ok=True)

    all_results = []
    best_val_loss = float('inf')
    global_step = 0
    accumulated_loss = 0.0
    accumulated_steps = 0

    print(f'\n── Starting {MAX_STEPS}-step smoke test ──')

    model.train()
    optimizer.zero_grad()
    step_start_time = time.time()

    number_of_sequences = len(train_data)
    sequence_index = 0
    permutation = torch.randperm(number_of_sequences)

    while global_step < MAX_STEPS:
        if sequence_index + BATCH_SIZE > number_of_sequences:
            permutation = torch.randperm(number_of_sequences)
            sequence_index = 0

        indices = permutation[sequence_index:sequence_index + BATCH_SIZE]
        sequence_index += BATCH_SIZE

        batch = train_data[indices].to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:].long()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            loss = loss / GRAD_ACCUM

        loss.backward()
        accumulated_loss += loss.item() * GRAD_ACCUM
        accumulated_steps += 1

        if accumulated_steps % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % LOG_EVERY == 0 or global_step == 1:
                step_time = (time.time() - step_start_time) / min(global_step, LOG_EVERY)
                average_loss = accumulated_loss / (LOG_EVERY * GRAD_ACCUM) if global_step > 1 else accumulated_loss / GRAD_ACCUM
                mfu = estimate_mfu(number_of_parameters, BATCH_SIZE, MAX_SEQ_LEN, step_time)

                if math.isnan(average_loss):
                    print(f'[step {global_step}] ⚠ NaN loss — aborting')
                    return

                print(f'[step {global_step:>4d}] loss={average_loss:.4f} '
                      f'lr={scheduler.get_last_lr()[0]:.2e} '
                      f'step_time={step_time*1000:.0f}ms '
                      f'mfu={mfu:.2%}')

                result = {
                    'step': global_step,
                    'loss': average_loss,
                    'step_time_ms': step_time * 1000,
                    'mfu': mfu,
                    'lr': scheduler.get_last_lr()[0],
                }

                if global_step % PASSKEY_EVERY == 0:
                    passkey = run_passkey_eval(model, tokenizer, device)
                    passkey_string = '  '.join(
                        f'd{r["distance"]}:{r["accuracy"]:.0%}' for r in passkey)
                    print(f'         passkey: {passkey_string}')
                    result['passkey'] = passkey
                    model.train()

                all_results.append(result)
                accumulated_loss = 0.0
                step_start_time = time.time()

    # ── Final validation ─────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    val_tokens = 0
    number_val_batches = min(64, len(val_data) // BATCH_SIZE)

    with torch.no_grad():
        for i in range(number_val_batches):
            batch = val_data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:].long()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(inputs)
            loss = F.cross_entropy(
                logits.float().reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            val_loss += loss.item() * targets.numel()
            val_tokens += targets.numel()

    val_perplexity = math.exp(val_loss / val_tokens)
    print(f'\n── Final: val_ppl={val_perplexity:.3f} ──')

    passkey = run_passkey_eval(model, tokenizer, device)
    passkey_string = '  '.join(f'd{r["distance"]}:{r["accuracy"]:.0%}' for r in passkey)
    print(f'   passkey: {passkey_string}')

    torch.save({
        'model_state': model.state_dict(),
        'val_ppl': val_perplexity,
        'steps': MAX_STEPS,
        'passkey': passkey,
    }, BEST_CKPT)
    print(f'Saved checkpoint to {BEST_CKPT}')

    all_results.append({
        'final': True,
        'val_ppl': val_perplexity,
        'passkey': passkey,
    })
    with open(RESULT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'Results saved to {RESULT_FILE}')


if __name__ == '__main__':
    train()
