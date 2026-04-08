"""
🧪 DART D=512 L=8 — 300-STEP SMOKE TEST

Purpose: Verify DART architecture trains, routes, and retrieves.
MDMT baseline: loss 10.47→6.35, passkey 0% at all distances.
DART hypothesis: explicit routing + exact leaf retrieval should show
nonzero passkey accuracy even in 300 steps.

Architecture: D=512, H=8, HD=64, L=8, FFN=1024, vocab=32000
  Local attention W=32, tree chunk_size=32 (64 leaves, 6 routing levels)
  Semantic EMA bank M=6 states (rank=64)
  FA anchor at layer 2

Dataset: fineweb_edu_encoded_2048_v2.pt (first 50K seqs)
BS=8, GA=8 (eff=64), LR=3e-4, BF16

Key diagnostics:
  - Gradient norm through AddressEncoder.W_c (starvation canary)
  - Routing entropy per tree level
  - Passkey accuracy at d=16/64/256/512

Run:
  cd /home/dlewis3/Desktop/AI/DWARF
  .venv/bin/python3 -u train/train_dart_d512_smoke.py 2>&1 | tee logs/run_dart_d512_smoke.log
"""

EMBEDDING_DIMENSION = 512
NUMBER_OF_HEADS = 8
FEEDFORWARD_DIMENSION = 1024
NUMBER_OF_LAYERS = 8
FULL_ATTENTION_LAYER = 2
SEQUENCE_LENGTH = 2048
CHUNK_SIZE = 32
WINDOW_SIZE = 16
VOCAB_SIZE = 32000

LEARNING_RATE = 3e-4
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 2
CROSS_ENTROPY_CHUNK = 512
MAX_STEPS = 300
LOG_INTERVAL = 50
PASSKEY_INTERVAL = 100

MAX_TRAIN_SEQUENCES = 50_000
DATASET_PATH = 'logs/fineweb_edu_encoded_2048_v2.pt'
CHECKPOINT_DIRECTORY = 'autoresearch/checkpoints'

PASSKEY_DISTANCES = [16, 64, 256, 512]
PASSKEY_TRIALS = 50
PASSKEY_BATCH_SIZE = 16
_PASSKEY_WORDS = ['apple', 'banana', 'orange', 'cherry', 'grape',
                  'lemon', 'mango', 'peach', 'plum', 'berry']
_FILLER_SENTENCE = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE = 'the secret word is {word} .'
_RETRIEVAL_CUE = 'the secret word is'

TOKENIZER_CANDIDATES = [
    'results/fineweb_tokenizer_32k.json',
    'results/fineweb_v32k_v2_tokenizer.json',
]

import contextlib, math, os, subprocess, sys, time
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

torch.set_float32_matmul_precision('medium')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import pathlib as _pathlib
_project_root = str(_pathlib.Path(__file__).resolve().parent.parent)
_model_dir = os.path.join(_project_root, 'models')
for _directory in [_model_dir, _project_root]:
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

from dart import DARTTransformer, DARTBlock

try:
    import bitsandbytes as bnb
    _BITSANDBYTES_AVAILABLE = True
except ImportError:
    _BITSANDBYTES_AVAILABLE = False
    print("WARNING: bitsandbytes not available, using standard AdamW")

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    _LIGER_AVAILABLE = True
except ImportError:
    _LIGER_AVAILABLE = False

USE_LIGER_CE = _LIGER_AVAILABLE and os.getenv("DWARF_LIGER", "1") != "0"


def get_gpu_peak_flops(device="cuda"):
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(device)
    if "H200" in name: return 1979e12
    if "H100" in name: return 989e12
    if "4090" in name: return 330e12
    if "3090" in name: return 142e12
    if "A100" in name: return 312e12
    return None


def _amp_context(device):
    if device == 'cuda':
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


class BPETokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    def decode(self, ids):
        return self.tokenizer.decode(ids)
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()


def compute_tau(step, max_steps):
    return 1.0 - 0.5 * step / max_steps


def compute_alpha(step, max_steps):
    return 0.3 * step / max_steps


def compute_lambda_entropy(step, max_steps):
    return 0.01 * (1.0 - 0.9 * step / max_steps)


LAMBDA_BALANCE = 0.01


def address_encoder_gradient_norm(model):
    total = 0.0
    for param in model.address_encoder_parameters():
        if param.grad is not None:
            total += param.grad.norm().item() ** 2
    return total ** 0.5


@torch.inference_mode()
def passkey_accuracy(model, tokenizer, device):
    model.eval()
    filler_ids = tokenizer.encode(_FILLER_SENTENCE)
    cue_ids = tokenizer.encode(_RETRIEVAL_CUE)
    word_token_ids = {}
    for word in _PASSKEY_WORDS:
        encoded = tokenizer.encode(' ' + word) or tokenizer.encode(word)
        if not encoded:
            raise ValueError(f'Could not encode: {word}')
        word_token_ids[word] = encoded[0]

    results = {}
    for distance in PASSKEY_DISTANCES:
        sequences, last_positions, candidate_rows = [], [], []
        for trial in range(PASSKEY_TRIALS):
            target = _PASSKEY_WORDS[trial % len(_PASSKEY_WORDS)]
            others = [w for w in _PASSKEY_WORDS if w != target]
            intro_ids = tokenizer.encode(_INTRO_TEMPLATE.format(word=target))
            available = SEQUENCE_LENGTH - 1 - len(intro_ids) - len(cue_ids) - 1
            if distance > available:
                continue
            filler = []
            while len(filler) < distance:
                filler.extend(filler_ids)
            full_sequence = intro_ids + filler[:distance] + cue_ids
            if len(full_sequence) >= SEQUENCE_LENGTH:
                continue
            sequences.append(
                full_sequence + [0] * (SEQUENCE_LENGTH - len(full_sequence)))
            last_positions.append(len(full_sequence) - 1)
            candidate_rows.append(
                [word_token_ids[target]] +
                [word_token_ids[w] for w in others[:9]])

        if not sequences:
            results[distance] = 0.0
            continue

        ids = torch.tensor(sequences, dtype=torch.long, device=device)
        positions = torch.tensor(last_positions, dtype=torch.long, device=device)
        candidates = torch.tensor(candidate_rows, dtype=torch.long, device=device)
        correct = 0

        for start in range(0, ids.size(0), PASSKEY_BATCH_SIZE):
            batch_ids = ids[start:start + PASSKEY_BATCH_SIZE]
            batch_positions = positions[start:start + PASSKEY_BATCH_SIZE]
            batch_candidates = candidates[start:start + PASSKEY_BATCH_SIZE]
            with _amp_context(device):
                logits, _ = model(batch_ids)
            row_indices = torch.arange(batch_ids.size(0), device=device)
            next_logits = logits[row_indices, batch_positions, :]
            candidate_logits = torch.gather(next_logits, 1, batch_candidates)
            correct += (candidate_logits.argmax(dim=1) == 0).sum().item()

        results[distance] = correct / ids.size(0)
    return results


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    training_start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    git_hash = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    print('=' * 70)
    print('  🧪 DART D512-L8 FA@L2 — 300-STEP SMOKE TEST')
    print('=' * 70)
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Liger CE: {"enabled" if USE_LIGER_CE else "disabled"}')
    print(f'  AdamW8bit: {"enabled" if _BITSANDBYTES_AVAILABLE else "disabled"}')
    print(f'  D={EMBEDDING_DIMENSION}, H={NUMBER_OF_HEADS}, '
          f'hd={EMBEDDING_DIMENSION // NUMBER_OF_HEADS}, '
          f'L={NUMBER_OF_LAYERS}, FFN={FEEDFORWARD_DIMENSION}')
    print(f'  FA@L{FULL_ATTENTION_LAYER}, chunk={CHUNK_SIZE}, window={WINDOW_SIZE}')
    print(f'  Tree: {SEQUENCE_LENGTH // CHUNK_SIZE} leaves, '
          f'{int(math.log2(SEQUENCE_LENGTH // CHUNK_SIZE))} routing levels')
    print(f'  EMA bank: M=6, rank=64')
    print(f'  Batch: BS={BATCH_SIZE}×GA={GRADIENT_ACCUMULATION}'
          f'=eff_batch={BATCH_SIZE * GRADIENT_ACCUMULATION}')
    print(f'  LR={LEARNING_RATE}, max_steps={MAX_STEPS}')
    print(f'  git={git_hash}')

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer_path = next(
        (p for p in TOKENIZER_CANDIDATES if os.path.exists(p)), None)
    if tokenizer_path is None:
        raise FileNotFoundError(f'Tokenizer not found: {TOKENIZER_CANDIDATES}')
    from tokenizers import Tokenizer
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(tokenizer_path))
    print(f'Tokenizer: {tokenizer_path} (vocab={tokenizer.vocab_size():,})')

    # ── Dataset ───────────────────────────────────────────────────────────────
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f'Dataset not found: {DATASET_PATH}')
    print(f'Loading dataset from {DATASET_PATH}')
    cache = torch.load(DATASET_PATH, weights_only=True)
    if isinstance(cache, dict):
        train_data = cache['train'].long()
    else:
        train_data = cache.long()

    if len(train_data) > MAX_TRAIN_SEQUENCES:
        train_data = train_data[:MAX_TRAIN_SEQUENCES]
    print(f'  train: {len(train_data):,} seqs')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DARTTransformer(
        vocab_size=VOCAB_SIZE,
        embedding_dimension=EMBEDDING_DIMENSION,
        number_of_layers=NUMBER_OF_LAYERS,
        number_of_heads=NUMBER_OF_HEADS,
        feedforward_dimension=FEEDFORWARD_DIMENSION,
        sequence_length=SEQUENCE_LENGTH,
        chunk_size=CHUNK_SIZE,
        window_size=WINDOW_SIZE,
        full_attention_layer=FULL_ATTENTION_LAYER,
    ).to(device)

    number_of_parameters = model.parameter_count()
    print(f'Parameters: {number_of_parameters:,} ({number_of_parameters / 1e6:.1f}M)')

    # ── Optimizer ─────────────────────────────────────────────────────────────
    AdamWClass = bnb.optim.AdamW8bit if _BITSANDBYTES_AVAILABLE else torch.optim.AdamW
    optimizer = AdamWClass(
        model.parameters(), lr=LEARNING_RATE,
        weight_decay=0.1, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_STEPS)

    if USE_LIGER_CE:
        liger_ce_function = LigerFusedLinearCrossEntropyLoss()

    os.makedirs(CHECKPOINT_DIRECTORY, exist_ok=True)
    best_loss = float('inf')

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f'Warming up (BS={BATCH_SIZE})...')
    warmup_batch_size = min(BATCH_SIZE, len(train_data))
    warmup_x = train_data[:warmup_batch_size, :-1].to(device)
    warmup_y = train_data[:warmup_batch_size, 1:].to(device)
    with _amp_context(device):
        warmup_logits, _ = model(warmup_x)
    warmup_loss = F.cross_entropy(
        warmup_logits.reshape(-1, warmup_logits.size(-1)),
        warmup_y.reshape(-1))
    warmup_loss.backward()
    optimizer.zero_grad(set_to_none=True)
    del warmup_x, warmup_y, warmup_logits, warmup_loss
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print('  Warmup complete.')

    # ── Training loop ─────────────────────────────────────────────────────────
    gpu_peak_flops = get_gpu_peak_flops(device)
    tokens_per_step = BATCH_SIZE * GRADIENT_ACCUMULATION * (SEQUENCE_LENGTH - 1)
    flops_per_step = 6 * number_of_parameters * tokens_per_step
    timing_window = deque(maxlen=20)
    indices = torch.randperm(len(train_data))
    global_step = 0

    print(f'\n  Starting training ({MAX_STEPS} steps)...\n')

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for accumulation_step in range(MAX_STEPS):
        timer_start = torch.cuda.Event(enable_timing=True)
        timer_end = torch.cuda.Event(enable_timing=True)
        timer_start.record()

        tau = compute_tau(accumulation_step, MAX_STEPS)
        alpha = compute_alpha(accumulation_step, MAX_STEPS)
        lambda_entropy = compute_lambda_entropy(accumulation_step, MAX_STEPS)
        accumulated_ce_loss = 0.0

        for ga_index in range(GRADIENT_ACCUMULATION):
            data_index = (
                (accumulation_step * GRADIENT_ACCUMULATION + ga_index) * BATCH_SIZE
            ) % len(train_data)
            batch_indices = indices[data_index:data_index + BATCH_SIZE]
            if len(batch_indices) < BATCH_SIZE:
                batch_indices = indices[:BATCH_SIZE]
            batch = train_data[batch_indices]
            x = batch[:, :-1].to(device, non_blocking=True)
            y = batch[:, 1:].to(device, non_blocking=True)

            with _amp_context(device):
                logits, auxiliary = model(x, tau=tau, alpha=alpha)

            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1))
            auxiliary_loss = (
                -lambda_entropy * auxiliary['entropy_loss']
                + LAMBDA_BALANCE * auxiliary['balance_loss'])
            total_loss = ce_loss + auxiliary_loss

            (total_loss / GRADIENT_ACCUMULATION).backward()
            accumulated_ce_loss += ce_loss.item()
            last_auxiliary = {
                'entropy_loss': auxiliary['entropy_loss'].item(),
                'balance_loss': auxiliary['balance_loss'].item(),
                'per_level_entropy': auxiliary.get('per_level_entropy', []),
            }

            del logits, auxiliary

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        address_grad = address_encoder_gradient_norm(model)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        timer_end.record()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            step_milliseconds = timer_start.elapsed_time(timer_end)
        else:
            step_milliseconds = 1.0
        timing_window.append(step_milliseconds)
        global_step += 1

        average_ce = accumulated_ce_loss / GRADIENT_ACCUMULATION

        if average_ce < best_loss:
            best_loss = average_ce

        if global_step % LOG_INTERVAL == 0 or global_step == 1:
            average_ms = sum(timing_window) / len(timing_window)
            tokens_per_second = tokens_per_step / (average_ms / 1000.0)
            mfu_string = ''
            if gpu_peak_flops:
                mfu = (flops_per_step / (average_ms / 1000.0)) / gpu_peak_flops * 100
                mfu_string = f' | MFU {mfu:.1f}%'

            per_level = last_auxiliary.get('per_level_entropy', [])
            entropy_string = ' '.join(
                f'L{i}:{e:.3f}' for i, e in enumerate(per_level))

            print(
                f'  Step {global_step}/{MAX_STEPS}'
                f' | CE {average_ce:.4f}'
                f' | τ={tau:.3f} α={alpha:.3f}'
                f' | ∇addr={address_grad:.2e}'
                f' | ent=[{entropy_string}]'
                f' | bal={last_auxiliary["balance_loss"]:.4f}'
                f'{mfu_string}'
                f' | {tokens_per_second:.0f} tok/s',
                flush=True)

            if address_grad < 1e-6 and global_step > 10:
                print('  ⚠️  GRADIENT STARVATION: AddressEncoder grad < 1e-6')

        if global_step % PASSKEY_INTERVAL == 0:
            passkey_results = passkey_accuracy(model, tokenizer, device)
            passkey_parts = [
                f'd={d}:{int(passkey_results[d] * 100)}%'
                for d in PASSKEY_DISTANCES]
            passkey_mean = sum(passkey_results.values()) / len(passkey_results)
            print(f'  📌 Passkey mean={passkey_mean * 100:.1f}%  '
                  + '  '.join(passkey_parts), flush=True)
            model.train()

    # ── Save final checkpoint ─────────────────────────────────────────────────
    clean_state = {
        k.replace('._orig_mod', ''): v
        for k, v in model.state_dict().items()}
    torch.save(clean_state, os.path.join(
        CHECKPOINT_DIRECTORY, 'dart_d512_smoke_best.pt'))

    # ── Final passkey eval ────────────────────────────────────────────────────
    final_passkey = passkey_accuracy(model, tokenizer, device)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed_seconds = time.time() - training_start_time
    memory_megabytes = (
        torch.cuda.max_memory_allocated() / 1e6
        if torch.cuda.is_available() else 0)

    print('\n--- RESULTS ---')
    print(f'steps: {MAX_STEPS}')
    print(f'best_ce_loss: {best_loss:.4f}')
    for distance in PASSKEY_DISTANCES:
        print(f'passkey_d={distance}: {int(final_passkey.get(distance, 0) * 100)}%')
    print(f'memory_mb: {memory_megabytes:.1f}')
    print(f'elapsed_s: {elapsed_seconds:.1f}')
    print(f'parameters: {number_of_parameters:,} ({number_of_parameters / 1e6:.1f}M)')
    print(f'description: DART D={EMBEDDING_DIMENSION} H={NUMBER_OF_HEADS} '
          f'L={NUMBER_OF_LAYERS} FFN={FEEDFORWARD_DIMENSION} '
          f'FA@L{FULL_ATTENTION_LAYER} chunk={CHUNK_SIZE} window={WINDOW_SIZE}')


if __name__ == '__main__':
    train()
