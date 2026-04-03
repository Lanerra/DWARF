"""
Build blended SFT dataset: OpenHermes-2.5 (60%) + Open-Orca GPT-4 subset (40%).

Target: 500K total examples
  - 300K from teknium/OpenHermes-2.5  (diverse CoT, commonsense, reasoning)
  - 200K from Open-Orca/OpenOrca      (GPT-4 augmented portion only, ~1M examples)

Output format: identical to sft_retrieval_dataset.pt
  list of (ids_list, labels_0_1) tuples
  labels: 0 = system/user (ignored), 1 = assistant (supervised)

Why these two:
  OpenHermes-2.5: ~1M GPT-4 generated examples, diverse domains, strong CoT.
    Good for HellaSwag (commonsense) and ARC-Easy gains.
  Open-Orca GPT-4: explicit reasoning chains, step-by-step explanations.
    Specifically closes ARC-Challenge gap (requires reasoning, not just retrieval).
  Neither dataset trains on benchmark distributions directly.

Output: logs/openhermes_orca_sft_encoded.pt
"""

import random, time, os
import torch
from datasets import load_dataset
from tokenizers import Tokenizer

# ── Config ────────────────────────────────────────────────────────────────────
TOKENIZER_PATH      = 'results/fineweb_tokenizer_32k.json'
OUTPUT_PATH         = 'logs/openhermes_orca_sft_encoded.pt'
MAX_SEQ_LEN         = 2048
VAL_SPLIT           = 2000
RANDOM_SEED         = 42

OPENHERMES_TARGET   = 300_000
ORCA_GPT4_TARGET    = 200_000

os.makedirs('logs', exist_ok=True)
random.seed(RANDOM_SEED)

# ── Load tokenizer ─────────────────────────────────────────────────────────────
print(f"Loading tokenizer from {TOKENIZER_PATH}...")
tok = Tokenizer.from_file(TOKENIZER_PATH)
vocab_size  = tok.get_vocab_size()
eos_id      = tok.token_to_id("<|endoftext|>")
sys_open    = tok.encode("<|system|>",    add_special_tokens=False).ids[0]
usr_open    = tok.encode("<|user|>",      add_special_tokens=False).ids[0]
ast_open    = tok.encode("<|assistant|>", add_special_tokens=False).ids[0]
print(f"  vocab={vocab_size}, eos={eos_id}")

def enc(text):
    return tok.encode(str(text), add_special_tokens=False).ids

def make_example(turns):
    """
    turns: list of (role, content) — role in {system, user, assistant}
    Returns (ids, labels) or None if too long / no supervised tokens.
    """
    ids, labels = [], []
    for role, content in turns:
        if role == 'system':
            t = [sys_open] + enc(content) + [eos_id]
            ids.extend(t); labels.extend([0]*len(t))
        elif role == 'user':
            t = [usr_open] + enc(content) + [eos_id]
            ids.extend(t); labels.extend([0]*len(t))
        elif role == 'assistant':
            t = [ast_open] + enc(content) + [eos_id]
            ids.extend(t); labels.extend([1]*len(t))
    if len(ids) > MAX_SEQ_LEN:
        return None
    if sum(labels) == 0:
        return None
    return (ids, labels)


# ── OpenHermes-2.5 ─────────────────────────────────────────────────────────────
# Format: {conversations: [{from: 'human'/'gpt'/'system', value: '...'}], ...}
# 'human' → user, 'gpt' → assistant, 'system' → system

def process_openhermes(ex):
    convs = ex.get('conversations', [])
    turns = []
    for msg in convs:
        role = msg.get('from', '')
        val  = msg.get('value', '')
        if role == 'system':
            turns.append(('system', val))
        elif role in ('human', 'user'):
            turns.append(('user', val))
        elif role in ('gpt', 'assistant'):
            turns.append(('assistant', val))
    return make_example(turns)

print(f"\n{'='*60}")
print(f"Downloading OpenHermes-2.5 (target: {OPENHERMES_TARGET:,} examples)...")
print(f"{'='*60}")

oh_examples = []
oh_skipped  = 0
t0 = time.time()

ds_oh = load_dataset('teknium/OpenHermes-2.5', split='train', streaming=True)

for i, ex in enumerate(ds_oh):
    result = process_openhermes(ex)
    if result is None:
        oh_skipped += 1
    else:
        oh_examples.append(result)

    if (i+1) % 50_000 == 0:
        elapsed = time.time() - t0
        rate = (i+1) / elapsed
        print(f"  [{i+1:,} raw] kept={len(oh_examples):,}, skipped={oh_skipped:,}, "
              f"{rate:.0f} ex/s, {elapsed:.0f}s elapsed")

    if len(oh_examples) >= OPENHERMES_TARGET:
        print(f"  Reached target {OPENHERMES_TARGET:,} — stopping OpenHermes stream")
        break

print(f"OpenHermes done: {len(oh_examples):,} kept from {i+1:,} raw "
      f"({oh_skipped:,} skipped) in {time.time()-t0:.1f}s")


# ── Open-Orca GPT-4 subset ─────────────────────────────────────────────────────
# Parquet: 1M-GPT4-Augmented.parquet
# Format: {id, system_prompt, question, response}

def process_orca(ex):
    turns = []
    sys_p = ex.get('system_prompt', '').strip()
    q     = ex.get('question', '').strip()
    r     = ex.get('response', '').strip()
    if sys_p:
        turns.append(('system', sys_p))
    if q:
        turns.append(('user', q))
    if r:
        turns.append(('assistant', r))
    return make_example(turns)

print(f"\n{'='*60}")
print(f"Downloading Open-Orca GPT-4 subset (target: {ORCA_GPT4_TARGET:,} examples)...")
print(f"{'='*60}")

orca_examples = []
orca_skipped  = 0
t0 = time.time()

# Load only the GPT-4 parquet file
ds_orca = load_dataset(
    'Open-Orca/OpenOrca',
    data_files={'train': '1M-GPT4-Augmented.parquet'},
    split='train',
    streaming=True,
)

for i, ex in enumerate(ds_orca):
    result = process_orca(ex)
    if result is None:
        orca_skipped += 1
    else:
        orca_examples.append(result)

    if (i+1) % 50_000 == 0:
        elapsed = time.time() - t0
        rate = (i+1) / elapsed
        print(f"  [{i+1:,} raw] kept={len(orca_examples):,}, skipped={orca_skipped:,}, "
              f"{rate:.0f} ex/s, {elapsed:.0f}s elapsed")

    if len(orca_examples) >= ORCA_GPT4_TARGET:
        print(f"  Reached target {ORCA_GPT4_TARGET:,} — stopping Orca stream")
        break

print(f"Open-Orca GPT-4 done: {len(orca_examples):,} kept from {i+1:,} raw "
      f"({orca_skipped:,} skipped) in {time.time()-t0:.1f}s")


# ── Blend & split ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Blending and splitting...")
print(f"{'='*60}")

all_examples = oh_examples + orca_examples
random.shuffle(all_examples)

val   = all_examples[:VAL_SPLIT]
train = all_examples[VAL_SPLIT:]

# Sequence length stats
lens = [len(x[0]) for x in train]
print(f"Train: {len(train):,} examples")
print(f"Val:   {len(val):,} examples")
print(f"Seq lengths — min: {min(lens)}, max: {max(lens)}, mean: {sum(lens)/len(lens):.0f}")
print(f"Blend: {len(oh_examples):,} OpenHermes + {len(orca_examples):,} Orca-GPT4")
print(f"Actual ratio: {len(oh_examples)/len(all_examples)*100:.1f}% / "
      f"{len(orca_examples)/len(all_examples)*100:.1f}%")

# ── Save ───────────────────────────────────────────────────────────────────────
dataset = {
    'train':        train,
    'val':          val,
    'vocab_size':   vocab_size,
    'max_seq_len':  MAX_SEQ_LEN,
    'eos_id':       eos_id,
    'tokenizer':    TOKENIZER_PATH,
    'composition':  f'OpenHermes-2.5 ({len(oh_examples):,}) + Open-Orca GPT-4 ({len(orca_examples):,})',
    'description':  '60/40 blend targeting HellaSwag + ARC-Challenge gains. '
                    'No benchmark distributions included.',
    'openhermes_count': len(oh_examples),
    'orca_gpt4_count':  len(orca_examples),
}

print(f"\nSaving to {OUTPUT_PATH}...")
torch.save(dataset, OUTPUT_PATH)

import os
size_gb = os.path.getsize(OUTPUT_PATH) / 1e9
print(f"Done. {OUTPUT_PATH} ({size_gb:.2f} GB)")
print(f"\nReady for: train/train_d768_l32_sft_4090_bf16.py")
print(f"  Change SFT_DATASET_PATH = 'logs/openhermes_orca_sft_encoded.pt'")
