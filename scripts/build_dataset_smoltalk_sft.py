"""
Tokenize smoltalk_50k.pt into the DWARF SFT format for the scale_embed freeze ablation.

Input:  logs/smoltalk_50k.pt  — raw chat dicts {messages, source}, 50K examples
Output: logs/smoltalk_sft_encoded.pt — (ids_list, labels_0_1) tuples

Label format: 0 = prompt/system/user tokens (not supervised), 1 = assistant tokens.

This is the identical format as sft_retrieval_dataset.pt so the existing
train_d768_l32_sft_4090_bf16.py script can load it unchanged (just swap dataset path).

Hypothesis being tested:
  The March 21 smoltalk SFT failure (passkey 99.2% → 72.5%) was caused by
  scale_embed not being frozen, NOT by the chat-only dataset composition.
  If passkey survives with frozen scale_embed on smoltalk, the freeze was the fix.
"""

import random, time
import torch
from tokenizers import Tokenizer

# ── Config ────────────────────────────────────────────────────────────────────
TOKENIZER_PATH  = 'results/fineweb_tokenizer_32k.json'
INPUT_PATH      = 'logs/smoltalk_50k.pt'
OUTPUT_PATH     = 'logs/smoltalk_sft_encoded.pt'
MAX_SEQ_LEN     = 2048
VAL_SPLIT       = 500
RANDOM_SEED     = 42

random.seed(RANDOM_SEED)

# ── Load tokenizer ─────────────────────────────────────────────────────────────
print(f"Loading tokenizer from {TOKENIZER_PATH}...")
tok = Tokenizer.from_file(TOKENIZER_PATH)
vocab_size = tok.get_vocab_size()
eos_id = tok.token_to_id("<|endoftext|>")
print(f"  vocab_size={vocab_size}, eos_id={eos_id}")

# ── Special token IDs ──────────────────────────────────────────────────────────
sys_open  = tok.token_to_id("<|system|>")    or tok.encode("<|system|>").ids[0]
usr_open  = tok.token_to_id("<|user|>")      or tok.encode("<|user|>").ids[0]
ast_open  = tok.token_to_id("<|assistant|>") or tok.encode("<|assistant|>").ids[0]

def encode_no_special(text):
    """Encode text, returning list of token ids (no added specials)."""
    return tok.encode(text, add_special_tokens=False).ids

def build_example(messages):
    """
    Convert a list of {role, content} messages into (ids, labels) tuple.
    labels: 0 = ignore (system/user), 1 = supervise (assistant).
    Returns None if sequence > MAX_SEQ_LEN.
    """
    ids, labels = [], []

    for msg in messages:
        role = msg['role']
        content = msg['content']

        if role == 'system':
            turn_ids = [sys_open] + encode_no_special(content) + [eos_id]
            ids.extend(turn_ids)
            labels.extend([0] * len(turn_ids))

        elif role == 'user':
            turn_ids = [usr_open] + encode_no_special(content) + [eos_id]
            ids.extend(turn_ids)
            labels.extend([0] * len(turn_ids))

        elif role == 'assistant':
            turn_ids = [ast_open] + encode_no_special(content) + [eos_id]
            ids.extend(turn_ids)
            labels.extend([1] * len(turn_ids))  # supervise assistant tokens

    if len(ids) > MAX_SEQ_LEN:
        return None
    if sum(labels) == 0:
        return None  # no supervised tokens — skip

    return (ids, labels)


# ── Load raw smoltalk ──────────────────────────────────────────────────────────
print(f"\nLoading {INPUT_PATH}...")
raw = torch.load(INPUT_PATH, map_location='cpu', weights_only=False)
train_raw = raw['train']   # list of {messages, source} dicts
val_raw   = raw['val']
print(f"  train={len(train_raw)}, val={len(val_raw)}")

# ── Encode ─────────────────────────────────────────────────────────────────────
def encode_split(examples, name):
    encoded, skipped_long, skipped_empty = [], 0, 0
    t0 = time.time()
    for i, ex in enumerate(examples):
        if i % 5000 == 0 and i > 0:
            elapsed = time.time() - t0
            print(f"  [{name}] {i}/{len(examples)} — {len(encoded)} kept, "
                  f"{skipped_long} too long, {skipped_empty} no labels "
                  f"({elapsed:.0f}s)")
        result = build_example(ex['messages'])
        if result is None:
            # check which reason
            if sum(1 for m in ex['messages'] if m['role']=='assistant') == 0:
                skipped_empty += 1
            else:
                skipped_long += 1
            continue
        encoded.append(result)
    print(f"  [{name}] done: {len(encoded)}/{len(examples)} kept "
          f"({skipped_long} too long, {skipped_empty} no-label), "
          f"{time.time()-t0:.1f}s")
    return encoded

print("\nEncoding train split...")
train_enc = encode_split(train_raw, 'train')

print("Encoding val split...")
val_enc = encode_split(val_raw, 'val')

# Sequence length stats
train_lens = [len(x[0]) for x in train_enc]
print(f"\nTrain seq lengths: min={min(train_lens)}, "
      f"max={max(train_lens)}, "
      f"mean={sum(train_lens)/len(train_lens):.0f}")

# ── Save ───────────────────────────────────────────────────────────────────────
dataset = {
    'train':        train_enc,
    'val':          val_enc,
    'vocab_size':   vocab_size,
    'max_seq_len':  MAX_SEQ_LEN,
    'eos_id':       eos_id,
    'tokenizer':    TOKENIZER_PATH,
    'composition':  'smoltalk_50k (chat-only, role-masked labels)',
    'description':  'Ablation: scale_embed frozen during SFT to test whether '
                    'dataset composition (vs freeze) caused Mar21 passkey collapse',
}

torch.save(dataset, OUTPUT_PATH)
print(f"\nSaved {OUTPUT_PATH}")
print(f"  train: {len(train_enc)} examples")
print(f"  val:   {len(val_enc)} examples")
