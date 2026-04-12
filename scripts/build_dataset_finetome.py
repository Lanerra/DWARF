"""
Encode FineTome-100k-dedup for DWARF SFT training.

Format: ShareGPT conversations (from/value) → ChatML → tokenized
Output: logs/finetome_sft_encoded.pt
  dict with keys:
    'train': list of {'input_ids': LongTensor, 'label_mask': BoolTensor}
    'val':   same (last VAL_SPLIT examples)

Label mask = True only on assistant turns (loss computed there only).

Usage:
    .venv/bin/python3 -u scripts/build_dataset_finetome.py \
        > logs/build_finetome.log 2>&1 &
"""

import os, random, time
from multiprocessing import Pool
import torch
from datasets import load_dataset
from tokenizers import Tokenizer

TOKENIZER_PATH = 'results/fineweb_tokenizer_32k.json'
OUTPUT_PATH    = 'logs/finetome_sft_encoded.pt'
MAX_SEQ_LEN    = 2048
VAL_SPLIT      = 1000
RANDOM_SEED    = 42
NUM_WORKERS    = 12

os.makedirs('logs', exist_ok=True)
random.seed(RANDOM_SEED)

print('FineTome-100k Encoder')
print(f'  Tokenizer : {TOKENIZER_PATH}')
print(f'  Max seq   : {MAX_SEQ_LEN}')
print(f'  Val split : {VAL_SPLIT}')
print(f'  Workers   : {NUM_WORKERS}')
print()

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
eos_id = tokenizer.token_to_id('<|endoftext|>')

def _tok(text):
    return tokenizer.encode(text, add_special_tokens=False).ids

SYS_OPEN  = _tok('<|system|>')
USER_OPEN = _tok('<|user|>')
ASST_OPEN = _tok('<|assistant|>')
EOT       = [eos_id]

SYSTEM_TEXT = 'You are a helpful assistant.'

def encode_conversation(conversations):
    """Convert ShareGPT conversation to (input_ids, label_mask)."""
    input_ids  = []
    label_mask = []

    # System turn
    sys_ids = SYS_OPEN + _tok(SYSTEM_TEXT) + EOT
    input_ids  += sys_ids
    label_mask += [False] * len(sys_ids)

    for turn in conversations:
        role  = turn.get('from', '')
        value = turn.get('value', '').strip()
        if not value:
            continue

        if role == 'human':
            ids = USER_OPEN + _tok(value) + EOT
            input_ids  += ids
            label_mask += [False] * len(ids)
        elif role == 'gpt':
            ids = ASST_OPEN + _tok(value) + EOT
            input_ids  += ids
            label_mask += [True] * len(ids)
        # skip any other roles

    if len(input_ids) > MAX_SEQ_LEN:
        input_ids  = input_ids[:MAX_SEQ_LEN]
        label_mask = label_mask[:MAX_SEQ_LEN]

    # Must have at least one assistant token with loss
    if not any(label_mask):
        return None

    return {
        'input_ids':  torch.tensor(input_ids,  dtype=torch.long),
        'label_mask': torch.tensor(label_mask, dtype=torch.bool),
    }

def worker(example):
    try:
        return encode_conversation(example['conversations'])
    except Exception:
        return None

def main():
    t0 = time.time()
    print('Loading dataset...')
    ds = load_dataset('mlabonne/FineTome-100k-dedup', split='train')
    examples = list(ds)
    random.shuffle(examples)
    print(f'  Loaded {len(examples):,} examples in {time.time()-t0:.1f}s')

    print(f'Encoding with {NUM_WORKERS} workers...')
    t1 = time.time()
    with Pool(NUM_WORKERS) as pool:
        results = pool.map(worker, examples, chunksize=500)

    encoded = [r for r in results if r is not None]
    skipped = len(examples) - len(encoded)
    print(f'  Encoded {len(encoded):,} | Skipped {skipped:,} ({time.time()-t1:.1f}s)')

    # Length stats
    lengths = [len(e['input_ids']) for e in encoded]
    lengths.sort()
    n = len(lengths)
    print(f'  Length p50={lengths[n//2]} p90={lengths[int(n*0.9)]} p99={lengths[int(n*0.99)]} max={lengths[-1]}')

    val   = encoded[:VAL_SPLIT]
    train = encoded[VAL_SPLIT:]
    print(f'  Train: {len(train):,} | Val: {len(val):,}')

    print(f'Saving to {OUTPUT_PATH}...')
    torch.save({'train': train, 'val': val}, OUTPUT_PATH)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f'  Saved {size_mb:.0f} MB in {time.time()-t0:.1f}s total')

if __name__ == '__main__':
    main()
