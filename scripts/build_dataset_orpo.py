"""
Encode orpo-dpo-mix-40k for DWARF ORPO training.

Format: chosen/rejected message lists → tokenized preference triples
Output: logs/orpo_mix_encoded.pt
  dict with keys:
    'train': list of {'prompt_ids', 'chosen_ids', 'rejected_ids'}
    'val':   same (last VAL_SPLIT examples)

Encoding:
  prompt_ids  = system + all turns up to (not including) last assistant turn
  chosen_ids  = last assistant turn from chosen (loss target)
  rejected_ids = last assistant turn from rejected (contrast target)

Usage:
    .venv/bin/python3 -u scripts/build_dataset_orpo.py \
        > logs/build_orpo.log 2>&1 &
"""

import os, random, time
from multiprocessing import Pool
import torch
from datasets import load_dataset
from tokenizers import Tokenizer

TOKENIZER_PATH = 'results/fineweb_tokenizer_32k.json'
OUTPUT_PATH    = 'logs/orpo_mix_encoded.pt'
MAX_SEQ_LEN    = 2048
MAX_RESP_LEN   = 1024   # max tokens for chosen/rejected response
VAL_SPLIT      = 500
RANDOM_SEED    = 42
NUM_WORKERS    = 12

os.makedirs('logs', exist_ok=True)
random.seed(RANDOM_SEED)

print('ORPO-DPO-Mix-40k Encoder')
print(f'  Tokenizer    : {TOKENIZER_PATH}')
print(f'  Max seq      : {MAX_SEQ_LEN}')
print(f'  Max resp     : {MAX_RESP_LEN}')
print(f'  Val split    : {VAL_SPLIT}')
print(f'  Workers      : {NUM_WORKERS}')
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

def _role_to_tokens(role, content):
    """Convert a message role/content pair to token ids."""
    content = (content or '').strip()
    if role in ('human', 'user'):
        return USER_OPEN + _tok(content) + EOT, False
    elif role in ('gpt', 'assistant'):
        return ASST_OPEN + _tok(content) + EOT, True
    elif role == 'system':
        return SYS_OPEN + _tok(content) + EOT, False
    return [], False

def _extract_last_assistant(messages):
    """Find the last assistant turn and return (prefix_ids, response_ids)."""
    # messages: list of dicts with 'role'/'from' and 'content'/'value'
    all_turns = []
    for m in messages:
        role    = m.get('role', m.get('from', ''))
        content = m.get('content', m.get('value', ''))
        ids, is_asst = _role_to_tokens(role, content)
        if ids:
            all_turns.append((ids, is_asst))

    # Find last assistant turn
    last_asst_idx = None
    for i in range(len(all_turns) - 1, -1, -1):
        if all_turns[i][1]:
            last_asst_idx = i
            break

    if last_asst_idx is None:
        return None, None

    prefix_ids  = []
    for ids, _ in all_turns[:last_asst_idx]:
        prefix_ids += ids
    response_ids = all_turns[last_asst_idx][0]
    return prefix_ids, response_ids

def encode_example(example):
    try:
        chosen   = example.get('chosen', [])
        rejected = example.get('rejected', [])

        if not chosen or not rejected:
            return None

        # Build system prefix once
        sys_ids = SYS_OPEN + _tok(SYSTEM_TEXT) + EOT

        chosen_prefix,   chosen_resp   = _extract_last_assistant(chosen)
        rejected_prefix, rejected_resp = _extract_last_assistant(rejected)

        if chosen_resp is None or rejected_resp is None:
            return None

        # Use chosen prefix as the canonical prompt (they should be identical)
        prompt_ids = sys_ids + chosen_prefix
        if len(prompt_ids) > MAX_SEQ_LEN:
            prompt_ids = prompt_ids[-MAX_SEQ_LEN:]  # keep recent context

        chosen_ids   = chosen_resp[:MAX_RESP_LEN + len(ASST_OPEN) + len(EOT)]
        rejected_ids = rejected_resp[:MAX_RESP_LEN + len(ASST_OPEN) + len(EOT)]

        # Must have actual response content
        if len(chosen_ids) <= len(ASST_OPEN) + len(EOT):
            return None
        if len(rejected_ids) <= len(ASST_OPEN) + len(EOT):
            return None

        return {
            'prompt_ids':   torch.tensor(prompt_ids,   dtype=torch.long),
            'chosen_ids':   torch.tensor(chosen_ids,   dtype=torch.long),
            'rejected_ids': torch.tensor(rejected_ids, dtype=torch.long),
        }
    except Exception:
        return None

def main():
    t0 = time.time()
    print('Loading dataset...')
    ds = load_dataset('mlabonne/orpo-dpo-mix-40k', split='train')
    examples = list(ds)
    random.shuffle(examples)
    print(f'  Loaded {len(examples):,} examples in {time.time()-t0:.1f}s')

    print(f'Encoding with {NUM_WORKERS} workers...')
    t1 = time.time()
    with Pool(NUM_WORKERS) as pool:
        results = pool.map(encode_example, examples, chunksize=200)

    encoded = [r for r in results if r is not None]
    skipped = len(examples) - len(encoded)
    print(f'  Encoded {len(encoded):,} | Skipped {skipped:,} ({time.time()-t1:.1f}s)')

    # Stats
    p_lens = sorted(len(e['prompt_ids'])   for e in encoded)
    c_lens = sorted(len(e['chosen_ids'])   for e in encoded)
    r_lens = sorted(len(e['rejected_ids']) for e in encoded)
    n = len(encoded)
    print(f'  Prompt  p50={p_lens[n//2]} p90={p_lens[int(n*0.9)]} max={p_lens[-1]}')
    print(f'  Chosen  p50={c_lens[n//2]} p90={c_lens[int(n*0.9)]} max={c_lens[-1]}')
    print(f'  Rejected p50={r_lens[n//2]} p90={r_lens[int(n*0.9)]} max={r_lens[-1]}')

    val   = encoded[:VAL_SPLIT]
    train = encoded[VAL_SPLIT:]
    print(f'  Train: {len(train):,} | Val: {len(val):,}')

    print(f'Saving to {OUTPUT_PATH}...')
    torch.save({'train': train, 'val': val}, OUTPUT_PATH)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f'  Saved {size_mb:.0f} MB in {time.time()-t0:.1f}s total')

if __name__ == '__main__':
    main()
