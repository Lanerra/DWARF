"""
Build the DWARF SFT dataset: retrieval-anchored instruction tuning.

Composition (relay-preserving):
  50% SQuAD v2          — document QA: relay exercises FA retrieval directly
  25% TriviaQA          — factual QA: knowledge injection with retrieval structure
  25% Dolly-15k         — instruction format, reasoning chains, open-ended tasks

Why this composition:
  SFT failure with smol-smoltalk was because 0% of examples required
  content-addressed retrieval from provided context. The FA layer's relay
  gradient signal decayed because nothing in the loss required it.

  SQuAD format: "Context: <passage>\nQuestion: <q>\nAnswer: <a>"
  This is structurally identical to passkey — the model must retrieve a specific
  span from a provided context. FA relay is directly exercised by every SQuAD example.

  TriviaQA: factual Q&A exercises knowledge retrieval (different from passkey but
  still a "retrieve and answer" pattern).

  Dolly: instruction following, open-ended generation — chat format without
  retrieval, kept at 25% so it doesn't overwhelm the retrieval signal.

Output format (ChatML):
  <|system|>You are a helpful assistant.<|endoftext|>
  <|user|>{question or instruction}<|endoftext|>
  <|assistant|>{answer}<|endoftext|>

  For SQuAD, the context is prepended to the user turn:
  <|user|>Context: {passage}\n\nQuestion: {question}<|endoftext|>
  <|assistant|>{answer}<|endoftext|>

Output:
  logs/sft_retrieval_dataset.pt   — tokenized, ready for SFT training

Usage:
    .venv/bin/python3 -u scripts/build_dataset_sft.py \
        > logs/build_dataset_sft.log 2>&1 &
"""

import json, os, random, time
import torch
from datasets import load_dataset
from tokenizers import Tokenizer

# ── Config ────────────────────────────────────────────────────────────────────
TOKENIZER_PATH  = 'results/fineweb_tokenizer_32k.json'
OUTPUT_PATH     = 'logs/sft_retrieval_dataset.pt'
MAX_SEQ_LEN     = 2048
VAL_SPLIT       = 500
RANDOM_SEED     = 42

# Target counts — will be trimmed/padded to these if dataset is large enough
SQUAD_TARGET    = 80_000   # SQuAD v2 train has ~130K answerable examples
TRIVIA_TARGET   = 40_000   # TriviaQA has ~78K train examples
DOLLY_TARGET    = 15_000   # Dolly-15k is exactly 15K

os.makedirs('logs', exist_ok=True)
random.seed(RANDOM_SEED)

print('DWARF SFT Dataset Builder — retrieval-anchored composition')
print(f'  Target: ~{SQUAD_TARGET + TRIVIA_TARGET + DOLLY_TARGET:,} examples')
print(f'  Mix: {SQUAD_TARGET//1000}K SQuAD + {TRIVIA_TARGET//1000}K TriviaQA + {DOLLY_TARGET//1000}K Dolly')
print(f'  Tokenizer: {TOKENIZER_PATH}')
print(f'  Max seq len: {MAX_SEQ_LEN}')
print()

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
vocab_size = tokenizer.get_vocab_size()
eos_id = tokenizer.token_to_id('<|endoftext|>')

# Assign special token IDs by encoding them
def tok_ids(text):
    return tokenizer.encode(text, add_special_tokens=False).ids

# Build special token IDs
SYSTEM_OPEN  = tok_ids('<|system|>')
USER_OPEN    = tok_ids('<|user|>')
ASST_OPEN    = tok_ids('<|assistant|>')
EOT          = [eos_id]   # <|endoftext|> doubles as end-of-turn

SYSTEM_TEXT  = 'You are a helpful assistant.'

def make_chatml(user_text, assistant_text, system_text=SYSTEM_TEXT):
    """Build a ChatML-formatted token sequence with loss mask."""
    ids = []
    mask = []  # 1 = compute loss, 0 = ignore

    # System turn (no loss)
    ids += SYSTEM_OPEN + tok_ids(system_text) + EOT
    mask += [0] * (len(SYSTEM_OPEN) + len(tok_ids(system_text)) + 1)

    # User turn (no loss)
    user_ids = tok_ids(user_text)
    ids += USER_OPEN + user_ids + EOT
    mask += [0] * (len(USER_OPEN) + len(user_ids) + 1)

    # Assistant turn (loss here)
    asst_ids = tok_ids(assistant_text)
    ids += ASST_OPEN + asst_ids + EOT
    mask += [0] * len(ASST_OPEN) + [1] * len(asst_ids) + [1]  # loss on answer + EOS

    return ids, mask


def truncate_or_skip(ids, mask, max_len):
    """Truncate to max_len. Returns None if answer portion would be < 4 tokens."""
    if len(ids) <= max_len:
        return ids, mask
    ids = ids[:max_len]
    mask = mask[:max_len]
    # Ensure at least some answer tokens remain
    if sum(mask) < 4:
        return None, None
    return ids, mask


# ── SQuAD v2 ─────────────────────────────────────────────────────────────────
print('Loading SQuAD v2...')
squad_examples = []
squad_ds = load_dataset('rajpurkar/squad_v2', split='train')
for ex in squad_ds:
    # Skip unanswerable questions
    if not ex['answers']['text']:
        continue
    context   = ex['context']
    question  = ex['question']
    answer    = ex['answers']['text'][0]  # first answer

    user_text  = f"Context: {context}\n\nQuestion: {question}"
    asst_text  = answer
    ids, mask  = make_chatml(user_text, asst_text)
    ids, mask  = truncate_or_skip(ids, mask, MAX_SEQ_LEN)
    if ids is None:
        continue
    squad_examples.append((ids, mask))

random.shuffle(squad_examples)
squad_examples = squad_examples[:SQUAD_TARGET]
print(f'  SQuAD v2: {len(squad_examples):,} examples')


# ── TriviaQA ─────────────────────────────────────────────────────────────────
print('Loading TriviaQA...')
trivia_examples = []
trivia_ds = load_dataset('trivia_qa', 'rc', split='train')
for ex in trivia_ds:
    question = ex['question']
    # Use the first alias answer (normalized)
    aliases  = ex['answer'].get('aliases', [])
    value    = ex['answer'].get('value', '')
    answer   = aliases[0] if aliases else value
    if not answer:
        continue

    # Include evidence context if available (exercises relay)
    context_parts = []
    sr = ex.get('search_results', {})
    if sr and isinstance(sr, dict):
        contexts = sr.get('search_context', [])
        if contexts and isinstance(contexts, list) and contexts[0]:
            snippet = contexts[0]
            if len(snippet) > 100:
                context_parts.append(snippet[:1000])

    if context_parts:
        user_text = f"Context: {context_parts[0]}\n\nQuestion: {question}"
    else:
        user_text = question

    ids, mask = make_chatml(user_text, answer)
    ids, mask = truncate_or_skip(ids, mask, MAX_SEQ_LEN)
    if ids is None:
        continue
    trivia_examples.append((ids, mask))

random.shuffle(trivia_examples)
trivia_examples = trivia_examples[:TRIVIA_TARGET]
print(f'  TriviaQA: {len(trivia_examples):,} examples')


# ── Dolly-15k ─────────────────────────────────────────────────────────────────
print('Loading Dolly-15k...')
dolly_examples = []
dolly_ds = load_dataset('databricks/databricks-dolly-15k', split='train')
for ex in dolly_ds:
    instruction = ex['instruction']
    context     = ex.get('context', '').strip()
    response    = ex['response']

    if not response.strip():
        continue

    if context:
        user_text = f"{instruction}\n\nContext: {context}"
    else:
        user_text = instruction

    ids, mask = make_chatml(user_text, response)
    ids, mask = truncate_or_skip(ids, mask, MAX_SEQ_LEN)
    if ids is None:
        continue
    dolly_examples.append((ids, mask))

random.shuffle(dolly_examples)
dolly_examples = dolly_examples[:DOLLY_TARGET]
print(f'  Dolly-15k: {len(dolly_examples):,} examples')


# ── Combine, shuffle, split ───────────────────────────────────────────────────
all_examples = squad_examples + trivia_examples + dolly_examples
random.shuffle(all_examples)
print(f'\nTotal: {len(all_examples):,} examples')

# Pad sequences to fixed length for batching (pad with EOS, mask=0)
def pad_example(ids, mask, length, pad_id):
    if len(ids) < length:
        pad_len = length - len(ids)
        ids  = ids  + [pad_id] * pad_len
        mask = mask + [0]      * pad_len
    return ids, mask

# Find max length in batch — use dynamic bucketing in training instead of fixed pad
# Save as variable-length list for SFT trainer to handle with bucket batching
val_examples   = all_examples[:VAL_SPLIT]
train_examples = all_examples[VAL_SPLIT:]

print(f'Train: {len(train_examples):,} | Val: {len(val_examples):,}')

# Stats
lengths = [len(ids) for ids, _ in train_examples]
print(f'Length stats — min: {min(lengths)} | median: {sorted(lengths)[len(lengths)//2]} | max: {max(lengths)}')
print(f'Retrieval examples (SQuAD+TriviaQA): {len(squad_examples)+len(trivia_examples):,} '
      f'({(len(squad_examples)+len(trivia_examples))/len(all_examples)*100:.0f}%)')

# ── Save ──────────────────────────────────────────────────────────────────────
save_data = {
    'train': train_examples,   # list of (ids, mask) tuples
    'val':   val_examples,
    'vocab_size': vocab_size,
    'max_seq_len': MAX_SEQ_LEN,
    'eos_id': eos_id,
    'tokenizer': TOKENIZER_PATH,
    'composition': {
        'squad_v2':  len(squad_examples),
        'trivia_qa': len(trivia_examples),
        'dolly_15k': len(dolly_examples),
    },
    'description': (
        'Retrieval-anchored SFT dataset for DWARF. '
        '50% SQuAD v2 document QA (relay-preserving), '
        '25% TriviaQA factual QA, '
        '25% Dolly-15k instruction following. '
        'ChatML format. Loss only on assistant turns.'
    ),
}
torch.save(save_data, OUTPUT_PATH)
size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
print(f'\nSaved: {OUTPUT_PATH} ({size_mb:.0f} MB)')
print('Done.')
