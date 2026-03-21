"""
Train a 32K BPE tokenizer on FineWeb (non-Edu) sample-10BT.

Rationale:
  - FineWeb proper (not Edu) gives broader vocabulary coverage: code, informal
    web text, multilingual tokens, technical jargon — all underrepresented in Edu.
  - At 147M+ scale, a tokenizer that compresses diverse text efficiently matters.
  - Training on FineWeb proper future-proofs the tokenizer for any training data.

Target: 1M documents streamed from sample-10BT, 32K vocab BPE, byte-level fallback.
Output: results/fineweb_tokenizer_32k.json

Usage:
    .venv/bin/python3 -u scripts/train_tokenizer_fineweb.py \
        > logs/train_tokenizer_fineweb.log 2>&1 &
"""

import json, os, time
from datasets import load_dataset
from tokenizers import Tokenizer, trainers, pre_tokenizers, normalizers, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

# ── Config ────────────────────────────────────────────────────────────────────
NUM_DOCS       = 1_000_000   # documents to stream for training
MIN_DOC_LEN    = 200         # chars; skip very short docs
VOCAB_SIZE     = 32_000
SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|unk|>"]
FW_DATASET     = 'HuggingFaceFW/fineweb'
FW_SUBSET      = 'sample-10BT'
OUT_PATH       = 'results/fineweb_tokenizer_32k.json'
CACHE_PATH     = 'benchmarks/logs/fineweb_toktraining_docs.jsonl'  # intermediate cache

os.makedirs('results', exist_ok=True)
os.makedirs('benchmarks/logs', exist_ok=True)

print(f'FineWeb BPE Tokenizer Training')
print(f'  Source:  {FW_DATASET} / {FW_SUBSET}')
print(f'  Docs:    {NUM_DOCS:,}')
print(f'  Vocab:   {VOCAB_SIZE:,}')
print(f'  Output:  {OUT_PATH}')
print()

# ── Step 1: Stream documents ──────────────────────────────────────────────────
def stream_docs():
    """Yield text from FineWeb, using JSONL cache if available."""
    if os.path.exists(CACHE_PATH):
        print(f'  Using cached docs: {CACHE_PATH}')
        with open(CACHE_PATH) as f:
            for i, line in enumerate(f):
                if i >= NUM_DOCS:
                    break
                doc = json.loads(line)
                yield doc['text']
        return

    print(f'  Streaming from HuggingFace (will cache to {CACHE_PATH})...')
    t0 = time.time()
    count = 0
    with open(CACHE_PATH, 'w') as cache_f:
        ds = load_dataset(FW_DATASET, name=FW_SUBSET, split='train', streaming=True)
        for doc in ds:
            text = doc.get('text', '')
            if len(text) < MIN_DOC_LEN:
                continue
            cache_f.write(json.dumps({'text': text}) + '\n')
            yield text
            count += 1
            if count % 50_000 == 0:
                elapsed = time.time() - t0
                rate = count / elapsed
                eta = (NUM_DOCS - count) / rate / 60
                print(f'  Streamed {count:,}/{NUM_DOCS:,} docs ({rate:.0f}/s, ETA {eta:.0f}m)')
            if count >= NUM_DOCS:
                break
    print(f'  Done streaming: {count:,} docs cached.')

# ── Step 2: Train tokenizer ───────────────────────────────────────────────────

print('Building tokenizer...')
tokenizer = Tokenizer(BPE(unk_token='<|unk|>'))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
    min_frequency=2,
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

print(f'Training BPE on {NUM_DOCS:,} FineWeb documents...')
t0 = time.time()
tokenizer.train_from_iterator(stream_docs(), trainer=trainer)
elapsed = time.time() - t0
print(f'Training complete in {elapsed/60:.1f} minutes')

# ── Step 3: Verify and save ───────────────────────────────────────────────────

actual_vocab = tokenizer.get_vocab_size()
print(f'Vocab size: {actual_vocab:,}')

# Quick sanity checks
test_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "def forward(self, x: torch.Tensor) -> torch.Tensor:",
    "In the field of machine learning, transformers have",
    "https://example.com/article?id=12345",
    "φ(n) = n ∏(1 - 1/p) for prime p|n",
]
print('\nSanity checks:')
for text in test_texts:
    enc = tokenizer.encode(text)
    ratio = len(text) / len(enc.ids)
    print(f'  {len(enc.ids):4d} tokens ({ratio:.1f} chars/tok): {text[:50]}')

tokenizer.save(OUT_PATH)
print(f'\nSaved: {OUT_PATH}')
print(f'Vocab size: {actual_vocab:,}')
print()
print('Next step: run scripts/build_dataset_fineweb.py to encode 1M docs with this tokenizer.')
