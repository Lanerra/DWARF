"""
Train a 32K BPE tokenizer on the mixed dataset (FineWeb 60% / PG19 25% / Stack 15%).

Rationale:
  The existing fineweb_tokenizer_32k was trained on FineWeb proper (web text).
  When applied to PG19 (Victorian prose) and The Stack (code), it fragments tokens
  sub-optimally — high perplexity not from model failure but from tokenizer mismatch.
  A tokenizer trained on the actual training distribution compresses all three sources
  efficiently, giving the model a fair substrate to learn from.

Training corpus: proportionally sampled from all three sources:
  - 600K docs from FineWeb-Edu sample-10BT  (60%)
  - 250K docs from PG19                     (25%)
  - 150K docs from The Stack dedup (Python/JS/Rust, ≥200 chars)  (15%)

Total: 1M documents. Same NUM_DOCS as fineweb_tokenizer_32k for direct comparison.

Output: results/mixed_tokenizer_32k.json

Usage:
    .venv/bin/python3 -u scripts/train_tokenizer_mixed.py \
        > logs/train_tokenizer_mixed.log 2>&1 &
"""

import json, os, random, time
# Force offline mode — PG19 and FineWeb are in local cache; Stack uses local JSON cache
os.environ['HF_DATASETS_OFFLINE'] = '1'
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

# ── Config ────────────────────────────────────────────────────────────────────
VOCAB_SIZE     = 32_000
SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|unk|>"]
OUT_PATH       = 'results/mixed_tokenizer_32k.json'

# Source proportions
N_FINEWEB  = 600_000   # 60%
N_PG19     = 250_000   # 25%
N_STACK    = 150_000   # 15%
MIN_DOC_LEN = 200      # chars

# Cache paths for each source (reuse if already present)
CACHE_FW    = 'benchmarks/logs/fineweb_toktraining_docs.jsonl'      # may already exist
CACHE_PG19  = 'benchmarks/logs/pg19_toktraining_docs.jsonl'
CACHE_STACK = 'benchmarks/logs/stack_toktraining_docs.jsonl'

os.makedirs('results', exist_ok=True)
os.makedirs('benchmarks/logs', exist_ok=True)
os.makedirs('logs', exist_ok=True)

print(f'Mixed BPE Tokenizer Training (32K vocab)')
print(f'  FineWeb-Edu:   {N_FINEWEB:,} docs (60%)')
print(f'  PG19:          {N_PG19:,} docs (25%)')
print(f'  The Stack:     {N_STACK:,} docs (15%)')
print(f'  Total:         {N_FINEWEB+N_PG19+N_STACK:,} docs')
print(f'  Output:        {OUT_PATH}')
print()

# ── Source streamers ──────────────────────────────────────────────────────────

def stream_fineweb(n_docs):
    """Stream FineWeb-Edu docs, using cache if available."""
    count = 0
    if os.path.exists(CACHE_FW):
        print(f'  [FineWeb] Using cache: {CACHE_FW}')
        with open(CACHE_FW) as f:
            for line in f:
                if count >= n_docs:
                    break
                doc = json.loads(line)
                text = doc.get('text', '')
                if len(text) >= MIN_DOC_LEN:
                    yield text
                    count += 1
        print(f'  [FineWeb] Yielded {count:,} docs from cache')
        return

    print(f'  [FineWeb] Streaming from HuggingFace...')
    t0 = time.time()
    with open(CACHE_FW, 'w') as cache_f:
        ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                          split='train', streaming=True)
        for doc in ds:
            text = doc.get('text', '')
            if len(text) < MIN_DOC_LEN:
                continue
            cache_f.write(json.dumps({'text': text}) + '\n')
            yield text
            count += 1
            if count % 50_000 == 0:
                rate = count / (time.time() - t0)
                print(f'  [FineWeb] {count:,}/{n_docs:,} ({rate:.0f}/s)')
            if count >= n_docs:
                break
    print(f'  [FineWeb] Done: {count:,} docs')

def stream_pg19(n_docs):
    """Stream PG19 public domain books, using cache if available."""
    count = 0
    if os.path.exists(CACHE_PG19):
        print(f'  [PG19] Using cache: {CACHE_PG19}')
        with open(CACHE_PG19) as f:
            for line in f:
                if count >= n_docs:
                    break
                doc = json.loads(line)
                text = doc.get('text', '')
                if len(text) >= MIN_DOC_LEN:
                    yield text
                    count += 1
        print(f'  [PG19] Yielded {count:,} docs from cache')
        return

    print(f'  [PG19] Streaming from HuggingFace...')
    t0 = time.time()

    # PG19 books are very long — chunk each into ~4096 char segments
    # so the tokenizer sees varied vocabulary, not one book at a time
    CHUNK_SIZE = 4096

    with open(CACHE_PG19, 'w') as cache_f:
        # emozilla/pg19 is the parquet mirror with stable loading
        ds = load_dataset('emozilla/pg19', split='train', streaming=True)
        for book in ds:
            text = book.get('text', '')
            if not text:
                continue
            # Chunk the book
            for start in range(0, len(text), CHUNK_SIZE):
                chunk = text[start:start + CHUNK_SIZE]
                if len(chunk) < MIN_DOC_LEN:
                    continue
                cache_f.write(json.dumps({'text': chunk}) + '\n')
                yield chunk
                count += 1
                if count % 50_000 == 0:
                    rate = count / (time.time() - t0)
                    print(f'  [PG19] {count:,}/{n_docs:,} ({rate:.0f}/s)')
                if count >= n_docs:
                    break
            if count >= n_docs:
                break
    print(f'  [PG19] Done: {count:,} chunks')

STACK_DOC_CACHE = 'logs/stack_doc_cache.json'  # JSON array of raw code strings

def stream_stack(n_docs):
    """Stream code from local stack_doc_cache.json (JSON array), with JSONL cache fallback."""
    count = 0

    # Prefer JSONL cache (per-doc format)
    if os.path.exists(CACHE_STACK) and os.path.getsize(CACHE_STACK) > 0:
        print(f'  [Stack] Using JSONL cache: {CACHE_STACK}')
        with open(CACHE_STACK) as f:
            for line in f:
                if count >= n_docs:
                    break
                doc = json.loads(line)
                text = doc.get('text', '')
                if len(text) >= MIN_DOC_LEN:
                    yield text
                    count += 1
        print(f'  [Stack] Yielded {count:,} docs from JSONL cache')
        return

    # Fall back to JSON array cache from build_dataset_stack.py
    if os.path.exists(STACK_DOC_CACHE) and os.path.getsize(STACK_DOC_CACHE) > 0:
        print(f'  [Stack] Loading from JSON array cache: {STACK_DOC_CACHE}')
        t0 = time.time()
        with open(STACK_DOC_CACHE) as f:
            docs = json.load(f)
        print(f'  [Stack] Loaded {len(docs):,} docs in {time.time()-t0:.1f}s')

        # Write JSONL cache for future use
        with open(CACHE_STACK, 'w') as cache_f:
            for text in docs:
                if not isinstance(text, str):
                    continue
                if len(text) < MIN_DOC_LEN:
                    continue
                cache_f.write(json.dumps({'text': text}) + '\n')
                yield text
                count += 1
                if count >= n_docs:
                    break
        print(f'  [Stack] Yielded {count:,} docs (JSONL cache written)')
        return

    # Last resort: stream from HuggingFace
    print(f'  [Stack] Streaming from HuggingFace (no local cache found)...')
    t0 = time.time()
    per_lang = n_docs // 3

    with open(CACHE_STACK, 'w') as cache_f:
        for lang, target in [('python', per_lang), ('javascript', per_lang),
                              ('rust', n_docs - 2*per_lang)]:
            lang_count = 0
            ds = load_dataset('bigcode/the-stack-dedup', data_dir=f'data/{lang}',
                              split='train', streaming=True)
            for doc in ds:
                text = doc.get('content', '')
                if len(text) < MIN_DOC_LEN:
                    continue
                cache_f.write(json.dumps({'text': text}) + '\n')
                yield text
                count += 1
                lang_count += 1
                if count % 25_000 == 0:
                    rate = count / (time.time() - t0)
                    print(f'  [Stack/{lang}] {lang_count:,}/{target:,} ({rate:.0f}/s)')
                if lang_count >= target:
                    break
    print(f'  [Stack] Done: {count:,} docs')

# ── Interleaved iterator ──────────────────────────────────────────────────────

def mixed_iterator():
    """Yield documents from all three sources, interleaved proportionally.
    Interleaving ensures the tokenizer sees all vocabulary types throughout
    training, not in three separate blocks."""
    print('Building mixed training iterator...')

    fw_gen    = stream_fineweb(N_FINEWEB)
    pg19_gen  = stream_pg19(N_PG19)
    stack_gen = stream_stack(N_STACK)

    # Proportional interleaving: for every 20 docs, yield 12 FW + 5 PG19 + 3 Stack
    sources = (
        [(fw_gen,    12)] +
        [(pg19_gen,   5)] +
        [(stack_gen,  3)]
    )

    total = 0
    while True:
        any_left = False
        for gen, n in sources:
            for _ in range(n):
                try:
                    text = next(gen)
                    yield text
                    total += 1
                    if total % 100_000 == 0:
                        print(f'  Yielded {total:,} docs for tokenizer training...')
                    any_left = True
                except StopIteration:
                    pass
        if not any_left:
            break

    print(f'  Total docs yielded: {total:,}')

# ── Train tokenizer ───────────────────────────────────────────────────────────

print('Initializing BPE tokenizer...')
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

print(f'Training BPE tokenizer on mixed corpus...')
t0 = time.time()
tokenizer.train_from_iterator(mixed_iterator(), trainer=trainer)
elapsed = time.time() - t0
print(f'Training complete in {elapsed/60:.1f} minutes')

# ── Verify and save ───────────────────────────────────────────────────────────

actual_vocab = tokenizer.get_vocab_size()
print(f'\nVocab size: {actual_vocab:,}')

test_texts = [
    # FineWeb-style
    "The researchers found that transformer models trained on diverse datasets",
    # PG19-style (Victorian prose)
    "It was a truth universally acknowledged, that a single man in possession",
    # Code (Python)
    "def forward(self, x: torch.Tensor) -> torch.Tensor:\n    return self.attn(x)",
    # Code (Rust)
    "impl<T: Clone> RelayChain<T> where T: Send + Sync {",
    # Mixed technical
    "φ(n) = n ∏(1 - 1/p) for prime p dividing n",
]

print('\nSanity checks (chars/token — higher = more efficient compression):')
print(f'  {"Source":<12} {"Tokens":>7} {"Chars/tok":>10}  Text')
print(f'  {"-"*12} {"-"*7} {"-"*10}  {"-"*40}')
for i, text in enumerate(test_texts):
    enc = tokenizer.encode(text)
    ratio = len(text) / len(enc.ids)
    sources = ['fineweb', 'fineweb', 'pg19', 'code-py', 'code-rs', 'technical']
    label = sources[i] if i < len(sources) else '?'
    print(f'  {label:<12} {len(enc.ids):>7} {ratio:>10.2f}  {text[:50]}')

# Also compare against fineweb_tokenizer_32k if it exists
fw_tok_path = 'results/fineweb_tokenizer_32k.json'
if os.path.exists(fw_tok_path):
    from tokenizers import Tokenizer as T2
    fw_tok = T2.from_file(fw_tok_path)
    print(f'\nComparison vs fineweb_tokenizer_32k (higher chars/tok = better compression):')
    print(f'  {"Text":<42} {"FW tok":>8} {"Mix tok":>8} {"Delta":>8}')
    print(f'  {"-"*42} {"-"*8} {"-"*8} {"-"*8}')
    for text in test_texts:
        fw_ids  = fw_tok.encode(text).ids
        mix_ids = tokenizer.encode(text).ids
        delta = len(fw_ids) - len(mix_ids)
        marker = '✓' if delta >= 0 else '✗'
        print(f'  {text[:42]:<42} {len(fw_ids):>8} {len(mix_ids):>8} {delta:>+7} {marker}')

tokenizer.save(OUT_PATH)
print(f'\nSaved: {OUT_PATH}')
print(f'Vocab size: {actual_vocab:,}')
print()
print('Next step: run scripts/build_dataset_mixed_v2.py to re-encode the')
print('mixed dataset with this tokenizer → logs/mixed_encoded_2048_v2.pt')
