"""
Byte-Patch Preprocessor — SpaceByte-style word-boundary patching.

Splits OpenWebText into word-boundary patches:
  - Each patch = leading whitespace + word, UTF-8 encoded
  - Truncated/padded to MAX_PATCH_BYTES (default 12)
  - Byte encoding: 0=pad, 1-256=bytes (1-indexed so 0 is unambiguous pad)
  - Packs into sequences of NUM_PATCHES (2048) patches
  - Output: uint8 tensors [N_seqs, NUM_PATCHES, MAX_PATCH_BYTES]

Why word-boundary patching vs pure bytes:
  - Pure bytes: 2048 bytes ≈ 400 words. Good spatial consistency, tiny context.
  - Word-boundary patches: 2048 patches ≈ 2048 words. Same coverage as BPE
    runs while keeping spatial structure consistent (each wave field position
    corresponds to one word, not a variable-length BPE token that might be
    1 char or 15 chars).
  - The Wave Field's convolution then means "j0 RF=4 tokens = 4 words"
    regardless of word length — spatially consistent in a way BPE is not.

Run:
  python3 benchmarks/prep_byte_patches.py

Outputs to: wave-field-llm/byte_patch_data/
  byte_patches_train.pt   [N_train, 2048, 12] uint8
  byte_patches_val.pt     [N_val,   2048, 12] uint8
  byte_patches_test.pt    [N_test,  2048, 12] uint8
  byte_patches_meta.json  stats + coverage info

CPU-only. Runs in background while GPU trains other conditions.
"""

import json, os, re, sys, time
import numpy as np
import torch

# ─── Config ───────────────────────────────────────────────────────────────────

MAX_PATCH_BYTES = 16      # fits >99% of English words incl. leading space
NUM_PATCHES     = 2048    # matches all prior conditions
NUM_DOCS        = 100_000 # same dataset size as BPE runs
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR      = os.path.join(SCRIPT_DIR, '..', 'byte_patch_data')


# ─── Word-boundary splitting ──────────────────────────────────────────────────

def split_patches(text):
    """
    Split text into word-boundary patches (SpaceByte-style).

    Each patch captures: [optional_leading_whitespace + word]
    The leading space encodes the word-boundary signal directly in the bytes.

    Returns list of bytes objects, each ≤ MAX_PATCH_BYTES bytes.
    """
    # Match optional leading whitespace + non-whitespace run
    words = re.findall(r'[ \t]*\S+', text)
    patches = []
    for w in words:
        b = w.encode('utf-8')[:MAX_PATCH_BYTES]  # truncate long words
        patches.append(b)
    return patches


def encode_patch(b_seq):
    """
    Encode bytes object as uint8 array of length MAX_PATCH_BYTES.
    Values: 0=pad, 1-256=bytes (byte_value + 1).
    Remaining positions after len(b_seq) are 0 (pad).
    """
    arr = np.zeros(MAX_PATCH_BYTES, dtype=np.uint8)
    for i, b in enumerate(b_seq):
        arr[i] = b + 1   # 0x00 → 1, 0xFF → 256; 0 reserved for pad
    return arr


# ─── Document processing ──────────────────────────────────────────────────────

# Doc-boundary marker: all-pad patch (all zeros) — signals document end
EOS_PATCH = np.zeros(MAX_PATCH_BYTES, dtype=np.uint8)


def process_docs(doc_list, split_name):
    """
    Process a list of documents into packed patch sequences.
    Returns list of arrays, each [NUM_PATCHES, MAX_PATCH_BYTES] uint8.
    """
    buffer    = []   # rolling patch buffer
    sequences = []
    total_patches = 0
    total_bytes   = 0
    truncated     = 0

    for di, doc in enumerate(doc_list):
        raw = split_patches(doc)
        for rp in raw:
            if len(rp) > MAX_PATCH_BYTES - 1:   # -1 for potential space prefix
                truncated += 1
            arr = encode_patch(rp)
            total_bytes += int((arr > 0).sum())
            buffer.append(arr)
        buffer.append(EOS_PATCH)   # doc separator

        # Drain complete sequences
        while len(buffer) >= NUM_PATCHES:
            seq = np.stack(buffer[:NUM_PATCHES], axis=0)
            sequences.append(seq)
            buffer = buffer[NUM_PATCHES:]
            total_patches += NUM_PATCHES

        if (di + 1) % 10_000 == 0:
            print(f'  [{split_name}] {di+1:,} docs → {len(sequences):,} sequences so far')

    avg_bytes = total_bytes / max(total_patches, 1)
    return sequences, avg_bytes, truncated


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    print('=' * 60)
    print('  Byte-Patch Preprocessor  (SpaceByte-style)')
    print(f'  MAX_PATCH_BYTES = {MAX_PATCH_BYTES}')
    print(f'  NUM_PATCHES     = {NUM_PATCHES} (= seq_len)')
    print(f'  Byte encoding   : 0=pad, 1-256=byte+1')
    print(f'  Output dir      : {OUTPUT_DIR}')
    print('=' * 60)

    # ── Load OpenWebText ───────────────────────────────────────────────────────
    print(f'\nLoading OpenWebText ({NUM_DOCS:,} docs, streaming)...')
    from datasets import load_dataset
    ds    = load_dataset('openwebtext', split='train', streaming=True)
    texts = []
    for i, item in enumerate(ds):
        if i >= NUM_DOCS: break
        texts.append(item['text'])
        if (i + 1) % 25_000 == 0:
            print(f'  {i+1:,} docs...')
    print(f'  {len(texts):,} docs loaded in {time.time()-t0:.1f}s')

    n = len(texts)
    splits = {
        'train': texts[:int(n * 0.95)],
        'val':   texts[int(n * 0.95): int(n * 0.95) + 2500],
        'test':  texts[int(n * 0.95) + 2500: int(n * 0.95) + 5000],
    }

    # ── Process each split ────────────────────────────────────────────────────
    meta = {
        'MAX_PATCH_BYTES': MAX_PATCH_BYTES,
        'NUM_PATCHES':     NUM_PATCHES,
        'byte_vocab_size': 257,
        'encoding':        '0=pad, 1-256=byte_value+1',
        'splits':          {},
    }

    for split_name, split_docs in splits.items():
        t1 = time.time()
        print(f'\nProcessing {split_name} ({len(split_docs):,} docs)...')
        seqs, avg_bytes, n_trunc = process_docs(split_docs, split_name)

        arr    = np.stack(seqs, axis=0)                      # [N, 2048, 12]
        tensor = torch.from_numpy(arr)                       # uint8
        path   = os.path.join(OUTPUT_DIR, f'byte_patches_{split_name}.pt')
        torch.save(tensor, path)

        elapsed = time.time() - t1
        print(f'  → {len(seqs):,} sequences ({len(seqs) * NUM_PATCHES:,} patches)')
        print(f'  → avg {avg_bytes:.2f} bytes/word (patch fill rate '
              f'{100*avg_bytes/MAX_PATCH_BYTES:.1f}%)')
        print(f'  → {n_trunc:,} words truncated to {MAX_PATCH_BYTES} bytes')
        print(f'  → saved {path}  ({arr.nbytes / 1e6:.1f} MB, {elapsed:.1f}s)')

        meta['splits'][split_name] = {
            'n_sequences':    len(seqs),
            'n_patches':      len(seqs) * NUM_PATCHES,
            'avg_bytes_word': round(avg_bytes, 3),
            'n_truncated':    n_trunc,
        }

    # ── Coverage stats ─────────────────────────────────────────────────────────
    print('\nByte coverage spot-check:')
    for sample in [' the', ' unprecedented', ' hello', " world!", ' uncharacteristically',
                   ' café', ' 日本語']:
        b = sample.encode('utf-8')
        tr = len(b) > MAX_PATCH_BYTES
        print(f'  {repr(sample):35s} {len(b):3d} bytes '
              f'{"→ TRUNCATED to "+str(MAX_PATCH_BYTES) if tr else "✓"}')

    meta['total_time_s'] = round(time.time() - t0, 1)
    meta_path = os.path.join(OUTPUT_DIR, 'byte_patches_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\nMeta → {meta_path}')
    print(f'Total time: {time.time()-t0:.1f}s')
    print('\nDone. Ready for train_2048_condByte.py')


if __name__ == '__main__':
    main()
