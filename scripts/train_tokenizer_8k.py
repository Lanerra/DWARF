#!/usr/bin/env python3
"""Train an 8K BPE tokenizer on the 100K FineWeb-Edu doc cache."""
import json, time
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

print("Loading doc cache...")
t0 = time.time()
with open('benchmarks/logs/condm_fineweb_edu_doc_cache.json') as f:
    docs = json.load(f)
print(f"Loaded {len(docs):,} docs in {time.time()-t0:.1f}s")

tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=8000,
    min_frequency=2,
    special_tokens=["<|endoftext|>", "<|unk|>", "<|pad|>"],
    show_progress=True,
)

print("Training 8K BPE tokenizer...")
t1 = time.time()
tokenizer.train_from_iterator(iter(docs), trainer=trainer)
print(f"Trained in {time.time()-t1:.1f}s")

# Add post-processor for EOS
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

out = 'results/fineweb_tokenizer_8k.json'
tokenizer.save(out)
print(f"Saved to {out}")
print(f"Vocab size: {tokenizer.get_vocab_size()}")
# Verify EOS token
vocab = tokenizer.get_vocab()
print(f"EOS token id: {vocab.get('<|endoftext|>', 'NOT FOUND')}")
