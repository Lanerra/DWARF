"""
Standalone test evaluation for condM-v2.
Loads the trained checkpoint and runs test PPL + temperature sweep.
Usage:
  .venv/bin/python3 -u benchmarks/eval_condM_v2.py 2>&1 | tee benchmarks/results/condM_v2_eval.log
"""
import sys, os
# Import all architecture and utilities from the training script
sys.path.insert(0, os.path.dirname(__file__))
from train_2048_condM_v2 import (
    CondMV2Transformer, load_data, encode_split, evaluate, generate,
    GEN_PROMPTS, BATCH_SIZE, MAX_SEQ_LEN, EMBEDDING_DIM, NUM_LAYERS,
    NUM_HEADS, VOCAB_SIZE, INTERFERENCE, FULL_ATTN_LAYER
)
import math, torch
from transformers import AutoTokenizer

CHECKPOINT = 'benchmarks/checkpoints/2048_condM_v2_checkpoints/best.pt'
CONDM_TEST_PPL = 54.529  # condM baseline for comparison


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Load tokenizer
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Build model and load checkpoint
    print('Building model...')
    model = CondMV2Transformer(
        vocab_size=VOCAB_SIZE,
        D=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_seq_len=MAX_SEQ_LEN,
        interference=INTERFERENCE,
        full_attn_layer=FULL_ATTN_LAYER,
    ).to(device)

    print(f'Loading checkpoint: {CHECKPOINT}')
    ckpt = torch.load(CHECKPOINT, weights_only=False, map_location=device)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    print(f'  epoch={ckpt.get("epoch","?")}  val_ppl={ckpt.get("val_ppl",0):.3f}')

    # Load test data
    print('Loading OpenWebText (100K docs for same test split as training)...')
    data = load_data()
    test_seqs = encode_split(data['test'], tokenizer, MAX_SEQ_LEN, 'test')

    # Test PPL
    print('\nEvaluating test PPL...')
    test_loss = evaluate(model, test_seqs, BATCH_SIZE, device)
    test_ppl  = math.exp(min(test_loss, 20))
    delta     = test_ppl - CONDM_TEST_PPL
    sign      = '+' if delta >= 0 else ''
    print(f'\n  condM-v2 TEST: PPL {test_ppl:.3f} | Loss {test_loss:.4f}')
    print(f'  condM baseline: PPL {CONDM_TEST_PPL:.3f}')
    print(f'  Delta: {sign}{delta:.3f} PPL')

    # Temperature sweep
    print('\n── Temperature sweep ──')
    for temp in [0.0, 0.5, 0.7, 1.0]:
        label = 'greedy' if temp == 0.0 else f'T={temp}'
        print(f'\n[{label}]')
        gens = generate(model, tokenizer, GEN_PROMPTS, device, temperature=temp, top_p=0.9)
        for prompt, gen in zip(GEN_PROMPTS, gens):
            print(f'  {repr(prompt)} -> {repr(gen[:80])}')

    verdict = 'v2 worse — use condM for 85M' if delta > 1.0 else \
              ('v2 better — use condM-v2 for 85M' if delta < -1.0 else \
               'essentially tied — prefer condM for cleaner ablation')
    print('\n' + '='*60)
    print(f'RESULT: condM-v2 {test_ppl:.3f}  vs  condM {CONDM_TEST_PPL:.3f}  ({sign}{delta:.3f} PPL)')
    print(f'85M decision: {verdict}')
    print('='*60)


if __name__ == '__main__':
    main()
