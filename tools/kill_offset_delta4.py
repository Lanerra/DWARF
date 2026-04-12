"""
kill_offset_delta4.py

Creates a modified copy of a d768_l32 checkpoint with delta=4 (offset index 3)
zeroed out across all DSQG layers. Zeroing scale_embed[3] collapses its gating
contribution to ~0; zeroing pos_bias[3] removes its positional bias.
This replicates the relay_path_tracer ablation hook as a permanent checkpoint edit.

Usage:
  python3 tools/kill_offset_delta4.py \
    --input  autoresearch/checkpoints/d768_l32_mixed_scratch_best.pt \
    --output autoresearch/checkpoints/d768_l32_mixed_scratch_no_d4.pt
"""

import argparse, torch

OFFSETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024]
KILL_DELTA = 4
KILL_IDX   = OFFSETS.index(KILL_DELTA)   # 3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    print(f'Loading {args.input} ...')
    ck = torch.load(args.input, map_location='cpu')

    # Determine if it's a raw state dict or wrapped checkpoint
    if 'model_state_dict' in ck:
        state = ck['model_state_dict']
        wrapped = True
    elif 'embedding.weight' in ck:
        state = ck
        wrapped = False
    else:
        # try generic model key
        state = ck.get('model', ck)
        wrapped = 'model' in ck

    modified = 0
    skipped_fa = 0
    for key, tensor in state.items():
        # Only touch DSQG attn blocks — skip the full_attn (FA) layer
        if 'attn.scale_embed' in key:
            # shape: [J=24, HD=64]
            assert tensor.shape[0] == len(OFFSETS), \
                f"Unexpected J={tensor.shape[0]} in {key}, expected {len(OFFSETS)}"
            # Check if this is the FA layer (full_attn_layer=8 → blocks.8)
            block_num = int(key.split('.')[1])
            if block_num == 8:
                skipped_fa += 1
                continue
            old_norm = tensor[KILL_IDX].norm().item()
            state[key][KILL_IDX] = torch.zeros_like(tensor[KILL_IDX])
            print(f'  {key}[{KILL_IDX}] scale_embed zeroed  (was norm={old_norm:.4f})')
            modified += 1

        elif 'attn.pos_bias' in key:
            # shape: [J=24, H=12]
            assert tensor.shape[0] == len(OFFSETS), \
                f"Unexpected J={tensor.shape[0]} in {key}, expected {len(OFFSETS)}"
            block_num = int(key.split('.')[1])
            if block_num == 8:
                skipped_fa += 1
                continue
            old_val = tensor[KILL_IDX].mean().item()
            state[key][KILL_IDX] = torch.zeros_like(tensor[KILL_IDX])
            print(f'  {key}[{KILL_IDX}] pos_bias  zeroed  (was mean={old_val:.4f})')
            modified += 1

    print(f'\nModified {modified} tensors across DSQG layers (skipped {skipped_fa} FA-layer entries)')
    print(f'Saving to {args.output} ...')

    if wrapped and 'model_state_dict' in ck:
        ck['model_state_dict'] = state
        torch.save(ck, args.output)
    elif wrapped:
        ck['model'] = state
        torch.save(ck, args.output)
    else:
        torch.save(state, args.output)

    print('Done.')

if __name__ == '__main__':
    main()
