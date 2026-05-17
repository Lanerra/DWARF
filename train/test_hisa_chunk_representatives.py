#!/usr/bin/env python3
"""
Focused regression tests for HISA chunk representative construction.

Goal: preserve distinctive token signals for Stage-1 routing without changing the
Triton interface. The representative should be an actual token vector selected
from each chunk, not a mean-pooled blend that dilutes a standout key.
"""

import sys
import torch

sys.path.insert(0, '/home/dlewis3/Desktop/AI/DWARF/kernels')

from hierarchical_sparse_attn_v15_hisa import _compute_chunk_representatives


def test_distinctive_token_is_preserved_by_chunk_representative():
    # One chunk with a single standout token and several distractors that would
    # dilute the signal under mean pooling.
    K_pad = torch.tensor(
        [[[ 
            [9.0, -7.0, 5.0],
            [0.2, 0.1, 0.0],
            [-0.1, 0.2, 0.1],
            [0.0, -0.2, 0.1],
        ]]],
        dtype=torch.float32,
    )

    chunk_reps = _compute_chunk_representatives(K_pad, num_chunks=1)
    expected = K_pad[:, :, 0:1, :]

    assert chunk_reps.shape == (1, 1, 1, 3)
    assert torch.allclose(chunk_reps, expected)


def test_padding_does_not_override_last_real_token():
    # Last chunk has one real token plus padded zeros. The representative should
    # still be the real token rather than a padding vector.
    K_pad = torch.tensor(
        [[[ 
            [0.0, 0.0],
            [0.0, 0.0],
            [0.3, -0.4],
            [0.0, 0.0],
        ]]],
        dtype=torch.float32,
    )

    chunk_reps = _compute_chunk_representatives(K_pad, num_chunks=1)
    expected = torch.tensor([[[[0.3, -0.4]]]], dtype=torch.float32)

    assert torch.allclose(chunk_reps, expected)


if __name__ == '__main__':
    test_distinctive_token_is_preserved_by_chunk_representative()
    test_padding_does_not_override_last_real_token()
    print('ok')
