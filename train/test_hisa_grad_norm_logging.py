#!/usr/bin/env python3
"""
Focused regression test for grad_norm logging in the active HISA trainer.

Goal: the displayed grad norm must be computed from live gradients before
optimizer.zero_grad(set_to_none=True) clears them.
"""

import importlib.util
from pathlib import Path

import torch

TRAINER_PATH = Path('/home/dlewis3/Desktop/AI/DWARF/train/train_d512_l20_hisa_hd32.py')

spec = importlib.util.spec_from_file_location('train_d512_l20_hisa_hd32', TRAINER_PATH)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def test_grad_norm_helper_reads_live_grads_before_zero_grad():
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    x = torch.randn(3, 4)
    y = torch.randn(3, 2)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()

    grad_norm_before = trainer._grad_norm_from_parameters(model.parameters())
    assert grad_norm_before > 0.0

    optimizer.zero_grad(set_to_none=True)
    grad_norm_after = trainer._grad_norm_from_parameters(model.parameters())
    assert grad_norm_after == 0.0


if __name__ == '__main__':
    test_grad_norm_helper_reads_live_grads_before_zero_grad()
    print('ok')
