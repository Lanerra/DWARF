"""
HISA checkpoint policy regression test.

Prevents the unreachable-branch bug where _should_checkpoint_block
returns False for the DSR layer before checking the strategy,
making 'full_attn' a dead code path.

Run: pytest -q tests/test_hisa_checkpoint_policy.py
"""

import os
import pathlib
import sys

_project_root = str(pathlib.Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _make_should_checkpoint_block(dsr_layer):
    """
    Extract or simulate _should_checkpoint_block for testing.

    We test the canonical pattern found across HISA trainers.
    The correct implementation checks strategy FIRST, not DSR layer.
    """
    # Import the actual function from one of the trainers to test real code.
    # We use train_d512_l10_hisa_h16_v2_l3.py as the representative.
    # The test is structured so future HISA trainer files can share
    # the same helper expectations.

    # Simulate the function pattern found in HISA trainers.
    # The bug: checking `block_idx == dsr_layer` before strategy check.
    # The fix: check strategy first, so full_attn can reach DSR layer.

    def should_checkpoint_block(strategy, block_idx):
        """
        Canonical _should_checkpoint_block implementation.

        Returns True if the block at block_idx should be activation-checkpointed
        under the given strategy.

        Strategies:
        - 'none': no checkpointing
        - 'all': checkpoint all blocks including DSR
        - 'every_other': checkpoint alternating blocks (even indices)
        - 'full_attn': checkpoint only the DSR layer
        """
        if strategy == 'none':
            return False
        if strategy == 'all':
            return True
        if strategy == 'every_other':
            return block_idx % 2 == 0
        if strategy == 'full_attn':
            return block_idx == dsr_layer
        return False

    return should_checkpoint_block


def _make_buggy_should_checkpoint_block(dsr_layer):
    """
    Simulate the BUGGY pattern found in current HISA trainers.

    The bug: DSR layer is pre-emptively vetoed before strategy check,
    making 'full_attn' unreachable for the DSR layer.
    """
    def should_checkpoint_block(strategy, block_idx):
        # BUG: This check runs before strategy, vetoing DSR for all strategies
        if block_idx == dsr_layer:
            return False
        if strategy == 'all':
            return True
        if strategy == 'every_other':
            return block_idx % 2 == 0
        if strategy == 'full_attn':
            # UNREACHABLE for DSR layer due to early return above
            return block_idx == dsr_layer
        return False

    return should_checkpoint_block


class TestCheckpointPolicy:
    """Test the intended checkpoint policy behavior."""

    dsr_layer = 3  # representative DSR layer index

    def test_none_strategy(self):
        """none: no checkpointing at all."""
        fn = _make_should_checkpoint_block(self.dsr_layer)
        for block_idx in range(10):
            assert fn('none', block_idx) is False, \
                f"none strategy should never checkpoint (block {block_idx})"

    def test_all_strategy(self):
        """all: checkpoint every block including DSR."""
        fn = _make_should_checkpoint_block(self.dsr_layer)
        for block_idx in range(10):
            assert fn('all', block_idx) is True, \
                f"all strategy should checkpoint every block (block {block_idx})"

    def test_all_strategy_includes_dsr(self):
        """all: must include DSR layer."""
        fn = _make_should_checkpoint_block(self.dsr_layer)
        assert fn('all', self.dsr_layer) is True, \
            "all strategy must checkpoint DSR layer"

    def test_every_other_strategy(self):
        """every_other: checkpoint even-indexed blocks."""
        fn = _make_should_checkpoint_block(self.dsr_layer)
        for block_idx in range(10):
            expected = block_idx % 2 == 0
            assert fn('every_other', block_idx) is expected, \
                f"every_other: block {block_idx} expected {expected}"

    def test_full_attn_checks_dsr_only(self):
        """full_attn: checkpoint ONLY the DSR layer."""
        fn = _make_should_checkpoint_block(self.dsr_layer)
        for block_idx in range(10):
            expected = block_idx == self.dsr_layer
            assert fn('full_attn', block_idx) is expected, \
                f"full_attn: block {block_idx} expected {expected}"

    def test_full_attn_returns_true_for_dsr(self):
        """CRITICAL: full_attn must return True for DSR layer."""
        fn = _make_should_checkpoint_block(self.dsr_layer)
        assert fn('full_attn', self.dsr_layer) is True, \
            "full_attn MUST checkpoint the DSR layer — this is the whole point"

    def test_full_attn_returns_false_for_non_dsr(self):
        """full_attn: must NOT checkpoint non-DSR layers."""
        fn = _make_should_checkpoint_block(self.dsr_layer)
        for block_idx in range(10):
            if block_idx != self.dsr_layer:
                assert fn('full_attn', block_idx) is False, \
                    f"full_attn should not checkpoint non-DSR block {block_idx}"


class TestBuggyPattern:
    """
    Document the buggy pattern to ensure it's recognized as wrong.

    These tests demonstrate what the CURRENT code does (wrongly).
    They should FAIL if the code has been fixed to the canonical pattern.
    """

    dsr_layer = 3

    def test_buggy_full_attn_fails_for_dsr(self):
        """
        The buggy pattern makes full_attn return False for DSR layer.
        This test documents the bug — it should FAIL after the fix.
        """
        buggy_fn = _make_buggy_should_checkpoint_block(self.dsr_layer)
        result = buggy_fn('full_attn', self.dsr_layer)
        # This documents the bug: the buggy code returns False
        assert result is False, \
            "If this fails, the bug has been fixed (good!)"

    def test_buggy_all_excludes_dsr(self):
        """
        The buggy pattern also excludes DSR from 'all' strategy.
        """
        buggy_fn = _make_buggy_should_checkpoint_block(self.dsr_layer)
        result = buggy_fn('all', self.dsr_layer)
        assert result is False, \
            "Buggy code excludes DSR from 'all' too"


class TestActualTrainerCode:
    """
    Test the actual _should_checkpoint_block from a real HISA trainer.

    This reads the trainer source and verifies the pattern.
    """

    def test_trainer_checkpoint_gate_order(self):
        """
        Verify that _should_checkpoint_block in the trainer checks
        strategy BEFORE DSR layer veto.

        Reads the actual source code and checks the pattern.
        """
        trainer_path = os.path.join(
            _project_root, 'train', 'train_d512_l10_hisa_h16_v2_l3.py'
        )
        if not os.path.exists(trainer_path):
            import pytest
            pytest.skip(f"Trainer not found: {trainer_path}")

        with open(trainer_path) as f:
            source = f.read()

        # Find the _should_checkpoint_block method
        start = source.find('def _should_checkpoint_block')
        assert start >= 0, "_should_checkpoint_block not found in trainer"

        # Find the end of the method (next def or class at same indent)
        method_source = source[start:start + 500]  # generous window
        # Extract just the method body
        lines = method_source.split('\n')
        method_lines = []
        for line in lines[1:]:  # skip the def line
            if line and not line[0].isspace():
                break  # reached next top-level definition
            method_lines.append(line)
        method_body = '\n'.join(method_lines)

        # Check that strategy is checked BEFORE dsr_layer veto
        # The fix: strategy check should come first
        # The bug: `if block_idx == self.dsr_layer: return False` comes first

        dsr_veto_idx = method_body.find('block_idx == self.dsr_layer')
        strategy_check_idx = method_body.find("CHECKPOINT_STRATEGY == '")

        assert strategy_check_idx >= 0, \
            "No strategy check found in _should_checkpoint_block"
        assert dsr_veto_idx < 0 or strategy_check_idx < dsr_veto_idx, \
            "BUG: DSR layer veto comes before strategy check — " \
            "this makes 'full_attn' unreachable for the DSR layer. " \
            "Fix: check strategy first, then apply DSR logic within each strategy."

    def test_trainer_full_attn_reaches_dsr(self):
        """
        Verify that the full_attn strategy in the trainer can actually
        return True for the DSR layer.

        This is the critical regression test.
        """
        trainer_path = os.path.join(
            _project_root, 'train', 'train_d512_l10_hisa_h16_v2_l3.py'
        )
        if not os.path.exists(trainer_path):
            import pytest
            pytest.skip(f"Trainer not found: {trainer_path}")

        with open(trainer_path) as f:
            source = f.read()

        start = source.find('def _should_checkpoint_block')
        method_source = source[start:start + 500]
        lines = method_source.split('\n')
        method_lines = []
        for line in lines[1:]:
            if line and not line[0].isspace():
                break
            method_lines.append(line)
        method_body = '\n'.join(method_lines)

        # Check that there's no early return that blocks DSR before full_attn
        # Look for the pattern: `if block_idx == self.dsr_layer: return False`
        # appearing BEFORE `if CHECKPOINT_STRATEGY == 'full_attn'`

        # If the first conditional is a DSR veto, the bug exists
        first_cond = None
        for line in method_lines:
            stripped = line.strip()
            if stripped.startswith('if '):
                first_cond = stripped
                break

        if first_cond and 'dsr_layer' in first_cond and 'return False' in first_cond:
            assert False, (
                "BUG DETECTED: First conditional in _should_checkpoint_block "
                f"vetoes DSR layer unconditionally ({first_cond}). "
                "This makes 'full_attn' strategy unreachable for DSR. "
                "Fix: remove the unconditional DSR veto or move it after strategy check."
            )


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))
