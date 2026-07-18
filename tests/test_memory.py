# tests/test_memory.py
"""RAM/VRAM budgeting, the batch gate, and OOM-safe batching."""

from __future__ import annotations

import pytest

from sonata import memory as M


def test_available_ram_positive():
    assert M.available_ram_bytes() > 0


def test_available_vram_none_without_cuda():
    from sonata.backends.base import capabilities

    if not capabilities().has_cuda:
        assert M.available_vram_bytes("torch") is None


def test_estimate_batch_size_respects_bounds():
    n = M.estimate_batch_size(item_bytes=8 * 1024**2, backend="cpu",
                              min_batch=2, max_batch=16)
    assert 2 <= n <= 16


def test_memory_gate_slot_acquires_and_releases():
    gate = M.MemoryGate(max_concurrent=2)
    with gate.slot():
        pass
    with gate:  # both context forms work
        pass
    with pytest.raises(ValueError):
        M.MemoryGate(max_concurrent=0)


def test_run_in_oom_safe_batches_preserves_order():
    out = M.run_in_oom_safe_batches(lambda chunk: [c + 1 for c in chunk],
                                    list(range(10)), initial_batch=3, backend="cpu")
    assert out == list(range(1, 11))


def test_run_in_oom_safe_batches_shrinks_on_oom():
    calls = {"n": 0}

    def flaky(chunk):
        # OOM once on the first (largest) batch, then succeed on halves.
        calls["n"] += 1
        if calls["n"] == 1 and len(chunk) > 2:
            raise MemoryError("simulated OOM")
        return [c * 2 for c in chunk]

    out = M.run_in_oom_safe_batches(flaky, list(range(8)), initial_batch=8,
                                    backend="cpu", min_batch=1)
    assert out == [c * 2 for c in range(8)]


def test_sync_every_yields_all():
    assert list(M.sync_every(range(9), every=4, backend="cpu")) == list(range(9))
