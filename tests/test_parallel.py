# tests/test_parallel.py
"""The n_threads convention, BLAS pinning, and serial/parallel equivalence."""

from __future__ import annotations

import os

from sonata import parallel as P


def _square(x):  # top-level so it is picklable by joblib
    return x * x


def test_serial_and_parallel_agree():
    items = list(range(25))
    expected = [x * x for x in items]
    assert P.parallel_map(_square, items, n_threads=1) == expected
    # n_threads>=2 clamps to available cores but must give identical results
    assert P.parallel_map(_square, items, n_threads=4) == expected


def test_effective_workers_bounds():
    assert P.effective_workers(1) == 1
    assert P.effective_workers(-1) >= 1
    assert P.effective_workers(10_000) == P.effective_workers(-1)


def test_pin_blas_threads_restores_environment():
    before = os.environ.get("OMP_NUM_THREADS")
    with P.pin_blas_threads(1):
        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ.get("OMP_NUM_THREADS") == before


def test_empty_input_returns_empty():
    assert P.parallel_map(_square, [], n_threads=4) == []


def test_parallel_without_blas_pinning_is_picklable():
    # Regression: pin_blas=False must use a module-level worker (not a lambda),
    # so loky can pickle it in the parallel path.
    items = list(range(12))
    got = P.parallel_map(_square, items, n_threads=4, pin_blas=False)
    assert got == [x * x for x in items]
