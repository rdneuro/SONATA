# tests/test_spectral_config.py
"""Spectral descriptor bookkeeping + the fixed-k invariance path (no SpectralBrain).

These exercise the parts of the spectral module that do not need the heavy
`spectralbrain` library: descriptor length/name bookkeeping, and the size/k
guards that return an all-NaN descriptor (which now run *before* the heavy import).
"""

from __future__ import annotations

import numpy as np

from sonata.config import SonataConfig, SpectralConfig
from sonata.spectral_features import (
    descriptor_length,
    descriptor_names,
    surface_descriptor,
)


def test_config_instantiates_and_fixed_k_default_true():
    cfg = SonataConfig()
    assert cfg.spectral.fixed_k is True
    assert cfg.spectral.n_eigen >= 4


def test_descriptor_length_matches_names():
    cfg = SpectralConfig()
    assert descriptor_length(cfg) == len(descriptor_names(cfg))
    assert descriptor_length(cfg) > 0


def test_too_small_surface_is_all_nan_without_spectralbrain():
    # A surface below min_vertices must be flagged as all-NaN and must NOT need
    # the heavy spectral library (the guard runs before the import).
    cfg = SpectralConfig(min_vertices=60, n_eigen=80, fixed_k=True)
    verts = np.zeros((10, 3))          # far below min_vertices
    faces = np.array([[0, 1, 2]])
    out = surface_descriptor(verts, faces, cfg)
    assert out.shape == (descriptor_length(cfg),)
    assert np.all(np.isnan(out))


def test_fixed_k_unsupported_surface_is_all_nan():
    # Above min_vertices but below k+2 -> cannot support the common truncation.
    cfg = SpectralConfig(min_vertices=10, n_eigen=80, fixed_k=True)
    verts = np.zeros((30, 3))          # >= min_vertices but < n_eigen+2
    faces = np.array([[0, 1, 2]])
    out = surface_descriptor(verts, faces, cfg)
    assert np.all(np.isnan(out))
