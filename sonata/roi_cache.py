# sonata/roi_cache.py
"""Per-ROI spectral-descriptor cache (disk + memory) — SONATA step [4].

Requirement (user, verbatim): once a ROI present in the compiled set [4b] has
had its metrics computed, if it appears again in another tract of the SAME
individual, the cached value is reused without recomputing.

We cache at TWO levels:
  - memory: a per-process dict keyed by (subject_id, roi_id) — reused across all
    tracts within a run (the common case: a ROI is an endpoint of several tracts).
  - disk:   cache/rois/{sid}/{roi_id}.npy — reused across runs, so re-running the
    pipeline never recomputes a ROI descriptor already on disk.

The cache stores the fixed-length spectral descriptor vector (see
spectral_features.surface_descriptor). Node morphometry is cheap and stays in
the parcellation object; only the expensive Laplace–Beltrami descriptor is cached.

Cache key includes a CONFIG HASH so that changing spectral settings (n_eigen,
descriptors, smoothing) invalidates stale vectors instead of silently reusing
them — a correctness guard the user's workflow needs when iterating on settings.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import SpectralConfig
from .spectral_features import descriptor_length, surface_descriptor
from .utils import get_logger

log = get_logger("sonata.roicache")


def spectral_config_hash(cfg: SpectralConfig) -> str:
    """Short stable hash of the spectral settings that affect the descriptor."""
    payload = json.dumps({
        "n_eigen": cfg.n_eigen, "laplacian": cfg.laplacian,
        "hks_n_times": cfg.hks_n_times, "wks_n_energies": cfg.wks_n_energies,
        "shapedna_normalize": cfg.shapedna_normalize,
        "use_descriptors": list(cfg.use_descriptors),
        "taubin_iter": cfg.taubin_iter, "taubin_lambda": cfg.taubin_lambda,
        "taubin_mu": cfg.taubin_mu, "min_vertices": cfg.min_vertices,
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


class ROIDescriptorCache:
    """Two-level (memory + disk) cache of per-ROI spectral descriptors.

    Usage
    -----
    >>> cache = ROIDescriptorCache(output_dir, sid, cfg.spectral)
    >>> vec = cache.get_or_compute(roi_id, vertices, faces)   # (D,) float

    The same `cache` is passed through the whole subject build so every tract
    that touches `roi_id` reuses the first computation.
    """

    def __init__(self, output_dir: Path, subject_id: str, cfg: SpectralConfig):
        self.cfg = cfg
        self.subject_id = str(subject_id)
        self.cfg_hash = spectral_config_hash(cfg)
        self.D = descriptor_length(cfg)
        self.mem: dict[int, np.ndarray] = {}
        self.disk_dir = Path(output_dir) / "cache" / "rois" / self.subject_id
        self.disk_dir.mkdir(parents=True, exist_ok=True)
        self.n_hits_mem = 0
        self.n_hits_disk = 0
        self.n_compute = 0

    def _disk_path(self, roi_id: int) -> Path:
        # config hash in the filename → settings change invalidates the cache
        return self.disk_dir / f"roi{int(roi_id):03d}_{self.cfg_hash}.npy"

    def get_or_compute(self, roi_id: int, vertices, faces) -> np.ndarray:
        roi_id = int(roi_id)
        # 1) memory
        v = self.mem.get(roi_id)
        if v is not None:
            self.n_hits_mem += 1
            return v
        # 2) disk
        p = self._disk_path(roi_id)
        if p.exists():
            try:
                v = np.load(p)
                if v.shape == (self.D,):
                    self.mem[roi_id] = v
                    self.n_hits_disk += 1
                    return v
            except Exception:
                pass  # corrupt cache file → recompute
        # 3) compute
        v = surface_descriptor(vertices, faces, self.cfg).astype(np.float32)
        self.n_compute += 1
        self.mem[roi_id] = v
        try:
            np.save(p, v)
        except Exception as exc:  # pragma: no cover
            log.warning("could not persist ROI %d descriptor: %r", roi_id, exc)
        return v

    def stats(self) -> dict:
        return {"mem_hits": self.n_hits_mem, "disk_hits": self.n_hits_disk,
                "computed": self.n_compute}
