# sonata/__init__.py
"""SONATA — Structure-tO-fuNction via spectrAl Tract Attributes.

A geometry-aware, edge-conditioned graph neural network that predicts functional
connectivity from structural connectivity using Laplace--Beltrami spectral shape
descriptors of white-matter tract surfaces (edge features) and of cortical ROI
surfaces (node features), benchmarked against classical diffusion scalars
(Aim 2 non-inferiority) and communication/gradient baselines.

Lazy import policy
------------------
The compute infrastructure (:mod:`sonata.backends`, :mod:`sonata.parallel`,
:mod:`sonata.memory`, :mod:`sonata.viz`) and configuration import with only the
scientific-Python core present, so ``import sonata`` never fails because a GPU
stack, ``spectralbrain`` or a neuroimaging I/O library is missing. The heavy
pipeline symbols (graph build, model, training, baselines) are resolved lazily on
first access via :pep:`562`, importing their dependencies only when actually used.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"

# ── always-available: config (light: dataclasses only) ────────────────────────
from .config import (  # noqa: E402
    FuncConfig,
    ModelConfig,
    NonInferiorityConfig,
    Paths,
    SonataConfig,
    SpectralConfig,
    TractConfig,
    TrainConfig,
)

#: Map ``public_name -> (submodule, attribute)`` for lazily-loaded heavy symbols.
_LAZY: dict[str, tuple[str, str]] = {
    "build_subject_features": (".graph", "build_subject_features"),
    "to_pyg_data": (".graph", "to_pyg_data"),
    "Sonata": (".model", "Sonata"),
    "build_model": (".model", "build_model"),
    "run_cv": (".train", "run_cv"),
    "run_all_baselines": (".baselines", "run_all_baselines"),
    "compare_feature_sets": (".noninferiority", "compare_feature_sets"),
    "paired_bootstrap_tost": (".noninferiority", "paired_bootstrap_tost"),
    "fit_gradients": (".gradients", "fit_gradients"),
    "persist_subject_spectra": (".persistence", "persist_subject_spectra"),
    "fit_attribution": (".attribution", "fit_attribution"),
    "build_horseshoe_model": (".attribution", "build_horseshoe_model"),
    "save_table": (".reporting", "save_table"),
    "summary_comparison": (".reporting", "summary_comparison"),
    "get_logger": (".utils", "get_logger"),
    "load_manifest": (".utils", "load_manifest"),
    "seed_everything": (".utils", "seed_everything"),
}


def __getattr__(name: str) -> Any:  # PEP 562 module-level lazy attribute access
    """Resolve heavy pipeline symbols on first access, importing deps then."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'sonata' has no attribute {name!r}")
    submodule, attr = target
    mod = importlib.import_module(submodule, __name__)
    value = getattr(mod, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


if TYPE_CHECKING:
    from .baselines import run_all_baselines
    from .graph import build_subject_features, to_pyg_data
    from .gradients import fit_gradients
    from .model import Sonata, build_model
    from .noninferiority import compare_feature_sets, paired_bootstrap_tost
    from .persistence import persist_subject_spectra
    from .reporting import save_table, summary_comparison
    from .train import run_cv
    from .utils import get_logger, load_manifest, seed_everything

__all__ = [
    "SonataConfig", "Paths", "SpectralConfig", "TractConfig", "FuncConfig",
    "ModelConfig", "TrainConfig", "NonInferiorityConfig",
    *_LAZY.keys(),
    "__version__",
]
