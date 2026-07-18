# sonata/viz/style.py
"""Publication style, colourblind-safe palettes, and a uniform save helper.

Every SONATA figure shares one visual language so panels composited from
different modules look like one document. The palette is Okabe--Ito (safe for the
common colour-vision deficiencies); the style keeps axis furniture minimal and
type legible at print size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

#: Okabe--Ito colourblind-safe qualitative palette (8 hues).
OKABE_ITO: dict[str, str] = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#999999",
}

#: Ordered palette for categorical series (e.g. models, feature sets).
PALETTE: list[str] = [
    OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["green"],
    OKABE_ITO["red"], OKABE_ITO["purple"], OKABE_ITO["sky"],
    OKABE_ITO["yellow"], OKABE_ITO["grey"],
]

#: Perceptually-uniform sequential/diverging maps used for heatmaps.
SEQ_CMAP = "magma"
DIVERGING_CMAP = "RdBu_r"

_RC = {
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 1.4,
}


def set_pub_style() -> None:
    """Apply SONATA's publication rcParams to the active matplotlib session."""
    mpl.rcParams.update(_RC)


def color_cycle(n: int) -> list[str]:
    """Return ``n`` palette colours, cycling if ``n`` exceeds the palette."""
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


def save(fig, name: str, outdir: str | Path, *, formats: Sequence[str] = ("png", "pdf")) -> list[Path]:
    """Save ``fig`` under ``outdir`` as each requested format; return the paths.

    A single call yields both a raster (``.png``, for quick viewing) and a vector
    (``.pdf``, for the manuscript) by default, so downstream code never has to
    remember to emit both.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in formats:
        p = outdir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    return paths


__all__ = [
    "OKABE_ITO",
    "PALETTE",
    "SEQ_CMAP",
    "DIVERGING_CMAP",
    "set_pub_style",
    "color_cycle",
    "save",
    "plt",
]
