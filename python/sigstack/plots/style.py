"""Shared plot style and byte-identical figure saving."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.typing import RcKeyType

__all__ = ["STYLE", "save_figure"]

STYLE: dict[RcKeyType, Any] = {
    "figure.figsize": (6.4, 4.0),
    "figure.dpi": 100.0,
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.0,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.2,
    "image.cmap": "viridis",
    "svg.hashsalt": "sigstack",
}

_FORMATS: dict[str, str] = {".png": "png", ".svg": "svg", ".pdf": "pdf"}


def save_figure(
    fig: Figure,
    path: str | Path,
    *,
    dpi: int | str | None = None,
) -> None:
    """Save a figure to disk with deterministic metadata.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    path : str or pathlib.Path
        Destination file path. The suffix determines the format.
    dpi : int or str, optional
        Resolution for raster formats. Defaults to ``"figure"``.

    Raises
    ------
    ValueError
        If the file suffix is not one of ``.png``, ``.svg``, ``.pdf``.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix not in _FORMATS:
        raise ValueError(f"Unsupported format: {suffix!r}")

    fmt = _FORMATS[suffix]

    metadata: dict[str, Any]
    if fmt == "png":
        metadata = {"Software": None}
    elif fmt == "svg":
        metadata = {"Date": None, "Creator": None}
    else:  # pdf
        metadata = {
            "CreationDate": None,
            "ModDate": None,
            "Producer": None,
            "Creator": None,
        }

    with mpl.rc_context({"svg.hashsalt": "sigstack"}):
        fig.savefig(p, format=fmt, dpi=dpi or "figure", metadata=metadata)
