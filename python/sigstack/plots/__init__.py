"""Plotting subpackage for sigstack."""

from __future__ import annotations

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "sigstack.plots needs matplotlib; install it with: pip install 'sigstack[plot]'"
    ) from exc

from sigstack.plots.figures import plot_band_energies, plot_response, plot_spectrogram
from sigstack.plots.style import STYLE, save_figure

__all__: list[str] = [
    "STYLE",
    "plot_band_energies",
    "plot_response",
    "plot_spectrogram",
    "save_figure",
]
