"""Standard plotting functions for sigstack."""

from __future__ import annotations

import collections.abc

import matplotlib
import numpy as np
import numpy.typing
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sigstack.plots.style import STYLE

__all__ = ["plot_band_energies", "plot_response", "plot_spectrogram"]


def plot_response(
    w: numpy.typing.ArrayLike,
    h: numpy.typing.ArrayLike,
    *,
    hz: bool = False,
    unwrap: bool = True,
    title: str | None = None,
    axes: collections.abc.Sequence[Axes] | None = None,
) -> Figure:
    """Plot frequency response magnitude and phase.

    Parameters
    ----------
    w : array_like
        Frequency axis (rad/sample or Hz).
    h : array_like
        Complex frequency response values. The magnitude is floored at 1e-12
        (-240 dB) so that zeros of the response stay finite; NaN values are drawn
        as gaps.
    hz : bool, optional
        If True, label x-axis as Hz. Default is False.
    unwrap : bool, optional
        If True, unwrap phase. Default is True.
    title : str, optional
        Title for the top subplot.
    axes : Sequence[Axes], optional
        Two Axes to draw into. If None, a new Figure is created.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plots.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if np.iscomplexobj(w):
        raise ValueError("w must be real")
    w_arr = np.asarray(w, dtype=np.float64)
    h_arr = np.asarray(h)

    if w_arr.ndim != 1 or not np.all(np.isfinite(w_arr)) or len(w_arr) < 2:
        raise ValueError("w must be a real 1-D finite array with length >= 2")
    if h_arr.ndim != 1 or len(h_arr) != len(w_arr):
        raise ValueError("h must be a 1-D array with same length as w")

    with matplotlib.rc_context(STYLE):  # artists read rcParams when they are created
        if axes is None:
            fig = Figure()
            top, bottom = fig.subplots(2, 1, sharex=True)
        else:
            top, bottom = axes
            root = top.get_figure(root=True)
            if not isinstance(root, Figure):
                raise ValueError("axes must belong to a Figure")
            fig = root

        top.plot(w_arr, 20 * np.log10(np.maximum(np.abs(h_arr), 1e-12)))
        top.set_ylabel("Magnitude (dB)")
        if title is not None:
            top.set_title(title)

        y = np.unwrap(np.angle(h_arr)) if unwrap else np.angle(h_arr)
        bottom.plot(w_arr, y)
        bottom.set_ylabel("Phase (rad)")
        bottom.set_xlabel("Frequency (Hz)" if hz else "Frequency (rad/sample)")

    return fig


def plot_spectrogram(
    t: numpy.typing.ArrayLike,
    f: numpy.typing.ArrayLike,
    s: numpy.typing.ArrayLike,
    *,
    db: bool = True,
    title: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot a spectrogram from precomputed power.

    Parameters
    ----------
    t : array_like
        Time axis (s).
    f : array_like
        Frequency axis (Hz).
    s : array_like
        Power values, shape (len(f), len(t)).
    db : bool, optional
        If True, plot 10*log10(power) in dB, with zero power floored at 1e-30
        (-300 dB). Default is True.
    title : str, optional
        Title for the plot.
    ax : Axes, optional
        Axis to draw into. If None, a new Figure is created.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if np.iscomplexobj(t) or np.iscomplexobj(f) or np.iscomplexobj(s):
        raise ValueError("t, f, s must be real")
    t_arr = np.asarray(t, dtype=np.float64)
    f_arr = np.asarray(f, dtype=np.float64)
    s_arr = np.asarray(s, dtype=np.float64)

    if (
        t_arr.ndim != 1
        or not np.all(np.isfinite(t_arr))
        or len(t_arr) < 2
        or not np.all(np.diff(t_arr) > 0)
    ):
        raise ValueError("t must be a real 1-D finite strictly increasing array with length >= 2")
    if (
        f_arr.ndim != 1
        or not np.all(np.isfinite(f_arr))
        or len(f_arr) < 2
        or not np.all(np.diff(f_arr) > 0)
    ):
        raise ValueError("f must be a real 1-D finite strictly increasing array with length >= 2")
    if s_arr.ndim != 2 or s_arr.shape != (len(f_arr), len(t_arr)):
        raise ValueError("s must have shape (len(f), len(t))")
    if not np.all(np.isfinite(s_arr)) or np.any(s_arr < 0):
        raise ValueError("s must be finite and non-negative")

    with matplotlib.rc_context(STYLE):  # artists read rcParams when they are created
        if ax is None:
            fig = Figure()
            ax = fig.add_subplot()
        else:
            root = ax.get_figure(root=True)
            if not isinstance(root, Figure):
                raise ValueError("ax must belong to a Figure")
            fig = root

        values = 10 * np.log10(np.maximum(s_arr, 1e-30)) if db else s_arr
        mesh = ax.pcolormesh(t_arr, f_arr, values, shading="nearest")
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Power (dB)" if db else "Power")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        if title is not None:
            ax.set_title(title)

    return fig


def plot_band_energies(
    edges: numpy.typing.ArrayLike,
    energies: numpy.typing.ArrayLike,
    *,
    db: bool = False,
    labels: collections.abc.Sequence[str] | None = None,
    title: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot band energies as a bar chart.

    Parameters
    ----------
    edges : array_like
        Band edges (Hz).
    energies : array_like
        Energy values for each band.
    db : bool, optional
        If True, plot 10*log10(energy) in dB, with zero energy floored at 1e-30
        (-300 dB). Default is False.
    labels : Sequence[str], optional
        Labels for the bands.
    title : str, optional
        Title for the plot.
    ax : Axes, optional
        Axis to draw into. If None, a new Figure is created.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if np.iscomplexobj(edges) or np.iscomplexobj(energies):
        raise ValueError("edges and energies must be real")
    edges_arr = np.asarray(edges, dtype=np.float64)
    energies_arr = np.asarray(energies, dtype=np.float64)

    if (
        edges_arr.ndim != 1
        or not np.all(np.isfinite(edges_arr))
        or len(edges_arr) < 2
        or not np.all(np.diff(edges_arr) > 0)
    ):
        raise ValueError("edges must be a real 1-D finite strictly increasing array")
    if (
        energies_arr.ndim != 1
        or not np.all(np.isfinite(energies_arr))
        or len(energies_arr) < 1
        or np.any(energies_arr < 0)
    ):
        raise ValueError("energies must be a real 1-D finite non-negative array with length >= 1")
    if len(edges_arr) != len(energies_arr) + 1:
        raise ValueError("len(edges) must equal len(energies) + 1")
    if labels is not None and len(labels) != len(energies_arr):
        raise ValueError("len(labels) must equal len(energies)")

    with matplotlib.rc_context(STYLE):  # artists read rcParams when they are created
        if ax is None:
            fig = Figure()
            ax = fig.add_subplot()
        else:
            root = ax.get_figure(root=True)
            if not isinstance(root, Figure):
                raise ValueError("ax must belong to a Figure")
            fig = root

        heights = 10 * np.log10(np.maximum(energies_arr, 1e-30)) if db else energies_arr
        ax.bar(edges_arr[:-1], heights, width=np.diff(edges_arr), align="edge")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Energy (dB)" if db else "Energy")
        if labels is not None:
            centers = edges_arr[:-1] + np.diff(edges_arr) / 2
            ax.set_xticks(centers)
            ax.set_xticklabels(labels)
        if title is not None:
            ax.set_title(title)

    return fig
