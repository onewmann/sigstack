r"""Design linear-phase FIR filters by the window method.

The impulse response is a symmetric or anti-symmetric sequence obtained by
multiplying an ideal (infinite) frequency response with a finite-length window.
For a filter of length :math:`N` and normalised cutoffs
:math:`0 < c_1 < \dots < c_M < 1`, the bands alternate between pass and stop,
starting with a passband if ``pass_zero`` is true.

The ideal response for a passband :math:`[a, b]` (normalised to the Nyquist
frequency) contributes

.. math::

   h_{\text{ideal}}[k] \;=\; b\,\mathrm{sinc}(b\,m_k) - a\,\mathrm{sinc}(a\,m_k),
   \qquad m_k = k - \tfrac{N-1}{2},

where :math:`\mathrm{sinc}(x) = \sin(\pi x)/(\pi x)` with :math:`\mathrm{sinc}(0)=1`.
The final impulse response is the sum over all passbands, multiplied by the
window. With ``scale=True`` the result is divided by its gain at the reference
frequency of the first passband so that the magnitude response equals one there.

References
----------
.. [OS2010] A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*,
   3rd ed., Pearson, 2010, sec. 7.5 (FIR design by windowing), 7.5.3 (Kaiser window method).
"""

from __future__ import annotations

import math
import numbers
import operator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sigstack.core.windows import get_window

__all__ = ["firwin"]


def _check_numtaps(numtaps: object) -> int:
    """Return ``numtaps`` as an int after checking that it is a positive integer."""
    if isinstance(numtaps, bool | np.bool_):
        raise TypeError("numtaps must be an integer, got bool")
    try:
        n = operator.index(numtaps)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(f"numtaps must be an integer, got {type(numtaps).__name__}") from None
    if n <= 0:
        raise ValueError(f"numtaps must be > 0, got {n}")
    return n


def _check_fs(fs: object) -> float:
    """Return ``fs`` as a positive finite float."""
    if isinstance(fs, bool) or not isinstance(fs, numbers.Real):
        raise TypeError(f"fs must be a real number, got {type(fs).__name__}")
    value = float(fs)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"fs must be finite and > 0, got {fs!r}")
    return value


def _normalise_cutoffs(
    cutoff: float | list[float] | tuple[float, ...] | ArrayLike, fs: float
) -> NDArray[np.float64]:
    """Return the cutoffs as a strictly increasing array in (0, 1)."""
    if isinstance(cutoff, (list, tuple)):
        arr = np.asarray(cutoff, dtype=np.float64)
    elif isinstance(cutoff, np.ndarray):
        arr = cutoff.astype(np.float64)
    else:
        arr = np.asarray([cutoff], dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("cutoff must be a non-empty scalar or 1-D sequence")
    if not np.all(np.isfinite(arr)):
        raise ValueError("cutoff values must be finite")
    norm = arr / (fs / 2.0)
    if np.any(norm <= 0.0) or np.any(norm >= 1.0):
        raise ValueError("all cutoffs must satisfy 0 < c < fs/2")
    if np.any(np.diff(norm) <= 0.0):
        raise ValueError("cutoffs must be strictly increasing")
    return norm


def _passbands(cutoffs: NDArray[np.float64], pass_zero: bool) -> list[tuple[float, float]]:
    """Return the list of passband intervals [a, b] in normalised frequency."""
    bands: list[tuple[float, float]] = []
    edges = np.concatenate([[0.0], cutoffs, [1.0]])
    for i in range(len(edges) - 1):
        is_pass = (i % 2 == 0) if pass_zero else (i % 2 == 1)
        if is_pass:
            bands.append((float(edges[i]), float(edges[i + 1])))
    return bands


def firwin(
    numtaps: int,
    cutoff: float | list[float] | tuple[float, ...] | ArrayLike,
    window: str = "hamming",
    *,
    pass_zero: bool = True,
    scale: bool = True,
    fs: float = 2.0,
    **kwargs: Any,
) -> NDArray[np.float64]:
    r"""Design a linear-phase FIR filter by the window method.

    The impulse response is the sum of ideal sinc responses for each passband,
    multiplied by a symmetric window and optionally scaled to unit gain at the
    reference frequency of the first passband.

    Parameters
    ----------
    numtaps : int
        Filter length :math:`N \ge 1`.
    cutoff : float or array_like
        Cutoff frequencies in Hz (if ``fs`` is given) or as a fraction of the
        Nyquist frequency. A scalar gives a lowpass/highpass; a sequence of
        strictly increasing values gives bandpass/bandstop/multiband filters.
        Every value must satisfy :math:`0 < c < f_s/2`.
    window : str or tuple, default 'hamming'
        Window name (``"rectangular"``, ``"hann"``, ``"hamming"``, ``"blackman"``,
        ``"blackman_harris"``, ``"flattop"``, ``"kaiser"``) or a tuple
        ``(name, param)`` for parameterised windows. Extra parameters are passed
        via ``**kwargs``.
    pass_zero : bool, default True
        If true the band starting at 0 is a passband; otherwise it is a stopband.
    scale : bool, default True
        If true, scale the impulse response so that its magnitude equals one at
        the reference frequency of the first passband (DC if the filter starts
        with a passband at 0, Nyquist if it ends there, otherwise the midpoint).
    fs : float, default 2.0
        Sampling frequency in Hz. Cutoffs are normalised by ``fs / 2``.
    **kwargs
        Additional parameters passed to :func:`sigstack.core.windows.get_window`
        (e.g. ``beta`` for the Kaiser window).

    Returns
    -------
    numpy.ndarray
        FIR filter coefficients of shape ``(numtaps,)`` and dtype ``float64``.

    Raises
    ------
    TypeError
        If ``numtaps`` is not an integer, ``fs`` is not a real number, or the
        window name is not a string.
    ValueError
        If ``numtaps <= 0``, ``fs <= 0``, cutoffs are empty, not strictly
        increasing, outside (0, fs/2), the window is unknown, or an even-length
        filter would have a passband ending at Nyquist.

    Examples
    --------
    >>> import numpy as np
    >>> h = firwin(31, 0.3)
    >>> h.shape
    (31,)
    """
    n = _check_numtaps(numtaps)
    sampling_rate = _check_fs(fs)
    cutoffs = _normalise_cutoffs(cutoff, sampling_rate)
    bands = _passbands(cutoffs, pass_zero)

    # Even-length filters cannot have a passband ending at Nyquist (b == 1.0)
    if n % 2 == 0 and any(b == 1.0 for _, b in bands):
        raise ValueError(f"even numtaps ({n}) with a passband ending at Nyquist is not supported")

    # Build the ideal impulse response
    k = np.arange(n, dtype=np.float64)
    m = k - (n - 1.0) / 2.0
    h_ideal = np.zeros(n, dtype=np.float64)
    for a, b in bands:
        h_ideal += b * np.sinc(b * m) - a * np.sinc(a * m)

    # Apply the symmetric window
    w = get_window(window, n, sym=True, **kwargs)
    h = h_ideal * w

    # Scale to unit gain at the reference frequency of the first passband
    if scale and bands:
        a0, b0 = bands[0]
        if a0 == 0.0:
            f_ref = 0.0
        elif b0 == 1.0:
            f_ref = 1.0
        else:
            f_ref = (a0 + b0) / 2.0
        s = float(np.sum(h * np.cos(np.pi * f_ref * m)))
        if s != 0.0:
            h = h / s

    return h
