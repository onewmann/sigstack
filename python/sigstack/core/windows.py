r"""Window functions for spectral analysis and FIR design.

All windows are computed from their closed-form definitions in ``float64``.

Symmetric and periodic windows
------------------------------
A symmetric window of length :math:`N` is defined on :math:`k = 0, \dots, N-1`
with :math:`L = N - 1`, so that :math:`w[k] = w[N-1-k]`. Use it for FIR filter
design (``sym=True``).

A periodic (DFT-even) window of length :math:`N` is the symmetric window of length
:math:`N + 1` without its last sample, so that :math:`w[k] = w[N-k]` for
:math:`k = 1, \dots, N-1`. Harris (1978) explains why DFT-based spectral analysis needs this form;
it is the default (``sym=False``).

A window of length 1 is ``[1.0]`` in both cases.

References
----------
.. [Harris1978] F. J. Harris, "On the use of windows for harmonic analysis with
   the discrete Fourier transform", *Proc. IEEE* 66(1), 51-83, 1978.
   doi:10.1109/PROC.1978.10837
.. [OS2010] A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*,
   3rd ed., Pearson, 2010, sec. 7.5.
"""

from __future__ import annotations

import math
import numbers
import operator
from collections.abc import Callable
from typing import Any, SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "blackman",
    "blackman_harris",
    "coherent_gain",
    "enbw",
    "flattop",
    "general_cosine",
    "get_window",
    "hamming",
    "hann",
    "kaiser",
    "rectangular",
]


def _check_length(n: object) -> int:
    """Return ``n`` as an int after checking that it is an integer >= 1."""
    if isinstance(n, bool | np.bool_):
        raise TypeError("n must be an integer, got bool")
    try:
        length = operator.index(n)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(f"n must be an integer, got {type(n).__name__}") from None
    if length < 1:
        raise ValueError(f"n must be >= 1, got {length}")
    return length


def _check_beta(beta: object) -> float:
    """Return ``beta`` as a float after checking that it is finite and >= 0."""
    if isinstance(beta, bool) or not isinstance(beta, numbers.Real):
        raise TypeError(f"beta must be a real number, got {type(beta).__name__}")
    value = float(beta)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"beta must be finite and >= 0, got {beta!r}")
    return value


def _check_sym(sym: object) -> bool:
    """Return ``sym`` after checking that it is a bool."""
    if not isinstance(sym, bool | np.bool_):
        raise TypeError(f"sym must be a bool, got {type(sym).__name__}")
    return bool(sym)


def _symmetric_length(n: int, sym: bool) -> int:
    """Length of the symmetric window that the requested window is cut from."""
    return n if sym else n + 1


def _mirror(first_half: NDArray[np.float64], m: int) -> NDArray[np.float64]:
    """Build a symmetric window of length ``m`` from its first ``ceil(m / 2)`` samples.

    Mirroring makes the window exactly symmetric, independent of rounding.
    """
    return np.concatenate([first_half, first_half[: m // 2][::-1]])


def _finish(symmetric: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Cut the symmetric window to length ``n``; drops the last sample of a periodic window."""
    return symmetric[:n]


def general_cosine(n: SupportsIndex, a: ArrayLike, *, sym: bool = False) -> NDArray[np.float64]:
    r"""Generalised cosine window (sum of cosines).

    .. math::

       w[k] = \sum_{m=0}^{M-1} (-1)^m \, a_m \cos\!\left(\frac{2\pi m k}{L}\right),
       \qquad k = 0, \dots, L,

    with :math:`L = N - 1` for the symmetric window. The periodic window uses
    :math:`L = N` and keeps :math:`k = 0, \dots, N-1`. Hann, Hamming, Blackman,
    Blackman-Harris and flat-top are special cases ([Harris1978]_ sec. V).

    Parameters
    ----------
    n : int
        Window length :math:`N \ge 1`.
    a : array_like
        Coefficients :math:`a_0, \dots, a_{M-1}`: a non-empty 1-D sequence of
        finite real numbers.
    sym : bool, default False
        ``True`` for the symmetric window, ``False`` for the periodic (DFT-even) one.

    Returns
    -------
    numpy.ndarray
        Window of shape ``(n,)`` and dtype ``float64``.

    Raises
    ------
    TypeError
        If ``n`` is not an integer or ``sym`` is not a bool.
    ValueError
        If ``n < 1`` or ``a`` is empty, not 1-D or not finite.
    """
    length = _check_length(n)
    is_sym = _check_sym(sym)
    coeffs = np.asarray(a, dtype=np.float64)
    if coeffs.ndim != 1 or coeffs.size == 0:
        raise ValueError(f"a must be a non-empty 1-D sequence, got shape {coeffs.shape}")
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("a must contain only finite values")
    if length == 1:
        return np.ones(1)
    m = _symmetric_length(length, is_sym)
    k = np.arange((m + 1) // 2, dtype=np.float64)
    signs = (-1.0) ** np.arange(coeffs.size)
    phase = 2.0 * np.pi * k / (m - 1)
    half = np.zeros_like(k)
    for order, (sign, coeff) in enumerate(zip(signs, coeffs, strict=True)):
        half += sign * coeff * np.cos(order * phase)
    return _finish(_mirror(half, m), length)


def rectangular(n: SupportsIndex, *, sym: bool = False) -> NDArray[np.float64]:
    r"""Rectangular (boxcar, Dirichlet) window.

    .. math::

       w[k] = 1, \qquad k = 0, \dots, N-1.

    Symmetric and periodic windows are identical. Source: [Harris1978]_ sec. V,
    [OS2010]_ sec. 7.5.

    Parameters
    ----------
    n : int
        Window length :math:`N \ge 1`.
    sym : bool, default False
        Accepted for a uniform signature; it does not change the result.

    Returns
    -------
    numpy.ndarray
        ``n`` ones, dtype ``float64``.

    Raises
    ------
    TypeError
        If ``n`` is not an integer or ``sym`` is not a bool.
    ValueError
        If ``n < 1``.
    """
    length = _check_length(n)
    _check_sym(sym)
    return np.ones(length)


def hann(n: SupportsIndex, *, sym: bool = False) -> NDArray[np.float64]:
    r"""Hann window.

    .. math::

       w[k] = 0.5 - 0.5 \cos\!\left(\frac{2\pi k}{L}\right)

    Generalised cosine with :math:`a = (0.5, 0.5)`; :math:`L` as in
    :func:`general_cosine`. Source: [Harris1978]_ sec. V (:math:`\cos^2` window),
    [OS2010]_ sec. 7.5.

    Parameters
    ----------
    n : int
        Window length :math:`N \ge 1`.
    sym : bool, default False
        ``True`` for the symmetric window, ``False`` for the periodic (DFT-even) one.

    Returns
    -------
    numpy.ndarray
        Window of shape ``(n,)`` and dtype ``float64``.

    Raises
    ------
    TypeError
        If ``n`` is not an integer or ``sym`` is not a bool.
    ValueError
        If ``n < 1``.
    """
    return general_cosine(n, (0.5, 0.5), sym=sym)


def hamming(n: SupportsIndex, *, sym: bool = False) -> NDArray[np.float64]:
    r"""Hamming window.

    .. math::

       w[k] = 0.54 - 0.46 \cos\!\left(\frac{2\pi k}{L}\right)

    Generalised cosine with :math:`a = (0.54, 0.46)`; :math:`L` as in
    :func:`general_cosine`. Source: [Harris1978]_ sec. V, [OS2010]_ sec. 7.5.

    Parameters
    ----------
    n : int
        Window length :math:`N \ge 1`.
    sym : bool, default False
        ``True`` for the symmetric window, ``False`` for the periodic (DFT-even) one.

    Returns
    -------
    numpy.ndarray
        Window of shape ``(n,)`` and dtype ``float64``.

    Raises
    ------
    TypeError
        If ``n`` is not an integer or ``sym`` is not a bool.
    ValueError
        If ``n < 1``.
    """
    return general_cosine(n, (0.54, 0.46), sym=sym)


def blackman(n: SupportsIndex, *, sym: bool = False) -> NDArray[np.float64]:
    r"""Blackman window.

    .. math::

       w[k] = 0.42 - 0.5 \cos\!\left(\frac{2\pi k}{L}\right)
              + 0.08 \cos\!\left(\frac{4\pi k}{L}\right)

    Generalised cosine with :math:`a = (0.42, 0.5, 0.08)`; :math:`L` as in
    :func:`general_cosine`. Source: [Harris1978]_ sec. V, [OS2010]_ sec. 7.5.

    Parameters
    ----------
    n : int
        Window length :math:`N \ge 1`.
    sym : bool, default False
        ``True`` for the symmetric window, ``False`` for the periodic (DFT-even) one.

    Returns
    -------
    numpy.ndarray
        Window of shape ``(n,)`` and dtype ``float64``.

    Raises
    ------
    TypeError
        If ``n`` is not an integer or ``sym`` is not a bool.
    ValueError
        If ``n < 1``.
    """
    return general_cosine(n, (0.42, 0.5, 0.08), sym=sym)


def blackman_harris(n: SupportsIndex, *, sym: bool = False) -> NDArray[np.float64]:
    r"""Minimum 4-term Blackman-Harris window (-92 dB sidelobes).

    .. math::

       w[k] = \sum_{m=0}^{3} (-1)^m \, a_m \cos\!\left(\frac{2\pi m k}{L}\right),
       \qquad a = (0.35875, 0.48829, 0.14128, 0.01168)

    :math:`L` as in :func:`general_cosine`. Source: [Harris1978]_ sec. V
    (Blackman-Harris family, minimum 4-term -92 dB variant).

    Parameters
    ----------
    n : int
        Window length :math:`N \ge 1`.
    sym : bool, default False
        ``True`` for the symmetric window, ``False`` for the periodic (DFT-even) one.

    Returns
    -------
    numpy.ndarray
        Window of shape ``(n,)`` and dtype ``float64``.

    Raises
    ------
    TypeError
        If ``n`` is not an integer or ``sym`` is not a bool.
    ValueError
        If ``n < 1``.
    """
    return general_cosine(n, (0.35875, 0.48829, 0.14128, 0.01168), sym=sym)


def flattop(n: SupportsIndex, *, sym: bool = False) -> NDArray[np.float64]:
    r"""Five-term flat-top window for amplitude measurements.

    .. math::

       w[k] = \sum_{m=0}^{4} (-1)^m \, a_m \cos\!\left(\frac{2\pi m k}{L}\right),

    with :math:`a = (0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368)`
    and :math:`L` as in :func:`general_cosine`.

    Source: G. D'Antona and A. Ferrero, *Digital Signal Processing for Measurement
    Systems*, Springer, 2006, p. 70, doi:10.1007/0-387-28666-7. This is the variant
    in ``scipy.signal.windows.flattop``. Harris (1978) lists a different, 3-term
    flat-top window.

    Parameters
    ----------
    n : int
        Window length :math:`N \ge 1`.
    sym : bool, default False
        ``True`` for the symmetric window, ``False`` for the periodic (DFT-even) one.

    Returns
    -------
    numpy.ndarray
        Window of shape ``(n,)`` and dtype ``float64``. Some samples are negative.

    Raises
    ------
    TypeError
        If ``n`` is not an integer or ``sym`` is not a bool.
    ValueError
        If ``n < 1``.
    """
    coeffs = (0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368)
    return general_cosine(n, coeffs, sym=sym)


def kaiser(n: SupportsIndex, beta: float, *, sym: bool = False) -> NDArray[np.float64]:
    r"""Kaiser (Kaiser-Bessel) window.

    .. math::

       w[k] = \frac{I_0\!\left(\beta \sqrt{1 - \left(2k/L - 1\right)^2}\right)}{I_0(\beta)},

    where :math:`I_0` is the zeroth-order modified Bessel function of the first kind
    (computed with :func:`numpy.i0`) and :math:`L` is as in :func:`general_cosine`.
    :math:`\beta = 0` gives the rectangular window; larger :math:`\beta` lowers the
    sidelobes and widens the main lobe. Harris writes the same window with
    :math:`\alpha = \beta / \pi`.

    Source: J. F. Kaiser, "Nonrecursive digital filter design using the
    :math:`I_0`-sinh window function", *Proc. IEEE ISCAS*, 1974, pp. 20-23;
    [OS2010]_ sec. 7.5; [Harris1978]_ sec. V (Kaiser-Bessel).

    Parameters
    ----------
    n : int
        Window length :math:`N \ge 1`.
    beta : float
        Shape parameter :math:`\beta \ge 0`, finite.
    sym : bool, default False
        ``True`` for the symmetric window, ``False`` for the periodic (DFT-even) one.

    Returns
    -------
    numpy.ndarray
        Window of shape ``(n,)`` and dtype ``float64``.

    Raises
    ------
    TypeError
        If ``n`` is not an integer, ``beta`` is not a real number or ``sym`` is
        not a bool.
    ValueError
        If ``n < 1`` or ``beta`` is negative or not finite.
    """
    length = _check_length(n)
    is_sym = _check_sym(sym)
    shape = _check_beta(beta)
    if length == 1:
        return np.ones(1)
    m = _symmetric_length(length, is_sym)
    k = np.arange((m + 1) // 2, dtype=np.float64)
    r = 2.0 * k / (m - 1) - 1.0
    half = np.i0(shape * np.sqrt(1.0 - r * r)) / np.i0(shape)
    return _finish(_mirror(half, m), length)


_WINDOWS: dict[str, Callable[..., NDArray[np.float64]]] = {
    "rectangular": rectangular,
    "hann": hann,
    "hamming": hamming,
    "blackman": blackman,
    "blackman_harris": blackman_harris,
    "flattop": flattop,
    "kaiser": kaiser,
}


def get_window(
    name: str, n: SupportsIndex, *, sym: bool = False, **params: Any
) -> NDArray[np.float64]:
    r"""Return a window by name.

    Parameters
    ----------
    name : str
        One of ``"rectangular"``, ``"hann"``, ``"hamming"``, ``"blackman"``,
        ``"blackman_harris"``, ``"flattop"``, ``"kaiser"``.
    n : int
        Window length :math:`N \ge 1`.
    sym : bool, default False
        ``True`` for the symmetric window (filter design), ``False`` for the
        periodic (DFT-even) window (spectral analysis). The default matches
        ``scipy.signal.get_window`` (``fftbins=True``).
    **params
        Window parameters, passed to the window function: ``beta`` for
        ``"kaiser"``; the other windows take none.

    Returns
    -------
    numpy.ndarray
        Window of shape ``(n,)`` and dtype ``float64``.

    Raises
    ------
    ValueError
        If ``name`` is unknown, ``n < 1`` or a parameter value is invalid.
    TypeError
        If ``name`` is not a string, ``n`` is not an integer, a required
        parameter is missing or an unknown parameter is given.

    Examples
    --------
    >>> from sigstack import get_window
    >>> get_window("hann", 4)
    array([0. , 0.5, 1. , 0.5])
    >>> get_window("kaiser", 5, sym=True, beta=0.0)
    array([1., 1., 1., 1., 1.])
    """
    if not isinstance(name, str):
        raise TypeError(f"name must be a string, got {type(name).__name__}")
    try:
        window = _WINDOWS[name]
    except KeyError:
        known = ", ".join(sorted(_WINDOWS))
        raise ValueError(f"unknown window {name!r}; known windows: {known}") from None
    return window(n, sym=sym, **params)


def _as_window(w: ArrayLike) -> NDArray[np.float64]:
    """Return ``w`` as a non-empty 1-D float64 array."""
    arr = np.asarray(w, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"w must be a non-empty 1-D array, got shape {arr.shape}")
    return arr


def coherent_gain(w: ArrayLike) -> float:
    r"""Coherent gain of a window: the gain it applies to a tone at a bin centre.

    .. math::

       \mathrm{CG} = \frac{1}{N} \sum_{k=0}^{N-1} w[k]

    Source: [Harris1978]_ sec. IV (figures of merit).

    Parameters
    ----------
    w : array_like
        Window samples, non-empty and 1-D.

    Returns
    -------
    float
        Coherent gain, 1.0 for the rectangular window.

    Raises
    ------
    ValueError
        If ``w`` is empty or not 1-D.
    """
    arr = _as_window(w)
    return float(np.sum(arr) / arr.size)


def enbw(w: ArrayLike) -> float:
    r"""Equivalent noise bandwidth of a window in DFT bins.

    .. math::

       \mathrm{ENBW} = \frac{N \sum_{k} w[k]^2}{\left(\sum_{k} w[k]\right)^2}

    The width of an ideal rectangular filter that passes the same white-noise
    power as the window's DFT bin. Source: [Harris1978]_ sec. IV (figures of merit).

    Parameters
    ----------
    w : array_like
        Window samples, non-empty and 1-D.

    Returns
    -------
    float
        ENBW in bins: 1.0 for the rectangular window, 1.5 for the periodic Hann
        window.

    Raises
    ------
    ValueError
        If ``w`` is empty, not 1-D, or its samples sum to zero.
    """
    arr = _as_window(w)
    total = float(np.sum(arr))
    if total == 0.0:
        raise ValueError("ENBW is undefined: the window samples sum to zero")
    return float(arr.size * np.sum(arr * arr) / (total * total))
