"""IIR filter design functions returning second-order sections."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from sigstack.filters.analog import buttap, cheb1ap, cheb2ap, ellipap
from sigstack.filters.zpk import (
    bilinear_zpk,
    lp2bp_zpk,
    lp2bs_zpk,
    lp2hp_zpk,
    lp2lp_zpk,
    zpk2sos,
)

__all__ = ["butter", "cheby1", "cheby2", "ellip"]


def _check_order(order: object) -> int:
    """Validate that ``order`` is an integer and return it.

    Parameters
    ----------
    order : object
        The filter order to check. Must be a Python or NumPy integer;
        booleans are rejected even though they subclass ``int``.

    Returns
    -------
    int
        The validated order as a Python ``int``, unchanged in value.

    Raises
    ------
    TypeError
        If ``order`` is not an integer (including floats such as ``4.0``,
        non-integer values such as ``4.5``, booleans, or strings).
    """
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
        raise TypeError(f"order must be an integer, got {type(order).__name__}")
    return int(order)


def _validate(
    order: int,
    wn: float | list[float],
    btype: str,
    fs: float | None,
    rp: float | None = None,
    rs: float | None = None,
) -> tuple[int, NDArray[np.float64], str]:
    """Check the design arguments; return ``(order, wn as float64 array, btype)``.

    Parameters
    ----------
    order : int
        Filter order (>= 1).
    wn : float or list of float
        Cutoff frequency(ies); see :func:`butter` for the accepted forms.
    btype : str
        One of ``'low'``, ``'high'``, ``'band'`` or ``'stop'``.
    fs : float or None
        Sampling frequency in Hz, or None for Nyquist-normalised ``wn``.
    rp : float or None
        Passband ripple in dB (Chebyshev I and elliptic).
    rs : float or None
        Stopband attenuation in dB (Chebyshev II and elliptic).

    Returns
    -------
    tuple[int, numpy.ndarray, str]
        The validated order, ``wn`` as a ``float64`` array, and ``btype``.

    Raises
    ------
    ValueError
        If ``order < 1``, if ``btype`` is not recognised, if any of ``rp``,
        ``rs`` or ``fs`` is non-finite, if ``fs <= 0``, if any element of
        ``wn`` is non-finite, or if the ``wn`` edges are out of range.
    """
    order = _check_order(order)
    if order < 1:
        raise ValueError("order must be >= 1")
    if btype not in ("low", "high", "band", "stop"):
        raise ValueError("btype must be 'low', 'high', 'band' or 'stop'")

    for name, value in (("rp", rp), ("rs", rs), ("fs", fs)):
        if value is not None and not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if fs is not None and fs <= 0:
        raise ValueError("fs must be > 0")

    wn_arr = np.asarray(wn, dtype=np.float64)
    if not np.all(np.isfinite(wn_arr)):
        raise ValueError("Wn must be finite")

    if btype in ("low", "high"):
        if wn_arr.ndim != 0:
            raise ValueError("Wn must be a scalar for low/high pass")
        wn_val = float(wn_arr)
        if fs is None:
            if not (0.0 < wn_val < 1.0):
                raise ValueError("Wn must satisfy 0 < Wn < 1 when fs is None")
        else:
            if not (0.0 < wn_val < fs / 2.0):
                raise ValueError(f"Wn must satisfy 0 < Wn < {fs / 2} when fs={fs}")
    else:
        if wn_arr.ndim != 1 or wn_arr.size != 2:
            raise ValueError("Wn must be a two-element list for band/stop")
        w0, w1 = float(wn_arr[0]), float(wn_arr[1])
        if not (w0 < w1):
            raise ValueError("Wn edges must be increasing")
        if fs is None:
            if not (0.0 < w0 and w1 < 1.0):
                raise ValueError("Wn edges must satisfy 0 < Wn[0] < Wn[1] < 1 when fs is None")
        else:
            if not (0.0 < w0 and w1 < fs / 2.0):
                raise ValueError(f"Wn edges must satisfy 0 < Wn[0] < Wn[1] < {fs / 2} when fs={fs}")

    if rp is not None and rp <= 0:
        raise ValueError("rp must be > 0")
    if rs is not None and rs <= 0:
        raise ValueError("rs must be > 0")
    if rp is not None and rs is not None and rs <= rp:
        raise ValueError("rs must be > rp")

    return order, wn_arr, btype


def _prewarp(wn: float) -> float:
    """Prewarp a digital frequency for the bilinear transform with fs=2."""
    return 4.0 * math.tan(math.pi * wn / 2.0)


def butter(
    order: int,
    wn: float | list[float],
    btype: str = "low",
    fs: float | None = None,
) -> NDArray[np.float64]:
    r"""Design a Butterworth IIR filter as second-order sections.

    Parameters
    ----------
    order : int
        Filter order (>= 1).
    wn : float or list of float
        Cutoff frequency(ies). If ``fs`` is None, normalised to Nyquist
        ($0 < W_n < 1$); otherwise in Hz with $0 < W_n < f_s/2$.
        For band/stop, a two-element increasing list.
    btype : {'low', 'high', 'band', 'stop'}, optional
        Filter type. Default is ``'low'``.
    fs : float or None, optional
        Sampling frequency in Hz. If None, ``wn`` is normalised to Nyquist.

    Returns
    -------
    numpy.ndarray
        Second-order sections array of shape ``(ceil(order/2), 6)`` with
        dtype ``float64``. Each row is ``[b0, b1, b2, 1.0, a1, a2]``.

    Raises
    ------
    ValueError
        If ``order < 1``; if ``btype`` is not one of ``'low'``, ``'high'``,
        ``'band'`` or ``'stop'``; if ``wn`` is not a scalar for ``'low'`` and
        ``'high'`` or not a two-element array for ``'band'`` and ``'stop'``; if the
        band edges are not increasing; if ``wn`` lies outside ``0 < wn < 1``
        (``0 < wn < fs / 2`` when ``fs`` is given).

    Notes
    -----
    The Butterworth prototype has poles on the unit circle in the left-half
    $s$-plane: $p_\ell = e^{j\pi(2\ell+N-1)/(2N)}$, $\ell=1,\dots,N$.

    References
    ----------
    .. [1] S. K. Mitra, *Digital Signal Processing: A Computer-Based Approach*, 4th ed.,
           McGraw-Hill, 2011, App. A.2 (Butterworth approximation) and
           App. B (analog highpass, bandpass and bandstop design).
    .. [2] A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed.,
           Pearson, 2010, Sec. 7.2.2 (bilinear transformation) and Sec. 7.4
           (frequency transformations of lowpass IIR filters).
    """
    order, wn_arr, btype = _validate(order, wn, btype, fs)

    if fs is not None:
        wn_scaled = wn_arr / (fs / 2.0)
    else:
        wn_scaled = wn_arr

    z, p, k = buttap(order)

    if btype == "low":
        wo = _prewarp(float(wn_scaled))
        z, p, k = lp2lp_zpk(z, p, k, wo)
    elif btype == "high":
        wo = _prewarp(float(wn_scaled))
        z, p, k = lp2hp_zpk(z, p, k, wo)
    elif btype == "band":
        w0 = _prewarp(float(wn_scaled[0]))
        w1 = _prewarp(float(wn_scaled[1]))
        wo = math.sqrt(w0 * w1)
        bw = w1 - w0
        z, p, k = lp2bp_zpk(z, p, k, wo, bw)
    else:  # stop
        w0 = _prewarp(float(wn_scaled[0]))
        w1 = _prewarp(float(wn_scaled[1]))
        wo = math.sqrt(w0 * w1)
        bw = w1 - w0
        z, p, k = lp2bs_zpk(z, p, k, wo, bw)

    z, p, k = bilinear_zpk(z, p, k, fs=2.0)
    sos = zpk2sos(z, p, k)
    return sos.astype(np.float64)


def cheby1(
    order: int,
    rp: float,
    wn: float | list[float],
    btype: str = "low",
    fs: float | None = None,
) -> NDArray[np.float64]:
    r"""Design a Chebyshev Type I IIR filter as second-order sections.

    Parameters
    ----------
    order : int
        Filter order (>= 1).
    rp : float
        Passband ripple in dB (> 0).
    wn : float or list of float
        Cutoff frequency(ies). See :func:`butter` for details.
    btype : {'low', 'high', 'band', 'stop'}, optional
        Filter type. Default is ``'low'``.
    fs : float or None, optional
        Sampling frequency in Hz.

    Returns
    -------
    numpy.ndarray
        Second-order sections array of shape ``(ceil(order/2), 6)``.

    Raises
    ------
    ValueError
        If ``order < 1``; if ``btype`` is not one of ``'low'``, ``'high'``,
        ``'band'`` or ``'stop'``; if ``wn`` is not a scalar for ``'low'`` and
        ``'high'`` or not a two-element array for ``'band'`` and ``'stop'``; if the
        band edges are not increasing; if ``wn`` lies outside ``0 < wn < 1``
        (``0 < wn < fs / 2`` when ``fs`` is given) or if ``rp <= 0``.

    Notes
    -----
    The Chebyshev Type I prototype has equiripple passband and monotonic
    stopband. Poles lie on an ellipse in the left-half $s$-plane with
    semi-axes determined by $\alpha = \sinh^{-1}(1/\epsilon)/N$ where
    $\epsilon = \sqrt{10^{r_p/10}-1}$.

    References
    ----------
    .. [1] S. K. Mitra, *Digital Signal Processing: A Computer-Based Approach*, 4th ed.,
           McGraw-Hill, 2011, App. A.3 (Chebyshev approximation) and
           App. B (analog highpass, bandpass and bandstop design).
    .. [2] A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed.,
           Pearson, 2010, Sec. 7.2.2 (bilinear transformation) and Sec. 7.4
           (frequency transformations of lowpass IIR filters).
    """
    order, wn_arr, btype = _validate(order, wn, btype, fs, rp=rp)

    if fs is not None:
        wn_scaled = wn_arr / (fs / 2.0)
    else:
        wn_scaled = wn_arr

    z, p, k = cheb1ap(order, rp)

    if btype == "low":
        wo = _prewarp(float(wn_scaled))
        z, p, k = lp2lp_zpk(z, p, k, wo)
    elif btype == "high":
        wo = _prewarp(float(wn_scaled))
        z, p, k = lp2hp_zpk(z, p, k, wo)
    elif btype == "band":
        w0 = _prewarp(float(wn_scaled[0]))
        w1 = _prewarp(float(wn_scaled[1]))
        wo = math.sqrt(w0 * w1)
        bw = w1 - w0
        z, p, k = lp2bp_zpk(z, p, k, wo, bw)
    else:  # stop
        w0 = _prewarp(float(wn_scaled[0]))
        w1 = _prewarp(float(wn_scaled[1]))
        wo = math.sqrt(w0 * w1)
        bw = w1 - w0
        z, p, k = lp2bs_zpk(z, p, k, wo, bw)

    z, p, k = bilinear_zpk(z, p, k, fs=2.0)
    sos = zpk2sos(z, p, k)
    return sos.astype(np.float64)


def cheby2(
    order: int,
    rs: float,
    wn: float | list[float],
    btype: str = "low",
    fs: float | None = None,
) -> NDArray[np.float64]:
    """Design a Chebyshev Type II IIR filter as second-order sections.

    Parameters
    ----------
    order : int
        Filter order (>= 1).
    rs : float
        Stopband attenuation in dB (> 0).
    wn : float or list of float
        Cutoff frequency(ies). See :func:`butter` for details.
    btype : {'low', 'high', 'band', 'stop'}, optional
        Filter type. Default is ``'low'``.
    fs : float or None, optional
        Sampling frequency in Hz.

    Returns
    -------
    numpy.ndarray
        Second-order sections array of shape ``(ceil(order/2), 6)``.

    Raises
    ------
    ValueError
        If ``order < 1``; if ``btype`` is not one of ``'low'``, ``'high'``,
        ``'band'`` or ``'stop'``; if ``wn`` is not a scalar for ``'low'`` and
        ``'high'`` or not a two-element array for ``'band'`` and ``'stop'``; if the
        band edges are not increasing; if ``wn`` lies outside ``0 < wn < 1``
        (``0 < wn < fs / 2`` when ``fs`` is given) or if ``rs <= 0``.

    Notes
    -----
    The Chebyshev Type II prototype has monotonic passband and equiripple
    stopband. Zeros lie on the imaginary axis; poles are in the left-half
    $s$-plane.

    References
    ----------
    .. [1] S. K. Mitra, *Digital Signal Processing: A Computer-Based Approach*, 4th ed.,
           McGraw-Hill, 2011, App. A.3 (Chebyshev approximation) and
           App. B (analog highpass, bandpass and bandstop design).
    .. [2] A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed.,
           Pearson, 2010, Sec. 7.2.2 (bilinear transformation) and Sec. 7.4
           (frequency transformations of lowpass IIR filters).
    """
    order, wn_arr, btype = _validate(order, wn, btype, fs, rs=rs)

    if fs is not None:
        wn_scaled = wn_arr / (fs / 2.0)
    else:
        wn_scaled = wn_arr

    z, p, k = cheb2ap(order, rs)

    if btype == "low":
        wo = _prewarp(float(wn_scaled))
        z, p, k = lp2lp_zpk(z, p, k, wo)
    elif btype == "high":
        wo = _prewarp(float(wn_scaled))
        z, p, k = lp2hp_zpk(z, p, k, wo)
    elif btype == "band":
        w0 = _prewarp(float(wn_scaled[0]))
        w1 = _prewarp(float(wn_scaled[1]))
        wo = math.sqrt(w0 * w1)
        bw = w1 - w0
        z, p, k = lp2bp_zpk(z, p, k, wo, bw)
    else:  # stop
        w0 = _prewarp(float(wn_scaled[0]))
        w1 = _prewarp(float(wn_scaled[1]))
        wo = math.sqrt(w0 * w1)
        bw = w1 - w0
        z, p, k = lp2bs_zpk(z, p, k, wo, bw)

    z, p, k = bilinear_zpk(z, p, k, fs=2.0)
    sos = zpk2sos(z, p, k)
    return sos.astype(np.float64)


def ellip(
    order: int,
    rp: float,
    rs: float,
    wn: float | list[float],
    btype: str = "low",
    fs: float | None = None,
) -> NDArray[np.float64]:
    """Design an elliptic (Cauer) IIR filter as second-order sections.

    Parameters
    ----------
    order : int
        Filter order (>= 1).
    rp : float
        Passband ripple in dB (> 0).
    rs : float
        Stopband attenuation in dB (> ``rp``).
    wn : float or list of float
        Cutoff frequency(ies). See :func:`butter` for details.
    btype : {'low', 'high', 'band', 'stop'}, optional
        Filter type. Default is ``'low'``.
    fs : float or None, optional
        Sampling frequency in Hz.

    Returns
    -------
    numpy.ndarray
        Second-order sections array of shape ``(ceil(order/2), 6)``.

    Raises
    ------
    ValueError
        If ``order < 1``; if ``btype`` is not one of ``'low'``, ``'high'``,
        ``'band'`` or ``'stop'``; if ``wn`` is not a scalar for ``'low'`` and
        ``'high'`` or not a two-element array for ``'band'`` and ``'stop'``; if the
        band edges are not increasing; if ``wn`` lies outside ``0 < wn < 1``
        (``0 < wn < fs / 2`` when ``fs`` is given), or if ``rp <= 0``, ``rs <= 0`` or ``rs <= rp``.

    Notes
    -----
    The elliptic prototype has equiripple passband and stopband, achieving
    the minimum order for given specifications. Poles and zeros are derived
    from Jacobi elliptic functions.

    References
    ----------
    .. [1] S. K. Mitra, *Digital Signal Processing: A Computer-Based Approach*, 4th ed.,
           McGraw-Hill, 2011, App. A.4 (elliptic approximation) and
           App. B (analog highpass, bandpass and bandstop design).
    .. [2] A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed.,
           Pearson, 2010, Sec. 7.2.2 (bilinear transformation) and Sec. 7.4
           (frequency transformations of lowpass IIR filters).
    """
    order, wn_arr, btype = _validate(order, wn, btype, fs, rp=rp, rs=rs)

    if fs is not None:
        wn_scaled = wn_arr / (fs / 2.0)
    else:
        wn_scaled = wn_arr

    z, p, k = ellipap(order, rp, rs)

    if btype == "low":
        wo = _prewarp(float(wn_scaled))
        z, p, k = lp2lp_zpk(z, p, k, wo)
    elif btype == "high":
        wo = _prewarp(float(wn_scaled))
        z, p, k = lp2hp_zpk(z, p, k, wo)
    elif btype == "band":
        w0 = _prewarp(float(wn_scaled[0]))
        w1 = _prewarp(float(wn_scaled[1]))
        wo = math.sqrt(w0 * w1)
        bw = w1 - w0
        z, p, k = lp2bp_zpk(z, p, k, wo, bw)
    else:  # stop
        w0 = _prewarp(float(wn_scaled[0]))
        w1 = _prewarp(float(wn_scaled[1]))
        wo = math.sqrt(w0 * w1)
        bw = w1 - w0
        z, p, k = lp2bs_zpk(z, p, k, wo, bw)

    z, p, k = bilinear_zpk(z, p, k, fs=2.0)
    sos = zpk2sos(z, p, k)
    return sos.astype(np.float64)
