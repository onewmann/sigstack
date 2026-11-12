"""Frequency response and group delay of digital filters."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["freq_response", "group_delay"]


def _validate(
    b: ArrayLike, a: ArrayLike, n: int, fs: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and convert filter coefficients.

    Parameters
    ----------
    b : ArrayLike
        Feedforward coefficients.
    a : ArrayLike
        Feedback coefficients.
    n : int
        Number of frequency points.
    fs : float
        Sampling frequency.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        Converted 1-D float64 arrays for b and a.

    Raises
    ------
    ValueError
        If any validation condition is violated.
    TypeError
        If n is not an integer or fs is not a number.
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if n < 1:
        raise ValueError("n must be >= 1")

    if not isinstance(fs, (float, int, np.floating, np.integer)):
        raise TypeError(f"fs must be a number, got {type(fs).__name__}")
    fs_val = float(fs)
    if not math.isfinite(fs_val) or fs_val <= 0.0:
        raise ValueError("fs must be positive and finite")

    b_arr = np.asarray(b, dtype=np.float64)
    a_arr = np.asarray(a, dtype=np.float64)

    if b_arr.ndim != 1:
        raise ValueError("b must be 1-D")
    if a_arr.ndim != 1:
        raise ValueError("a must be 1-D")
    if b_arr.size == 0:
        raise ValueError("b must be non-empty")
    if a_arr.size == 0:
        raise ValueError("a must be non-empty")
    if not np.all(np.isfinite(b_arr)):
        raise ValueError("b coefficients must be finite")
    if not np.all(np.isfinite(a_arr)):
        raise ValueError("a coefficients must be finite")
    if a_arr[0] == 0.0:
        raise ValueError("a[0] must not be zero")

    return b_arr, a_arr


def _freq_grid(n: int, fs: float, whole: bool) -> NDArray[np.float64]:
    """Build the frequency grid.

    Parameters
    ----------
    n : int
        Number of points.
    fs : float
        Sampling frequency.
    whole : bool
        If True, use full [0, 2*pi) range; otherwise [0, pi).

    Returns
    -------
    NDArray[np.float64]
        Frequency array in Hz (or rad/sample if fs=2*pi).
    """
    k = np.arange(n, dtype=np.float64)
    if whole:
        w = 2.0 * math.pi * k / n
    else:
        w = math.pi * k / n
    return w * fs / (2.0 * math.pi)


def freq_response(
    b: ArrayLike,
    a: ArrayLike = (1.0,),
    *,
    n: int = 512,
    fs: float = 2.0 * math.pi,
    whole: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    r"""Compute the frequency response of a digital filter.

    The transfer function is $H(z) = B(z) / A(z)$ where
    $B(z) = \sum_k b_k z^{-k}$ and $A(z) = \sum_k a_k z^{-k}$.
    On the unit circle, $z = e^{j\omega}$:

    .. math::
        H(e^{j\omega}) = \frac{\sum_i b_i e^{-j\omega i}}{\sum_i a_i e^{-j\omega i}}

    Parameters
    ----------
    b : ArrayLike
        Feedforward (numerator) coefficients.
    a : ArrayLike, optional
        Feedback (denominator) coefficients. Default is ``(1.0,)``.
    n : int, optional
        Number of frequency points. Default is 512.
    fs : float, optional
        Sampling frequency. Default is $2\pi$ (rad/sample).
    whole : bool, optional
        If True, grid spans $[0, 2\pi)$; otherwise $[0, \pi)$.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.complex128]]
        ``(f, h)`` where ``f`` is the frequency array and ``h`` is the
        complex frequency response.

    Raises
    ------
    ValueError
        If coefficients are invalid or parameters out of range.
    TypeError
        If ``n`` is not an integer.

    References
    ----------
    Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-Time Signal
    Processing*, 3rd ed., Pearson, sec. 5.1 (frequency response of LTI systems).
    """
    b_arr, a_arr = _validate(b, a, n, fs)
    f = _freq_grid(n, float(fs), whole)

    k = np.arange(n, dtype=np.float64)
    if whole:
        w = 2.0 * math.pi * k / n
    else:
        w = math.pi * k / n

    # Build exponent matrix: shape (n, max_len)
    nb = b_arr.size
    na = a_arr.size
    idx_b = np.arange(nb, dtype=np.float64)
    idx_a = np.arange(na, dtype=np.float64)

    # h_k = sum_i b_i exp(-1j w_k i) / sum_i a_i exp(-1j w_k i)
    num = np.exp(-1j * np.outer(w, idx_b)) @ b_arr
    den = np.exp(-1j * np.outer(w, idx_a)) @ a_arr

    h = (num / den).astype(np.complex128)
    return f, h


def group_delay(
    b: ArrayLike,
    a: ArrayLike = (1.0,),
    *,
    n: int = 512,
    fs: float = 2.0 * math.pi,
    whole: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Compute the group delay of a digital filter.

    The group delay is defined as $gd(\omega) = -\frac{d}{d\omega}\arg H(e^{j\omega})$
    in samples. For a rational system $H(z) = B(z)/A(z)$, it can be computed
    exactly via:

    .. math::
        c = b * \overline{a_{\text{rev}}}, \quad r_i = i

        gd(\omega_k) = \operatorname{Re}\!\left(
            \frac{\sum_i r_i\, c_i\, e^{-j\omega_k i}}
                 {\sum_i c_i\, e^{-j\omega_k i}}
        \right) - (|a| - 1)

    where $c = \text{conv}(b, \overline{a_{\text{rev}}})$.

    Parameters
    ----------
    b : ArrayLike
        Feedforward (numerator) coefficients.
    a : ArrayLike, optional
        Feedback (denominator) coefficients. Default is ``(1.0,)``.
    n : int, optional
        Number of frequency points. Default is 512.
    fs : float, optional
        Sampling frequency. Default is $2\pi$ (rad/sample).
    whole : bool, optional
        If True, grid spans $[0, 2\pi)$; otherwise $[0, \pi)$.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        ``(f, gd)`` where ``f`` is the frequency array and ``gd`` is the
        group delay in samples. Values are set to ``np.nan`` where the
        response magnitude is near zero.

    Raises
    ------
    ValueError
        If coefficients are invalid or parameters out of range.
    TypeError
        If ``n`` is not an integer.

    References
    ----------
    Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-Time Signal
    Processing*, 3rd ed., Pearson, sec. 5.1.1 (phase and group delay);
    S. K. Mitra (2011), *Digital Signal Processing*, 4th ed., McGraw-Hill, sec. 4.9.
    """
    b_arr, a_arr = _validate(b, a, n, fs)
    f = _freq_grid(n, float(fs), whole)

    k = np.arange(n, dtype=np.float64)
    if whole:
        w = 2.0 * math.pi * k / n
    else:
        w = math.pi * k / n

    # c = convolve(b, conj(a_reversed))
    a_rev_conj = np.conj(a_arr[::-1])
    c = np.convolve(b_arr, a_rev_conj)
    r = np.arange(c.size, dtype=np.float64)

    idx_c = np.arange(c.size, dtype=np.float64)
    num = np.exp(-1j * np.outer(w, idx_c)) @ (r * c)
    den = np.exp(-1j * np.outer(w, idx_c)) @ c

    # Threshold for near-zero response
    threshold = 1e-10 * np.sum(np.abs(c))
    mask = np.abs(den) < threshold

    gd = np.empty(n, dtype=np.float64)
    valid = ~mask
    gd[valid] = np.real(num[valid] / den[valid]) - (a_arr.size - 1)
    gd[mask] = np.nan

    return f, gd
