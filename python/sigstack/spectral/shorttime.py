r"""Short-time Fourier transform and its inverse by weighted overlap-add.

:func:`stft` computes the short-time Fourier transform of a real or complex signal,
:func:`istft` reconstructs the signal by weighted overlap-add, and :func:`check_cola`
and :func:`check_nola` test a window and hop for the constant overlap-add (COLA) and
nonzero overlap-add (NOLA) conditions.

Definition
----------
With window :math:`w` of length :math:`N` (``nperseg``), hop :math:`R` (``hop``), FFT
length :math:`N_\mathrm{fft} \ge N` and :math:`h = \lfloor N/2 \rfloor`, frame :math:`r`
and frequency bin :math:`k` are

.. math::

    Z[k, r] = \frac{1}{\sum_m w[m]} \sum_{m=0}^{N-1} x[rR - h + m]\, w[m]\,
    e^{-j 2 \pi k m / N_\mathrm{fft}},

with :math:`x[n] = 0` outside the signal. This samples the time-dependent Fourier
transform [OS2010]_ at :math:`n = rR - h` and :math:`\lambda = 2 \pi k / N_\mathrm{fft}`,
so that frame :math:`r` is centred on sample :math:`rR`. The factor
:math:`1 / \sum_m w[m]` makes a cosine of amplitude :math:`A` at a bin centre appear
with :math:`|Z| = A/2`. The layout (zero padding, frame count, frequency and time axes)
matches ``scipy.signal.stft`` with its default ``boundary="zeros"`` and ``padded=True``.

Reconstruction
--------------
:func:`istft` inverts each frame with an inverse FFT and combines the frames by the
least-squares weighted overlap-add of Griffin and Lim [GL1984]_:

.. math::

    \hat{x}[n] = \frac{\sum_r w[n - rR + h]\, y_r[n - rR + h]}{\sum_r w^2[n - rR + h]},

where :math:`y_r` is the inverse FFT of frame :math:`r`. The denominator must not vanish:
this is the NOLA condition. Without modification of :math:`Z` the reconstruction is
exact. Plain overlap-add without the denominator reconstructs the signal up to a
constant gain exactly when the window satisfies COLA, O&S eq. (10.47).

References
----------
.. [OS2010] A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*,
   3rd ed., Pearson, 2010, sec. 10.3, eq. (10.18) (time-dependent Fourier transform) and
   sec. 10.3.5, eqs. (10.44)-(10.47) (overlap-add reconstruction and its condition).
.. [GL1984] D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time
   Fourier transform", *IEEE Trans. Acoust., Speech, Signal Process.*, vol. 32, no. 2,
   pp. 236-243, 1984. doi:10.1109/TASSP.1984.1164317
"""

from __future__ import annotations

import math
import numbers
import operator
from typing import Any, SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sigstack.core import get_window

__all__ = [
    "check_cola",
    "check_nola",
    "istft",
    "stft",
]


def _int(name: str, value: object, minimum: int = 1) -> int:
    """Return ``value`` as an int after checking it is an integer >= ``minimum``."""
    if isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be an integer, got bool")
    try:
        result = operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from None
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _hop(nperseg: int, hop: object) -> int:
    """Return the hop between frames; ``None`` gives ``max(1, nperseg // 2)``."""
    if hop is None:
        return max(1, nperseg // 2)
    result = _int("hop", hop, minimum=1)
    if result > nperseg:
        raise ValueError(f"hop must be <= nperseg ({nperseg}), got {result}")
    return result


def _nfft(nperseg: int, nfft: object) -> int:
    """Return the FFT length; ``None`` gives ``nperseg``."""
    if nfft is None:
        return nperseg
    return _int("nfft", nfft, minimum=nperseg)


def _window(window: str | ArrayLike, nperseg: int) -> NDArray[np.float64]:
    """Return the window as a 1-D ``float64`` array of length ``nperseg``.

    A string goes to :func:`sigstack.core.get_window` (periodic window; its errors
    propagate); anything else must be a finite 1-D real array of length ``nperseg``.
    """
    if isinstance(window, str):
        return get_window(window, nperseg)
    w = np.asarray(window)
    if w.ndim != 1:
        raise ValueError(f"window must be 1-D, got shape {w.shape}")
    if w.dtype.kind not in "biuf":
        raise ValueError(f"window must be real-valued, got dtype {w.dtype}")
    if w.shape[0] != nperseg:
        raise ValueError(f"window length must be nperseg ({nperseg}), got {w.shape[0]}")
    w64 = w.astype(np.float64)
    if not np.all(np.isfinite(w64)):
        raise ValueError("window must contain only finite values")
    return w64


def _real(name: str, value: object) -> float:
    """Return ``value`` as a finite float > 0 (0-d numeric arrays are unwrapped)."""
    if isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype.kind in "iuf":
        value = value.item()
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite, got {value!r}") from None
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return result


def _overlap_sums(w: NDArray[np.float64], hop: int, power: int) -> NDArray[np.float64]:
    r"""Return :math:`s[j] = \sum_{k \ge 0} w[j + k \cdot hop]^{power}` for ``j < hop``."""
    padded = np.concatenate((w**power, np.zeros((-w.shape[0]) % hop)))
    return np.asarray(padded.reshape(-1, hop).sum(axis=0), dtype=np.float64)


def stft(
    x: ArrayLike,
    fs: float = 1.0,
    *,
    window: str | ArrayLike = "hann",
    nperseg: SupportsIndex = 256,
    hop: SupportsIndex | None = None,
    nfft: SupportsIndex | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]]:
    r"""Short-time Fourier transform.

    The signal is padded with :math:`h = \lfloor N/2 \rfloor` zeros at both ends and
    with up to ``hop - 1`` further zeros at the end, so that the frames
    ``k * hop, ..., k * hop + nperseg - 1`` of the padded signal cover it completely:
    with :math:`L = n + 2h` and :math:`r = (N - L) \bmod R` there are
    :math:`K = (L + r - N) / R + 1` frames. Each frame is multiplied by the window,
    transformed with an FFT of length ``nfft`` and divided by :math:`\sum_m w[m]`
    (see the module docstring for the formula).

    Parameters
    ----------
    x : array_like
        Signal with time on the last axis; leading axes are channels, e.g. shape
        ``(n,)``, ``(c, n)`` or ``(a, b, n)``. Boolean, integer and real input is
        converted to ``float64`` and gives a one-sided spectrum; complex input is
        converted to ``complex128`` and gives a two-sided spectrum.
    fs : float, default 1.0
        Sampling frequency; it only scales the frequency and time axes.
    window : str or array_like, default "hann"
        Window name for :func:`sigstack.core.get_window` (periodic window) or a
        1-D array of length ``nperseg``. Its sum must not be zero.
    nperseg : int, default 256
        Frame length :math:`N \ge 1`.
    hop : int, optional
        Frame step :math:`1 \le R \le N`; default ``max(1, nperseg // 2)``.
    nfft : int, optional
        FFT length :math:`N_\mathrm{fft} \ge N` (zero padding of each frame); default
        ``nperseg``.

    Returns
    -------
    f : numpy.ndarray
        Frequencies, ``rfftfreq(nfft, 1 / fs)`` for real and ``fftfreq(nfft, 1 / fs)``
        for complex input.
    t : numpy.ndarray
        Frame centre times ``arange(K) * hop / fs``.
    Z : numpy.ndarray
        ``complex128`` array of shape ``x.shape[:-1] + (len(f), K)``. A cosine of
        amplitude :math:`A` at a bin centre gives :math:`|Z| = A/2`.

    Raises
    ------
    ValueError
        If ``x`` is not numeric, has no samples or contains non-finite values; if
        the window is invalid or sums to zero; if ``fs`` is not finite and positive;
        if an integer parameter is out of range.
    TypeError
        If ``fs`` is not a real number or ``nperseg``, ``hop`` or ``nfft`` is not an
        integer.
    """
    arr = np.asarray(x)
    if arr.dtype.kind not in "biufc":
        raise ValueError(f"x must be numeric, got dtype {arr.dtype}")
    if arr.ndim < 1 or arr.shape[-1] < 1:
        raise ValueError(f"x must have at least one sample on its last axis, got {arr.shape}")
    cplx = arr.dtype.kind == "c"
    data: NDArray[Any] = arr.astype(np.complex128 if cplx else np.float64)
    if not np.all(np.isfinite(data)):
        raise ValueError("x must contain only finite values")
    rate = _real("fs", fs)
    nps = _int("nperseg", nperseg, minimum=1)
    step = _hop(nps, hop)
    nf = _nfft(nps, nfft)
    w = _window(window, nps)
    wsum = float(np.sum(w))
    if wsum == 0.0:
        raise ValueError("window must not sum to zero")

    half = nps // 2
    length = data.shape[-1] + 2 * half
    extra = (nps - length) % step
    n_frames = (length + extra - nps) // step + 1
    pad = [(0, 0)] * (data.ndim - 1) + [(half, half + extra)]
    padded = np.pad(data, pad)
    frames = np.lib.stride_tricks.sliding_window_view(padded, nps, axis=-1)[..., ::step, :] * w
    if cplx:
        spec = np.fft.fft(frames, n=nf, axis=-1)
        f = np.fft.fftfreq(nf, 1.0 / rate)
    else:
        spec = np.fft.rfft(frames, n=nf, axis=-1)
        f = np.fft.rfftfreq(nf, 1.0 / rate)
    z = np.moveaxis(spec / wsum, -1, -2).astype(np.complex128, copy=False)
    t = np.arange(n_frames) * step / rate
    return f.astype(np.float64), t.astype(np.float64), z


def istft(
    z: ArrayLike,
    *,
    window: str | ArrayLike = "hann",
    nperseg: SupportsIndex = 256,
    hop: SupportsIndex | None = None,
    nfft: SupportsIndex | None = None,
    onesided: bool = True,
    n: SupportsIndex | None = None,
) -> NDArray[Any]:
    r"""Inverse short-time Fourier transform by weighted overlap-add.

    Inverts :func:`stft` when called with the same ``window``, ``nperseg``, ``hop``
    and ``nfft``: each frame is multiplied by :math:`\sum_m w[m]`, inverted with an
    inverse FFT of length ``nfft`` and cut to ``nperseg`` samples, and the frames are
    combined by the weighted overlap-add of the module docstring. The ``nperseg // 2``
    padding samples are removed at both ends.

    Parameters
    ----------
    z : array_like
        Spectrum of shape ``(..., n_freq, K)`` as returned by :func:`stft`, with
        ``n_freq = nfft // 2 + 1`` if ``onesided`` else ``nfft``, and ``K >= 1``.
    window : str or array_like, default "hann"
        The analysis window (name or 1-D array of length ``nperseg``). Window and
        hop must satisfy NOLA (:func:`check_nola`).
    nperseg : int, default 256
        Frame length.
    hop : int, optional
        Frame step; default ``max(1, nperseg // 2)``.
    nfft : int, optional
        FFT length; default ``nperseg``.
    onesided : bool, default True
        ``True`` for the one-sided spectrum of a real signal (real output),
        ``False`` for the two-sided spectrum of a complex signal (complex output).
    n : int, optional
        Number of samples to return, ``1 <= n <=`` the full length
        ``(K - 1) * hop + nperseg - 2 * (nperseg // 2)``; pass the original signal
        length to undo the end padding of :func:`stft`. Default: the full length.

    Returns
    -------
    numpy.ndarray
        Signal of shape ``z.shape[:-2] + (n,)``, ``float64`` if ``onesided`` else
        ``complex128``.

    Raises
    ------
    ValueError
        If ``z`` is not numeric, has fewer than two axes, no frames, the wrong number
        of frequencies or non-finite values; if the window is invalid, sums to zero
        or violates NOLA for ``hop``; if an integer parameter is out of range.
    TypeError
        If ``onesided`` is not a bool or ``nperseg``, ``hop``, ``nfft`` or ``n`` is
        not an integer.
    """
    if not isinstance(onesided, bool | np.bool_):
        raise TypeError(f"onesided must be a bool, got {type(onesided).__name__}")
    one = bool(onesided)
    arr = np.asarray(z)
    if arr.dtype.kind not in "biufc":
        raise ValueError(f"z must be numeric, got dtype {arr.dtype}")
    if arr.ndim < 2 or arr.shape[-1] < 1:
        raise ValueError(f"z must have shape (..., n_freq, K) with K >= 1, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("z must contain only finite values")
    nps = _int("nperseg", nperseg, minimum=1)
    step = _hop(nps, hop)
    nf = _nfft(nps, nfft)
    w = _window(window, nps)
    expected = nf // 2 + 1 if one else nf
    if arr.shape[-2] != expected:
        raise ValueError(f"z must have {expected} frequencies on axis -2, got {arr.shape[-2]}")
    wsum = float(np.sum(w))
    if wsum == 0.0 or not check_nola(w, nps, step):
        raise ValueError("window and hop must satisfy NOLA (and the window must not sum to zero)")

    spec = arr.astype(np.complex128) * wsum
    inv = np.fft.irfft(spec, n=nf, axis=-2) if one else np.fft.ifft(spec, n=nf, axis=-2)
    n_frames = arr.shape[-1]
    total = (n_frames - 1) * step + nps
    y = np.zeros((*arr.shape[:-2], total), dtype=np.float64 if one else np.complex128)
    norm = np.zeros(total)
    for k in range(n_frames):
        y[..., k * step : k * step + nps] += inv[..., :nps, k] * w
        norm[k * step : k * step + nps] += w * w
    y /= np.where(norm > 1e-10, norm, 1.0)
    half = nps // 2
    y = y[..., half : total - half]
    if n is None:
        return y
    count = _int("n", n, minimum=1)
    if count > y.shape[-1]:
        raise ValueError(f"n must be <= {y.shape[-1]}, got {count}")
    return y[..., :count]


def check_cola(
    window: str | ArrayLike, nperseg: SupportsIndex, hop: SupportsIndex, *, tol: float = 1e-10
) -> bool:
    r"""Check the constant overlap-add (COLA) condition.

    COLA holds if the window shifted by multiples of the hop adds up to a constant,
    :math:`\sum_r w[n - rR] = C` for all :math:`n` (O&S eq. (10.47)); then plain
    overlap-add of the windowed frames gives the signal times :math:`C`. Computed
    over one period: :math:`s[j] = \sum_{k \ge 0} w[j + kR]` for :math:`0 \le j < R`
    must satisfy :math:`\max_j |s[j] - \operatorname{median}(s)| <` ``tol``.

    Parameters
    ----------
    window : str or array_like
        Window name for :func:`sigstack.core.get_window` (periodic window) or a 1-D
        array of length ``nperseg``.
    nperseg : int
        Window length :math:`N \ge 1`.
    hop : int
        Hop :math:`1 \le R \le N`.
    tol : float, default 1e-10
        Absolute tolerance, finite and > 0.

    Returns
    -------
    bool
        Whether the condition holds.

    Raises
    ------
    ValueError
        If the window is invalid, ``nperseg`` or ``hop`` is out of range, or ``tol``
        is not finite and positive.
    TypeError
        If ``nperseg`` or ``hop`` is not an integer or ``tol`` is not a real number.
    """
    nps = _int("nperseg", nperseg, minimum=1)
    step = _hop(nps, _required_hop(hop))
    w = _window(window, nps)
    limit = _real("tol", tol)
    s = _overlap_sums(w, step, 1)
    return bool(np.max(np.abs(s - np.median(s))) < limit)


def check_nola(
    window: str | ArrayLike, nperseg: SupportsIndex, hop: SupportsIndex, *, tol: float = 1e-10
) -> bool:
    r"""Check the nonzero overlap-add (NOLA) condition.

    NOLA holds if the squared window shifted by multiples of the hop is nonzero
    everywhere, :math:`\sum_r w^2[n - rR] > 0`; it is the condition for the weighted
    overlap-add inverse :func:`istft`. Computed over one period:
    :math:`\min_j \sum_{k \ge 0} w^2[j + kR] >` ``tol`` for :math:`0 \le j < R`.

    Parameters
    ----------
    window : str or array_like
        Window name for :func:`sigstack.core.get_window` (periodic window) or a 1-D
        array of length ``nperseg``.
    nperseg : int
        Window length :math:`N \ge 1`.
    hop : int
        Hop :math:`1 \le R \le N`.
    tol : float, default 1e-10
        Absolute tolerance, finite and > 0.

    Returns
    -------
    bool
        Whether the condition holds.

    Raises
    ------
    ValueError
        If the window is invalid, ``nperseg`` or ``hop`` is out of range, or ``tol``
        is not finite and positive.
    TypeError
        If ``nperseg`` or ``hop`` is not an integer or ``tol`` is not a real number.
    """
    nps = _int("nperseg", nperseg, minimum=1)
    step = _hop(nps, _required_hop(hop))
    w = _window(window, nps)
    limit = _real("tol", tol)
    return bool(np.min(_overlap_sums(w, step, 2)) > limit)


def _required_hop(hop: object) -> object:
    """Reject ``None`` for the hop of the COLA and NOLA checks (no default there)."""
    if hop is None:
        raise TypeError("hop must be an integer, got NoneType")
    return hop
