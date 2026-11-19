r"""Welch power spectral density estimate and band power.

:func:`welch` estimates the power spectral density of a real or complex signal by
averaging the periodograms of overlapping, windowed frames (Welch's method), and
:func:`band_power` integrates a density over arbitrary, non-uniform frequency bands.

Definition
----------
With window :math:`w` of length :math:`N` (``nperseg``), hop :math:`R` (``hop``) and FFT
length :math:`N_\mathrm{fft} \ge N`, the frames are the contiguous segments
:math:`x_r = x[rR : rR + N]` for :math:`r = 0, \ldots, K - 1` with
:math:`K = \lfloor (n - N) / R \rfloor + 1`; trailing samples that do not fill a frame
are ignored. Each frame is detrended, multiplied by the window and transformed with an
FFT of length ``nfft``. The estimate is the mean of the modified periodograms over the
frames ([OS2010]_ Sec. 10.5.3, Eqs. (10.82)-(10.84); [W1967]_):

.. math::

    P[k] = \frac{s}{K} \sum_{r=0}^{K-1} |X_r[k]|^2,

where :math:`s` is the scaling factor:

- ``scaling="density"``: :math:`s = 1 / (f_s \sum_m w[m]^2)`. By Parseval's relation,
  :math:`\sum_k P[k] \, \Delta f` with :math:`\Delta f = f_s / N_\mathrm{fft}` equals the
  mean over the frames of :math:`\sum_m |w[m] x_r[m]|^2 / \sum_m w[m]^2`, the
  window-weighted mean power, which estimates the mean power of the signal.
- ``scaling="spectrum"``: :math:`s = 1 / (\sum_m w[m])^2`. For real input a cosine of
  amplitude :math:`A` at a bin centre appears with :math:`P = A^2/2` (:math:`A^2` at DC
  and Nyquist). For complex input a complex exponential :math:`A e^{j \omega n}` at a bin
  centre appears with :math:`A^2`, and a cosine with :math:`A^2/4` at each of
  :math:`\pm f`.

For real input the one-sided spectrum is returned: every bin except DC and, for even
``nfft``, Nyquist is doubled, so that it also carries the power of the matching negative
frequency. For complex input the two-sided spectrum is returned in FFT order.

References
----------
.. [OS2010] A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*,
   3rd ed., Pearson, 2010, Sec. 10.5.3.
.. [W1967] P. D. Welch, "The use of fast Fourier transform for the estimation of
   power spectra: a method based on time averaging over short, modified periodograms",
   *IEEE Trans. Audio Electroacoust.*, vol. 15, no. 2, pp. 70-73, June 1967,
   doi:10.1109/TAU.1967.1161901.
"""

from __future__ import annotations

import math
from typing import Any, Literal, SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sigstack.spectral.shorttime import _hop, _int, _nfft, _real, _window

__all__ = [
    "band_power",
    "welch",
]


def welch(
    x: ArrayLike,
    fs: float = 1.0,
    *,
    window: str | ArrayLike = "hann",
    nperseg: SupportsIndex = 256,
    hop: SupportsIndex | None = None,
    nfft: SupportsIndex | None = None,
    detrend: Literal["constant", "linear", False] | None = "constant",
    scaling: Literal["density", "spectrum"] = "density",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Welch power spectral density estimate.

    The signal is split into overlapping frames of ``nperseg`` samples, separated by
    ``hop``; each frame is detrended, multiplied by the window and transformed with an
    FFT of length ``nfft``, and the periodograms are averaged (see the module docstring
    for the formula). Trailing samples that do not fill a frame are ignored.

    Parameters
    ----------
    x : array_like
        Signal with time on the last axis; leading axes are channels, e.g. shape
        ``(n,)``, ``(c, n)`` or ``(a, b, n)``. Boolean, integer and real input is
        converted to ``float64`` and gives a one-sided spectrum; complex input is
        converted to ``complex128`` and gives a two-sided spectrum.
    fs : float, default 1.0
        Sampling frequency; it scales the frequency axis and the density scaling.
    window : str or array_like, default "hann"
        Window name for :func:`sigstack.core.get_window` (periodic window) or a 1-D
        array of length ``nperseg``. Windows with parameters are passed as arrays, e.g.
        ``get_window("kaiser", nperseg, beta=8.0)``.
    nperseg : int, default 256
        Frame length :math:`N \ge 1`, at most the number of samples.
    hop : int, optional
        Frame step :math:`1 \le R \le N`; default ``max(1, nperseg // 2)``, as for
        :func:`sigstack.spectral.stft`. For odd ``nperseg`` this is one sample less than
        the default of :func:`scipy.signal.welch` (``noverlap = nperseg // 2``).
    nfft : int, optional
        FFT length :math:`N_\mathrm{fft} \ge N` (zero padding of each frame); default
        ``nperseg``.
    detrend : {"constant", "linear"} or None, default "constant"
        Remove a constant offset ("constant") or a linear trend ("linear") from each
        frame before windowing; ``None`` (or ``False``) leaves the frames unchanged.
    scaling : {"density", "spectrum"}, default "density"
        ``"density"``: power per hertz (units squared per hertz), its integral
        estimates the mean power; ``"spectrum"``: power per bin, a real cosine of
        amplitude :math:`A` at a bin centre appears with :math:`A^2/2` (complex input:
        see the module docstring).

    Returns
    -------
    f : numpy.ndarray
        Frequencies, ``rfftfreq(nfft, 1 / fs)`` for real and ``fftfreq(nfft, 1 / fs)``
        for complex input.
    psd : numpy.ndarray
        Power spectral density of shape ``x.shape[:-1] + (len(f),)``, ``float64``.

    Raises
    ------
    ValueError
        If ``x`` is not numeric, has no samples or contains non-finite values; if the
        window is invalid or its squared sum is zero; if ``scaling="spectrum"`` and the
        window sums to zero; if ``fs`` is not finite and positive (or so small that
        ``1 / fs`` overflows); if an integer parameter is out of range.
        Also if ``detrend`` or ``scaling`` is not one of the allowed values.
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
    if not math.isfinite(1.0 / rate):
        raise ValueError(f"fs is too small, 1 / fs overflows: {fs!r}")
    nps = _int("nperseg", nperseg, minimum=1)
    if nps > data.shape[-1]:
        raise ValueError(f"nperseg must be <= the number of samples ({data.shape[-1]}), got {nps}")
    step = _hop(nps, hop)
    nf = _nfft(nps, nfft)
    w = _window(window, nps)
    if detrend is None or (isinstance(detrend, bool | np.bool_) and not detrend):
        detrend = None
    elif not isinstance(detrend, str) or detrend not in ("constant", "linear"):
        raise ValueError(f"detrend must be 'constant', 'linear', None or False, got {detrend!r}")
    if scaling not in ("density", "spectrum"):
        raise ValueError(f"scaling must be 'density' or 'spectrum', got {scaling!r}")
    w2 = float(np.sum(w * w))
    if w2 == 0.0:
        raise ValueError("window squared sum must not be zero")
    wsum = float(np.sum(w))
    if scaling == "spectrum" and wsum == 0.0:
        raise ValueError("window must not sum to zero for spectrum scaling")

    n = data.shape[-1]
    k_frames = (n - nps) // step + 1
    idx = (
        np.arange(k_frames, dtype=np.intp)[:, None] * step + np.arange(nps, dtype=np.intp)[None, :]
    )
    frames = data[..., idx]
    if detrend == "constant":
        frames = frames - np.mean(frames, axis=-1, keepdims=True)
    elif detrend == "linear":
        frames = frames - np.mean(frames, axis=-1, keepdims=True)
        if nps > 1:
            t = (np.arange(nps, dtype=np.float64) - (nps - 1) / 2.0)[None, :]
            slope = np.sum(t * frames, axis=-1, keepdims=True) / float(np.sum(t * t))
            frames = frames - slope * t
    frames = frames * w[None, :]
    if cplx:
        spec = np.fft.fft(frames, n=nf, axis=-1)
        f = np.fft.fftfreq(nf, 1.0 / rate)
    else:
        spec = np.fft.rfft(frames, n=nf, axis=-1)
        f = np.fft.rfftfreq(nf, 1.0 / rate)
    power = np.mean(np.abs(spec) ** 2, axis=-2)
    # divide step by step: the products rate * w2 or wsum**2 can overflow for extreme fs
    psd = power / w2 / rate if scaling == "density" else power / wsum / wsum
    if not cplx:
        last = nf // 2 if nf % 2 == 0 else nf // 2 + 1
        psd[..., 1:last] *= 2.0
    return f.astype(np.float64), np.ascontiguousarray(psd).astype(np.float64)


def band_power(f: ArrayLike, psd: ArrayLike, edges: ArrayLike) -> NDArray[np.float64]:
    r"""Integrate a power spectral density over frequency bands.

    The bands are the intervals between consecutive edges; all but the last are half-open
    ``[lo, hi)`` and the last is closed ``[lo, hi]``. Each bin is assigned to the band
    that contains its centre frequency, and the band power is the sum of the density over
    those bins times the frequency spacing :math:`\Delta f`. A centre within
    :math:`10^{-9} \Delta f` of an edge counts as lying on that edge, so that rounding in
    ``rfftfreq`` (for example ``4.999999999999999`` for 5 Hz) does not move a bin to the
    lower band.

    The input must be a density (:func:`welch` with ``scaling="density"``). Bands that
    together cover every bin sum to :math:`\sum_k P[k] \, \Delta f`; for a density from
    :func:`welch` that is the window-weighted mean power (see the module docstring).

    Parameters
    ----------
    f : array_like
        1-D frequencies of length ``M``, finite; after sorting strictly increasing and
        uniformly spaced (``fftfreq`` order is accepted). The spacing is
        :math:`\Delta f`.
    psd : array_like
        Power spectral density with shape ``(..., M)`` matching the last axis of ``f``;
        real, non-negative and finite.
    edges : array_like
        1-D band edges of length at least two, finite and strictly increasing.

    Returns
    -------
    numpy.ndarray
        Band powers of shape ``psd.shape[:-1] + (len(edges) - 1,)``, ``float64``.

    Raises
    ------
    ValueError
        If ``f`` is not a finite 1-D real array that is uniformly spaced after sorting;
        if ``psd`` has the wrong shape or contains negative, non-finite or complex
        values; or if ``edges`` is not a finite, strictly increasing 1-D real array of
        length at least two.
    """
    fa = np.asarray(f)
    if fa.dtype.kind not in "iuf":
        raise ValueError(f"f must be a real numeric array, got dtype {fa.dtype}")
    if fa.ndim != 1 or fa.size < 2:
        raise ValueError(f"f must be a 1-D array with at least two values, got shape {fa.shape}")
    f64 = fa.astype(np.float64)
    if not np.all(np.isfinite(f64)):
        raise ValueError("f must contain only finite values")
    fsrt = np.sort(f64)
    d = np.diff(fsrt)
    if d.size < 1 or float(np.min(d)) <= 0.0:
        raise ValueError("f must be strictly increasing after sorting")
    dm = float(np.mean(d))
    if float(np.max(np.abs(d - dm))) > 1e-6 * dm:
        raise ValueError("f must have a uniform frequency spacing")

    pa = np.asarray(psd)
    if pa.dtype.kind not in "biuf":
        raise ValueError(f"psd must be a real numeric array, got dtype {pa.dtype}")
    if pa.ndim < 1 or pa.shape[-1] != f64.size:
        raise ValueError(
            f"psd last axis ({pa.shape[-1] if pa.ndim >= 1 else None}) "
            f"must match len(f) ({f64.size})"
        )
    p64 = pa.astype(np.float64)
    if not np.all(np.isfinite(p64)):
        raise ValueError("psd must contain only finite values")
    if float(np.min(p64)) < 0.0:
        raise ValueError("psd must be non-negative")

    ea = np.asarray(edges)
    if ea.dtype.kind not in "iuf":
        raise ValueError(f"edges must be a real numeric array, got dtype {ea.dtype}")
    if ea.ndim != 1 or ea.size < 2:
        raise ValueError(
            f"edges must be a 1-D array with at least two values, got shape {ea.shape}"
        )
    e64 = ea.astype(np.float64)
    if not np.all(np.isfinite(e64)):
        raise ValueError("edges must contain only finite values")
    if float(np.min(np.diff(e64))) <= 0.0:
        raise ValueError("edges must be strictly increasing")

    nb = e64.size - 1
    out = np.zeros((*p64.shape[:-1], nb), dtype=np.float64)
    tol = 1e-9 * dm  # a centre this close to an edge lies on it (rounding in rfftfreq)
    for i in range(nb):
        lo, hi = float(e64[i]) - tol, float(e64[i + 1])
        upper = f64 <= hi + tol if i == nb - 1 else f64 < hi - tol
        mask = (f64 >= lo) & upper
        out[..., i] = p64[..., mask].sum(axis=-1) * dm
    return out
