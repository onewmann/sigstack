"""Discrete wavelet transform with symmetric, zero, and periodization modes.

This module implements the one-dimensional discrete wavelet transform (DWT)
and its inverse for orthogonal wavelets using NumPy only.  Filter
coefficients are obtained from :mod:`sigstack.wavelets.filters`.

The public API mirrors PyWavelets' single-level functions:

* :func:`dwt` - one-level decomposition
* :func:`idwt` - one-level reconstruction
* :func:`wavedec` - multi-level decomposition
* :func:`waverec` - multi-level reconstruction
* :func:`dwt_max_level` - maximum admissible level

Each level is a two-channel analysis or synthesis filter bank with downsampling or
upsampling by two [Mertins2023]_. The boundary modes extend a finite signal periodically
(periodization, i.e. cyclic convolution), by half-sample symmetric reflection
(symmetric) or by zeros (zero) [Mertins2023]_; lengths and alignment match PyWavelets.

References
----------
.. [Mertins2023] A. Mertins, *Signaltheorie: Grundlagen der Signalbeschreibung,
   Filterbänke, Wavelets, Zeit-Frequenz-Analyse, Parameter- und Signalschätzung*, 5th ed.,
   Springer Vieweg, 2023,
   sec. 9.5.2 and 9.5.3 (wavelet analysis and synthesis by multirate filtering),
   sec. 7.10 (subband decomposition of finite-length signals).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sigstack.wavelets.filters import wavelet_filters

__all__ = ["dwt", "dwt_max_level", "idwt", "wavedec", "waverec"]

_VALID_MODES: tuple[str, ...] = ("symmetric", "zero", "periodization")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_float64_1d(x: ArrayLike) -> NDArray[np.float64]:
    """Convert *x* to a 1-D ``float64`` array, validating shape and values.

    Raises
    ------
    ValueError
        If *x* is not 1-D, is empty, contains non-finite values, or has a
        complex dtype.
    """
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"Input must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("Input must be non-empty")
    if np.iscomplexobj(arr):
        raise ValueError("Input must be real-valued")
    arr = arr.astype(np.float64, copy=True)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input must contain only finite values")
    return arr


def _validate_mode(mode: str) -> None:
    """Raise ``ValueError`` if *mode* is not a recognised boundary mode."""
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {_VALID_MODES}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dwt(
    x: ArrayLike,
    wavelet: str,
    mode: str = "symmetric",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""One-level discrete wavelet decomposition.

    Parameters
    ----------
    x : array_like
        1-D input signal of length $n$.
    wavelet : str
        Wavelet name (e.g. ``"haar"``, ``"db4"``, ``"sym5"``).
    mode : str, optional
        Boundary handling: ``"symmetric"`` (default), ``"zero"``, or
        ``"periodization"``.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        ``(ca, cd)`` - approximation and detail coefficients, each of length
        $(n + L - 1) // 2$ (symmetric/zero) or $P / 2$ where $P$ is the
        padded length (periodization).

    Raises
    ------
    ValueError
        If *x* is not a non-empty 1-D real finite array, if *wavelet* is
        unknown, or if *mode* is invalid.

    Notes
    -----
    For symmetric and zero modes with filter length $L$:

    .. math::

        K = (n + L - 1) // 2, \qquad
        ca[k] = \sum_{m=0}^{L-1} h_{dec}[m]\; x^{\sim}[2k + 1 - m]

    where $x^{\sim}$ is the extended signal.  For periodization, if $n$ is
    odd one extra sample $x[n-1]$ is appended so that $P$ is even, and

    .. math::

        ca[k] = \sum_{m=0}^{L-1} h_{dec}[m]\; x_p\!\left[(2k + L/2 - m)
        \bmod P\right].
    """
    x_arr = _to_float64_1d(x)
    _validate_mode(mode)
    wf = wavelet_filters(wavelet)
    dec_lo, dec_hi = wf.dec_lo, wf.dec_hi
    filt_len = len(dec_lo)
    n = len(x_arr)

    if mode == "periodization":
        # Pad to even length
        if n % 2 != 0:
            xp = np.concatenate([x_arr, x_arr[-1:]])
        else:
            xp = x_arr.copy()
        padded_len = len(xp)
        out_len = padded_len // 2
        ca = np.zeros(out_len, dtype=np.float64)
        cd = np.zeros(out_len, dtype=np.float64)
        k_vals = np.arange(out_len, dtype=np.int64)
        for m in range(filt_len):
            idx = (2 * k_vals + filt_len // 2 - m) % padded_len
            ca += dec_lo[m] * xp[idx]
            cd += dec_hi[m] * xp[idx]
    else:
        out_len = (n + filt_len - 1) // 2
        ca = np.zeros(out_len, dtype=np.float64)
        cd = np.zeros(out_len, dtype=np.float64)
        k_vals = np.arange(out_len, dtype=np.int64)
        for m in range(filt_len):
            idx_raw = 2 * k_vals + 1 - m
            if mode == "zero":
                mask = (idx_raw >= 0) & (idx_raw < n)
                gathered = np.zeros(out_len, dtype=np.float64)
                gathered[mask] = x_arr[idx_raw[mask]]
                ca += dec_lo[m] * gathered
                cd += dec_hi[m] * gathered
            else:  # symmetric
                idx = idx_raw % (2 * n)
                idx = np.where(idx < n, idx, 2 * n - 1 - idx)
                ca += dec_lo[m] * x_arr[idx]
                cd += dec_hi[m] * x_arr[idx]

    return ca, cd


def idwt(
    ca: ArrayLike,
    cd: ArrayLike,
    wavelet: str,
    mode: str = "symmetric",
) -> NDArray[np.float64]:
    r"""One-level discrete wavelet reconstruction.

    Parameters
    ----------
    ca : array_like
        Approximation coefficients of length $K$.
    cd : array_like
        Detail coefficients of length $K$ (must equal ``len(ca)``).
    wavelet : str
        Wavelet name.
    mode : str, optional
        Boundary handling: ``"symmetric"`` (default), ``"zero"``, or
        ``"periodization"``.

    Returns
    -------
    NDArray[np.float64]
        Reconstructed signal of length $2K - L + 2$ (symmetric/zero) or
        $2K$ (periodization).

    Raises
    ------
    ValueError
        If inputs are invalid, lengths mismatch, or the output length would
        be less than 1.

    Notes
    -----
    For symmetric and zero modes:

    .. math::

        x[i] = \sum_{k} \bigl( h_{rec}[j]\; ca[k] + g_{rec}[j]\; cd[k]
        \bigr), \quad j = i + L - 2 - 2k,\; 0 \le j < L

    For periodization:

    .. math::

        x\!\left[(2k + m - (L/2 - 1)) \bmod 2K\right]
        \mathrel{+}= h_{rec}[m]\; ca[k] + g_{rec}[m]\; cd[k].
    """
    a = _to_float64_1d(ca)
    d = _to_float64_1d(cd)
    if len(a) != len(d):
        raise ValueError(f"ca and cd must have equal length, got {len(a)} and {len(d)}")
    coeff_len = len(a)
    _validate_mode(mode)
    wf = wavelet_filters(wavelet)
    rec_lo, rec_hi = wf.rec_lo, wf.rec_hi
    filt_len = len(rec_lo)

    if mode == "periodization":
        out_len = 2 * coeff_len
        x = np.zeros(out_len, dtype=np.float64)
        k_vals = np.arange(coeff_len, dtype=np.int64)
        for m in range(filt_len):
            idx = (2 * k_vals + m - (filt_len // 2 - 1)) % out_len
            x[idx] += rec_lo[m] * a[k_vals] + rec_hi[m] * d[k_vals]
    else:
        out_len = 2 * coeff_len - filt_len + 2
        if out_len < 1:
            raise ValueError(
                f"Output length {out_len} is less than 1 "
                f"(coeff_len={coeff_len}, filt_len={filt_len}); input too short for this filter."
            )
        x = np.zeros(out_len, dtype=np.float64)
        k_vals = np.arange(coeff_len, dtype=np.int64)
        for m in range(filt_len):
            idx = m - (filt_len - 2) + 2 * k_vals
            mask = (idx >= 0) & (idx < out_len)
            x[idx[mask]] += rec_lo[m] * a[k_vals[mask]] + rec_hi[m] * d[k_vals[mask]]

    return x


def wavedec(
    x: ArrayLike,
    wavelet: str,
    mode: str = "symmetric",
    level: int | None = None,
) -> list[NDArray[np.float64]]:
    """Multi-level discrete wavelet decomposition.

    Parameters
    ----------
    x : array_like
        1-D input signal.
    wavelet : str
        Wavelet name.
    mode : str, optional
        Boundary handling (default ``"symmetric"``).
    level : int or None, optional
        Number of decomposition levels.  If ``None``, the maximum admissible
        level is used (see :func:`dwt_max_level`).

    Returns
    -------
    list[NDArray[np.float64]]
        ``[ca_j, cd_j, ..., cd_1]`` - coarsest approximation first, followed
        by detail coefficients from coarse to fine.  For ``level=0`` the
        input is returned as a single-element list.

    Raises
    ------
    ValueError
        If *x* is invalid, *wavelet* is unknown, *mode* is invalid, or
        *level* is negative.
    """
    x_arr = _to_float64_1d(x)
    _validate_mode(mode)
    wf = wavelet_filters(wavelet)
    filt_len = len(wf.dec_lo)

    if level is None:
        resolved_level = dwt_max_level(len(x_arr), filt_len)
    else:
        resolved_level = level
    if resolved_level < 0:
        raise ValueError(f"level must be >= 0, got {resolved_level}")

    if resolved_level == 0:
        return [x_arr.copy()]

    details: list[NDArray[np.float64]] = []
    a = x_arr
    for _ in range(resolved_level):
        a, d = dwt(a, wavelet, mode=mode)
        details.append(d)

    # details is [cd_1, cd_2, ..., cd_J]; reverse to get [cd_J, ..., cd_1]
    details.reverse()
    return [a, *details]


def waverec(
    coeffs: Sequence[ArrayLike],
    wavelet: str,
    mode: str = "symmetric",
) -> NDArray[np.float64]:
    """Multi-level discrete wavelet reconstruction.

    Parameters
    ----------
    coeffs : list of array_like
        ``[ca_j, cd_j, ..., cd_1]`` as returned by :func:`wavedec`.
    wavelet : str
        Wavelet name.
    mode : str, optional
        Boundary handling (default ``"symmetric"``).

    Returns
    -------
    NDArray[np.float64]
        Reconstructed signal.

    Raises
    ------
    ValueError
        If *coeffs* is empty, if any coefficient array is invalid, or if
        adjacent arrays have incompatible lengths (the only tolerated
        mismatch is ``len(a) == len(d) + 1``, in which case the last element
        of *a* is dropped before reconstruction).
    """
    if not coeffs:
        raise ValueError("coeffs must be a non-empty list")

    _validate_mode(mode)
    wavelet_filters(wavelet)  # validate name early

    # Convert all to float64 1-D arrays
    arrs = [_to_float64_1d(c) for c in coeffs]

    if len(arrs) == 1:
        return arrs[0].copy()

    a = arrs[0]
    for d in arrs[1:]:
        if len(a) == len(d) + 1:
            a = a[:-1]
        elif len(a) != len(d):
            raise ValueError(
                f"Coefficient length mismatch: len(a)={len(a)}, "
                f"len(d)={len(d)}; expected equal lengths or "
                f"len(a) == len(d) + 1."
            )
        a = idwt(a, d, wavelet, mode=mode)

    return a


def dwt_max_level(data_len: int, filter_len: int) -> int:
    r"""Largest admissible decomposition level.

    Parameters
    ----------
    data_len : int
        Signal length (must be ``>= 1``).
    filter_len : int
        Filter length (must be ``>= 2``).

    Returns
    -------
    int
        The largest integer $j \ge 0$ such that $(L - 1)\, 2^{j} \le n$.

    Raises
    ------
    ValueError
        If *data_len* < 1 or *filter_len* < 2.
    """
    if data_len < 1:
        raise ValueError(f"data_len must be >= 1, got {data_len}")
    if filter_len < 2:
        raise ValueError(f"filter_len must be >= 2, got {filter_len}")

    j = 0
    while (filter_len - 1) * (2 ** (j + 1)) <= data_len:
        j += 1
    return j
