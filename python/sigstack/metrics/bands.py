"""Band metrics for sampled frequency responses."""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["passband_ripple", "stopband_attenuation"]


def _validate_inputs(
    f: ArrayLike, h: ArrayLike, band: tuple[float, float]
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Validate and convert inputs to 1-D NumPy arrays.

    Parameters
    ----------
    f : ArrayLike
        Frequency sample points.
    h : ArrayLike
        Corresponding frequency response values.
    band : tuple[float, float]
        Inclusive frequency band ``(low, high)``.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.complex128]]
        Converted 1-D arrays for ``f`` and ``h``.

    Raises
    ------
    ValueError
        If shapes differ, inputs are empty, or band ordering is invalid.
    """
    f_arr = np.asarray(f, dtype=np.float64).ravel()
    h_arr = np.asarray(h, dtype=np.complex128).ravel()

    if f_arr.shape != h_arr.shape:
        raise ValueError("f and h must have the same shape")
    if f_arr.size == 0:
        raise ValueError("f and h must be non-empty")
    if band[0] > band[1]:
        raise ValueError("band[0] must be <= band[1]")

    return f_arr, h_arr


def _select_band(
    f: NDArray[np.float64], h: NDArray[np.complex128], band: tuple[float, float]
) -> NDArray[np.float64]:
    """Select magnitudes within the inclusive frequency band.

    Parameters
    ----------
    f : NDArray[np.float64]
        1-D array of frequency sample points.
    h : NDArray[np.complex128]
        1-D array of frequency response values.
    band : tuple[float, float]
        Inclusive frequency band ``(low, high)``.

    Returns
    -------
    NDArray[np.float64]
        Magnitudes ``|h|`` for grid points in the band.

    Raises
    ------
    ValueError
        If no grid point lies inside the band.
    """
    mask = (f >= band[0]) & (f <= band[1])
    if not np.any(mask):
        raise ValueError("no grid point inside the band")
    return np.abs(h[mask]).astype(np.float64)


def passband_ripple(f: ArrayLike, h: ArrayLike, band: tuple[float, float]) -> float:
    r"""Compute the passband ripple in dB.

    The passband ripple is defined as

    .. math::

        \text{ripple} = 20 \log_{10}\!\left(\frac{\max |H|}{\min |H|}\right)

    over all grid points satisfying ``band[0] <= f <= band[1]``.

    Parameters
    ----------
    f : ArrayLike
        Frequency sample points.
    h : ArrayLike
        Corresponding frequency response values (possibly complex).
    band : tuple[float, float]
        Inclusive frequency band ``(low, high)``.

    Returns
    -------
    float
        Passband ripple in dB. Returns ``math.inf`` if the minimum magnitude
        is zero and the maximum is positive. Returns ``0.0`` if all selected
        magnitudes are equal (including all zero).

    Raises
    ------
    ValueError
        If shapes differ, inputs are empty, band ordering is invalid, or no
        grid point lies inside the band.

    References
    ----------
    Oppenheim & Schafer, *Discrete-Time Signal Processing*, 3rd ed., sec. 7.1
    (filter specifications).
    """
    f_arr, h_arr = _validate_inputs(f, h, band)
    mags = _select_band(f_arr, h_arr, band)

    max_mag = float(np.max(mags))
    min_mag = float(np.min(mags))

    if max_mag == 0.0:
        return 0.0
    if min_mag == 0.0:
        return math.inf
    if max_mag == min_mag:
        return 0.0

    return float(20.0 * math.log10(max_mag / min_mag))


def stopband_attenuation(
    f: ArrayLike,
    h: ArrayLike,
    band: tuple[float, float],
    reference: float = 1.0,
) -> float:
    r"""Compute the stopband attenuation in dB.

    The stopband attenuation is defined as

    .. math::

        \text{attenuation} = 20 \log_{10}(\text{reference}) - 20 \log_{10}(\max |H|)

    over all grid points satisfying ``band[0] <= f <= band[1]``.

    Parameters
    ----------
    f : ArrayLike
        Frequency sample points.
    h : ArrayLike
        Corresponding frequency response values (possibly complex).
    band : tuple[float, float]
        Inclusive frequency band ``(low, high)``.
    reference : float, optional
        Positive magnitude reference. Default is ``1.0``.

    Returns
    -------
    float
        Stopband attenuation in dB. Returns ``math.inf`` if the maximum
        magnitude is zero.

    Raises
    ------
    ValueError
        If shapes differ, inputs are empty, band ordering is invalid, no grid
        point lies inside the band, or reference is not positive and finite.

    References
    ----------
    Oppenheim & Schafer, *Discrete-Time Signal Processing*, 3rd ed., sec. 7.1
    (filter specifications).
    """
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("reference must be positive and finite")

    f_arr, h_arr = _validate_inputs(f, h, band)
    mags = _select_band(f_arr, h_arr, band)

    max_mag = float(np.max(mags))

    if max_mag == 0.0:
        return math.inf

    return float(20.0 * math.log10(reference) - 20.0 * math.log10(max_mag))
