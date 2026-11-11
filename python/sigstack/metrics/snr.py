"""Signal-to-noise ratio (SNR) in decibels."""

import math

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["snr_db", "snr_db_ref"]


def snr_db(signal: ArrayLike, noise: ArrayLike) -> float:
    r"""Compute the signal-to-noise ratio (SNR) in decibels.

    The SNR is defined as:

    .. math::

        \mathrm{SNR} = 10 \log_{10}\left(\frac{P_s}{P_n}\right)

    where $P_x$ is the mean power of signal $x$, computed over all elements
    (flattened):

    .. math::

        P_x = \frac{1}{N} \sum_{i=1}^{N} |x_i|^2

    Parameters
    ----------
    signal : ArrayLike
        The signal array. Can be real or complex, any dimensionality.
    noise : ArrayLike
        The noise array. Must have the same shape as `signal`.

    Returns
    -------
    float
        The SNR in decibels. Returns ``math.inf`` if noise power is zero and
        signal power is positive, or ``-math.inf`` if signal power is zero and
        noise power is positive.

    Raises
    ------
    ValueError
        If `signal` and `noise` have different shapes, if the input is empty,
        or if both powers are exactly zero.

    """
    sig = np.asarray(signal)
    noi = np.asarray(noise)

    if sig.shape != noi.shape:
        raise ValueError(
            f"signal and noise must have the same shape, got {sig.shape} and {noi.shape}"
        )

    if sig.size == 0:
        raise ValueError("input arrays must not be empty")

    p_sig = float(np.mean(np.abs(sig) ** 2))
    p_noi = float(np.mean(np.abs(noi) ** 2))

    if p_noi == 0.0 and p_sig > 0.0:
        return math.inf
    if p_sig == 0.0 and p_noi > 0.0:
        return -math.inf
    if p_sig == 0.0 and p_noi == 0.0:
        raise ValueError("both signal and noise powers are zero")

    return float(10.0 * math.log10(p_sig / p_noi))


def snr_db_ref(reference: ArrayLike, observed: ArrayLike) -> float:
    r"""Compute SNR in dB using a reference signal.

    The noise is estimated as the difference between the observed and
    reference signals:

    .. math::

        n = \text{observed} - \text{reference}

    Then the SNR is computed as :func:`snr_db` with `reference` as the signal
    and $n$ as the noise.

    Parameters
    ----------
    reference : ArrayLike
        The clean reference signal.
    observed : ArrayLike
        The noisy observed signal. Must have the same shape as `reference`.

    Returns
    -------
    float
        The SNR in decibels.

    Raises
    ------
    ValueError
        If `reference` and `observed` have different shapes, or if the input
        is empty, or if both powers are exactly zero.

    Notes
    -----
    This is a common metric for evaluating signal reconstruction quality.
    """
    ref = np.asarray(reference)
    obs = np.asarray(observed)

    if ref.shape != obs.shape:
        raise ValueError(
            f"reference and observed must have the same shape, got {ref.shape} and {obs.shape}"
        )

    noise = obs - ref
    return snr_db(ref, noise)
