"""Spectral analysis: short-time Fourier transform, Welch PSD and band power.

This subpackage provides the short-time Fourier transform (:func:`stft`) and its
inverse by weighted overlap-add (:func:`istft`), together with the COLA and NOLA
consistency checks (:func:`check_cola`, :func:`check_nola`). It also provides Welch's
power spectral density estimate (:func:`welch`) and integration of a density over
frequency bands (:func:`band_power`). The transforms are implemented with NumPy only;
SciPy is used solely as a test reference.

The public names are re-exported from :mod:`sigstack.spectral.shorttime` and
:mod:`sigstack.spectral.psd` so that, for example, ``sigstack.spectral.stft`` and
``sigstack.spectral.shorttime.stft`` refer to the same object.
"""

from sigstack.spectral.psd import band_power, welch
from sigstack.spectral.shorttime import check_cola, check_nola, istft, stft

__all__ = [
    "band_power",
    "check_cola",
    "check_nola",
    "istft",
    "stft",
    "welch",
]
