"""Spectral analysis: short-time Fourier transform and overlap-add reconstruction.

This subpackage provides the short-time Fourier transform (:func:`stft`) and its
inverse by weighted overlap-add (:func:`istft`), together with the COLA and NOLA
consistency checks (:func:`check_cola`, :func:`check_nola`). The transforms are
implemented with NumPy only; SciPy is used solely as a test reference.

The public names are re-exported from :mod:`sigstack.spectral.shorttime` so that
``sigstack.spectral.stft`` and ``sigstack.spectral.shorttime.stft`` refer to the
same object.
"""

from sigstack.spectral.shorttime import check_cola, check_nola, istft, stft

__all__ = [
    "check_cola",
    "check_nola",
    "istft",
    "stft",
]
