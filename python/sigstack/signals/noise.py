"""Seeded noise signal generators.

This module provides :func:`white_noise` and :func:`pink_noise`, which generate
deterministic (seeded) noise signals with known theoretical power. Each generator
returns a tuple of the signal array and a :class:`~sigstack.signals.info.SignalInfo`
record describing the parameters and properties of the generated signal.

Randomness is controlled exclusively through the ``rng`` argument, which accepts an
integer seed, a :class:`numpy.random.Generator`, or ``None`` (for non-deterministic
output).

References
----------
.. [OS2010] A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed.,
   Pearson, 2010, sec. 2.10, Example 2.26 (white noise).
"""

from __future__ import annotations

import math
import operator

import numpy as np
from numpy.typing import NDArray

from sigstack.signals.info import SignalInfo

__all__ = ["pink_noise", "white_noise"]


def _resolve_rng(rng: int | np.random.Generator | None) -> np.random.Generator:
    """Resolve the ``rng`` argument to a :class:`numpy.random.Generator`.

    Parameters
    ----------
    rng : int, numpy.random.Generator, or None
        If an integer, a new generator is created with that seed.
        If a Generator, it is used directly.
        If None, a new generator is created with no seed (non-deterministic).

    Returns
    -------
    numpy.random.Generator
        The resolved random number generator.
    """
    if rng is None:
        return np.random.default_rng(None)
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, int) and not isinstance(rng, bool):
        return np.random.default_rng(rng)
    raise TypeError(
        f"rng must be an int, numpy.random.Generator, or None, got {type(rng).__name__}"
    )


def _check_n(n: object) -> int:
    """Validate that ``n`` is a positive integer (not bool).

    Parameters
    ----------
    n : object
        The value to check.

    Returns
    -------
    int
        The validated integer value.

    Raises
    ------
    TypeError
        If ``n`` is not an integer or is a boolean.
    ValueError
        If ``n`` is less than 1.
    """
    if isinstance(n, bool | np.bool_):
        raise TypeError("n must be an integer, got bool")
    try:
        value = operator.index(n)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(f"n must be an integer, got {type(n).__name__}") from None
    if value < 1:
        raise ValueError(f"n must be >= 1, got {value}")
    return value


def _check_fs(fs: object) -> float:
    """Validate that ``fs`` is a finite positive number.

    Parameters
    ----------
    fs : object
        The sampling frequency to check.

    Returns
    -------
    float
        The validated sampling frequency.

    Raises
    ------
    TypeError
        If ``fs`` is not a real number.
    ValueError
        If ``fs`` is not finite or not positive.
    """
    if isinstance(fs, bool) or not isinstance(fs, (int, float, np.integer, np.floating)):
        raise TypeError(f"fs must be a real number, got {type(fs).__name__}")
    value = float(fs)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"fs must be finite and positive, got {fs!r}")
    return value


def _check_power(power: object) -> float:
    """Validate that ``power`` is a finite non-negative number.

    Parameters
    ----------
    power : object
        The power value to check.

    Returns
    -------
    float
        The validated power value.

    Raises
    ------
    TypeError
        If ``power`` is not a real number.
    ValueError
        If ``power`` is not finite or is negative.
    """
    if isinstance(power, bool) or not isinstance(power, (int, float, np.integer, np.floating)):
        raise TypeError(f"power must be a real number, got {type(power).__name__}")
    value = float(power)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"power must be finite and >= 0, got {power!r}")
    return value


def white_noise(
    n: int,
    *,
    power: float = 1.0,
    fs: float = 1.0,
    as_complex: bool = False,
    rng: int | np.random.Generator | None = None,
) -> tuple[NDArray[np.float64] | NDArray[np.complex128], SignalInfo]:
    r"""Generate a white noise signal with known theoretical power.

    For real output, the signal is generated as:

    .. math::

       x[k] = \sqrt{P} \cdot g_k, \qquad k = 0, \dots, n-1

    where :math:`g_k` are independent standard normal random variables and
    :math:`P` is the requested power. The theoretical mean power is :math:`P`.

    For complex (analytic) output, the signal is generated as:

    .. math::

       x[k] = \sqrt{\frac{P}{2}} \left( g_{\text{re},k} + j \, g_{\text{im},k} \right)

    where :math:`g_{\text{re},k}` and :math:`g_{\text{im},k}` are independent
    standard normal random variables. The theoretical mean power (defined as
    :math:`\mathbb{E}[|x|^2]`) is :math:`P`.

    Parameters
    ----------
    n : int
        Number of samples, must be >= 1.
    power : float, default 1.0
        Theoretical mean power of the signal. Must be finite and >= 0.
    fs : float, default 1.0
        Sampling frequency in Hz. Must be finite and positive.
    as_complex : bool, default False
        If True, generate a complex (analytic) signal with dtype ``complex128``.
        Otherwise, generate a real signal with dtype ``float64``.
    rng : int, numpy.random.Generator, or None, default None
        Random number generator or seed. If an integer, a new generator is created
        with that seed. If a Generator, it is used directly. If None, a new
        non-deterministic generator is created.

    Returns
    -------
    tuple[numpy.ndarray, SignalInfo]
        A tuple containing:

        - The noise signal array of shape ``(n,)`` with dtype ``float64`` or
          ``complex128`` depending on ``as_complex``.
        - A :class:`~sigstack.signals.info.SignalInfo` record with metadata.

    Raises
    ------
    TypeError
        If ``n`` is not an integer, ``fs`` or ``power`` is not a real number,
        or ``rng`` has an invalid type.
    ValueError
        If ``n < 1``, ``fs`` is not finite and positive, or ``power`` is not
        finite and non-negative.

    Examples
    --------
    >>> import numpy as np
    >>> x, info = white_noise(4, power=2.0, rng=42)
    >>> x.shape
    (4,)
    >>> info.kind
    'white_noise'
    >>> info.power
    2.0

    Complex variant:

    >>> xc, _ = white_noise(4, as_complex=True, rng=42)
    >>> xc.dtype
    dtype('complex128')
    """
    n_val = _check_n(n)
    power_val = _check_power(power)
    fs_val = _check_fs(fs)
    generator = _resolve_rng(rng)

    if as_complex:
        z = generator.standard_normal((2, n_val))
        x = np.sqrt(power_val / 2.0) * (z[0] + 1j * z[1])
        x = x.astype(np.complex128)
    else:
        x = np.sqrt(power_val) * generator.standard_normal(n_val)
        x = x.astype(np.float64)

    info = SignalInfo(
        kind="white_noise",
        fs=fs_val,
        n_samples=n_val,
        freqs=(),
        amplitudes=(),
        power=power_val,
        params={},
    )
    return x, info


def pink_noise(
    n: int,
    *,
    power: float = 1.0,
    fs: float = 1.0,
    rng: int | np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], SignalInfo]:
    r"""Generate a pink noise signal with known theoretical power.

    Pink noise has a power spectral density proportional to :math:`1/f`. This
    implementation generates white noise, shapes its spectrum using the Voss-McCartney
    method (via FFT), removes DC, and normalizes to the requested power.

    The construction is:

    1. Generate white noise: :math:`w = g.\text{standard\_normal}(n)`
    2. Compute real FFT: :math:`X = \text{rfft}(w)`
    3. Compute frequency bins: :math:`f = \text{rfftfreq}(n)`
    4. Zero the DC component: :math:`X[0] = 0`
    5. Shape the spectrum: :math:`X[1:] = X[1:] / \sqrt{f[1:]}`
    6. Inverse FFT: :math:`y = \text{irfft}(X, n)`
    7. Normalize power: :math:`y = y \cdot \sqrt{P / \text{mean}(y^2)}`

    Parameters
    ----------
    n : int
        Number of samples, must be >= 2.
    power : float, default 1.0
        Theoretical mean power of the signal. Must be finite and >= 0.
    fs : float, default 1.0
        Sampling frequency in Hz. Must be finite and positive.
    rng : int, numpy.random.Generator, or None, default None
        Random number generator or seed. If an integer, a new generator is created
        with that seed. If a Generator, it is used directly. If None, a new
        non-deterministic generator is created.

    Returns
    -------
    tuple[numpy.ndarray, SignalInfo]
        A tuple containing:

        - The pink noise signal array of shape ``(n,)`` with dtype ``float64``.
        - A :class:`~sigstack.signals.info.SignalInfo` record with metadata.

    Raises
    ------
    TypeError
        If ``n`` is not an integer, ``fs`` or ``power`` is not a real number,
        or ``rng`` has an invalid type.
    ValueError
        If ``n < 2``, ``fs`` is not finite and positive, or ``power`` is not
        finite and non-negative.

    Examples
    --------
    >>> import numpy as np
    >>> y, info = pink_noise(64, power=0.5, rng=9)
    >>> y.shape
    (64,)
    >>> info.kind
    'pink_noise'
    >>> info.power
    0.5
    """
    n_val = _check_n(n)
    if n_val < 2:
        raise ValueError(f"n must be >= 2 for pink noise, got {n_val}")
    power_val = _check_power(power)
    fs_val = _check_fs(fs)
    generator = _resolve_rng(rng)

    w = generator.standard_normal(n_val)
    spec = np.fft.rfft(w)
    f = np.fft.rfftfreq(n_val)
    spec[0] = 0.0
    spec[1:] = spec[1:] / np.sqrt(f[1:])
    y = np.fft.irfft(spec, n_val)
    mean_sq = np.mean(y**2)
    if mean_sq > 0.0:
        y = y * np.sqrt(power_val / mean_sq)
    else:
        # Degenerate case: all zeros (extremely unlikely with standard normal input)
        y = np.zeros(n_val, dtype=np.float64)

    y = y.astype(np.float64)

    info = SignalInfo(
        kind="pink_noise",
        fs=fs_val,
        n_samples=n_val,
        freqs=(),
        amplitudes=(),
        power=power_val,
        params={},
    )
    return y, info
