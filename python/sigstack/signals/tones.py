"""Tone-based test signal generators.

This module provides :func:`multitone`, :func:`chirp` and :func:`impulse`.
Each generator returns a NumPy array and a :class:`~sigstack.signals.info.SignalInfo`
record describing the signal's parameters and theoretical properties.

References
----------
.. [OS2010] A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed.,
   Pearson, 2010: sec. 2.1 (unit sample and sinusoidal sequences, `multitone`, `impulse`);
   sec. 10.3, Example 10.9 (linear chirp).
"""

from __future__ import annotations

import math
import numbers
import operator
from collections.abc import Sequence
from typing import Literal, SupportsIndex, cast

import numpy as np
from numpy.typing import NDArray

from .info import SignalInfo

__all__ = ["chirp", "impulse", "multitone"]


def multitone(
    n: int,
    fs: float,
    freqs: Sequence[float],
    *,
    amplitudes: Sequence[float] | None = None,
    phases: Sequence[float] | None = None,
    as_complex: bool = False,
    rng: int | np.random.Generator | None = None,
) -> tuple[NDArray[np.float64] | NDArray[np.complex128], SignalInfo]:
    r"""Generate a multitone signal.

    The signal is the sum of sinusoidal tones:

    .. math::

       x[k] = \sum_{i} a_i \cos\!\left(2\pi f_i \frac{k}{f_s} + \phi_i\right)

    for real output, or

    .. math::

       x[k] = \sum_{i} a_i e^{j\left(2\pi f_i \frac{k}{f_s} + \phi_i\right)}

    for complex (analytic) output.

    Parameters
    ----------
    n : int
        Number of samples. Must be positive.
    fs : float
        Sampling frequency in Hz. Must be finite and positive.
    freqs : Sequence[float]
        Frequencies in Hz. Must be non-empty. For real signals, each frequency
        must satisfy :math:`0 < f_i < f_s/2`. For complex signals,
        :math:`|f_i| < f_s/2`.
    amplitudes : Sequence[float], optional
        Amplitudes corresponding to ``freqs``. Defaults to ones.
    phases : Sequence[float], optional
        Phases in radians corresponding to ``freqs``. If ``None`` and ``rng`` is
        not ``None``, random phases are drawn from a uniform distribution on
        :math:`[0, 2\pi)`. Otherwise defaults to zeros.
    as_complex : bool, default False
        If True, generate the complex analytic signal.
    rng : int or numpy.random.Generator, optional
        Seed or generator for random phases.

    Returns
    -------
    tuple
        A tuple ``(x, info)`` where ``x`` is the signal array and ``info`` is a
        :class:`SignalInfo` record.

    Raises
    ------
    TypeError
        If ``n`` is not an integer or ``fs`` is not a real number.
    ValueError
        If ``n <= 0``, ``fs`` is not positive, ``freqs`` is empty, frequencies are
        out of range, or ``amplitudes``/``phases`` have wrong length.

    Examples
    --------
    >>> x, info = multitone(8, 8.0, [1.0], amplitudes=[2.0])
    >>> info.kind
    'multitone'
    """
    # Validate n
    if isinstance(n, bool | np.bool_):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    try:
        n_val = operator.index(cast("SupportsIndex", n))
    except TypeError:
        raise TypeError(f"n must be an integer, got {type(n).__name__}") from None
    if n_val <= 0:
        raise ValueError(f"n must be positive, got {n_val}")

    # Validate fs using the shared helper to accept NumPy integer scalars
    fs_val = _check_finite_real("fs", fs, positive=True)

    # Validate freqs
    if len(freqs) == 0:
        raise ValueError("freqs must be non-empty")
    freqs_arr = np.asarray(freqs, dtype=np.float64)
    nyquist = fs_val / 2.0
    if as_complex:
        if not np.all(np.abs(freqs_arr) < nyquist):
            raise ValueError(f"all frequencies must satisfy |f| < {nyquist} for complex signals")
    else:
        if not (np.all(freqs_arr > 0.0) and np.all(freqs_arr < nyquist)):
            raise ValueError(f"all frequencies must satisfy 0 < f < {nyquist} for real signals")

    # Validate amplitudes
    if amplitudes is None:
        amps = np.ones(len(freqs), dtype=np.float64)
    else:
        if len(amplitudes) != len(freqs):
            raise ValueError(
                f"amplitudes length {len(amplitudes)} must match freqs length {len(freqs)}"
            )
        amps = np.asarray(amplitudes, dtype=np.float64)

    # Validate phases and resolve rng
    if phases is not None:
        if len(phases) != len(freqs):
            raise ValueError(f"phases length {len(phases)} must match freqs length {len(freqs)}")
        phis = np.asarray(phases, dtype=np.float64)
    elif rng is not None:
        if isinstance(rng, np.random.Generator):
            gen = rng
        else:
            gen = np.random.default_rng(rng)
        phis = gen.uniform(0.0, 2.0 * np.pi, size=len(freqs))
    else:
        phis = np.zeros(len(freqs), dtype=np.float64)

    # Generate signal
    t = np.arange(n_val, dtype=np.float64) / fs_val
    if as_complex:
        x = np.zeros(n_val, dtype=np.complex128)
        for f, a, p in zip(freqs_arr, amps, phis, strict=True):
            x += a * np.exp(1j * (2.0 * np.pi * f * t + p))
    else:
        x = np.zeros(n_val, dtype=np.float64)
        for f, a, p in zip(freqs_arr, amps, phis, strict=True):
            x += a * np.cos(2.0 * np.pi * f * t + p)

    # Compute power
    if as_complex:
        power = float(np.sum(amps**2))
    else:
        power = float(np.sum(amps**2) / 2.0)

    info = SignalInfo(
        kind="multitone",
        fs=fs_val,
        n_samples=n_val,
        freqs=tuple(freqs_arr.tolist()),
        amplitudes=tuple(amps.tolist()),
        power=power,
        params={},
    )
    return x, info


def _check_integer(name: str, value: object, *, positive: bool = False) -> int:
    """Validate an integer scalar and return its int value."""
    if isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    try:
        result = operator.index(cast("SupportsIndex", value))
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from None
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _check_finite_real(name: str, value: object, *, positive: bool = False) -> float:
    """Validate a finite real scalar and return its float value."""
    if isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if positive and value_float <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return value_float


def chirp(
    n: int,
    fs: float,
    f0: float,
    f1: float,
    *,
    amplitude: float = 1.0,
    phase: float = 0.0,
    method: Literal["linear", "logarithmic"] = "linear",
    as_complex: bool = False,
) -> tuple[NDArray[np.float64] | NDArray[np.complex128], SignalInfo]:
    r"""Generate a chirp signal.

    A chirp is a sinusoid whose instantaneous frequency changes over time. For a
    linear chirp,

    .. math::

       \theta[k] = 2\pi \left(f_0 t[k] + \frac{f_1 - f_0}{2T} t[k]^2\right) + \phi,

    and for a logarithmic chirp,

    .. math::

       \theta[k] =
       2\pi \frac{f_0 T}{\ln(f_1/f_0)}
       \left[\left(\frac{f_1}{f_0}\right)^{t[k]/T} - 1\right] + \phi.

    The output is

    .. math::

       x[k] = A \cos(\theta[k])

    for real output, or

    .. math::

       x[k] = A e^{j\theta[k]}

    for complex (analytic) output.

    Parameters
    ----------
    n : int
        Number of samples. Must be positive.
    fs : float
        Sampling frequency in Hz. Must be finite and positive.
    f0 : float
        Start frequency in Hz. Must be finite.
    f1 : float
        End frequency in Hz. Must be finite.
    amplitude : float, default 1.0
        Signal amplitude. Must be finite.
    phase : float, default 0.0
        Initial phase in radians. Must be finite.
    method : {"linear", "logarithmic"}, default "linear"
        Chirp type.
    as_complex : bool, default False
        If True, generate the complex analytic signal.

    Returns
    -------
    tuple
        A tuple ``(x, info)`` where ``x`` is the signal array and ``info`` is a
        :class:`SignalInfo` record.

    Raises
    ------
    TypeError
        If ``n`` is not an integer, or ``fs``, ``f0``, ``f1``, ``amplitude`` or
        ``phase`` are not real numbers.
    ValueError
        If ``n <= 0``, ``fs`` is not positive, a numeric value is not finite,
        ``method`` is not supported, a frequency is outside the allowed range, or
        logarithmic chirp parameters are invalid.
    """
    n_val = _check_integer("n", n, positive=True)
    fs_val = _check_finite_real("fs", fs, positive=True)
    f0_val = _check_finite_real("f0", f0)
    f1_val = _check_finite_real("f1", f1)
    amplitude_val = _check_finite_real("amplitude", amplitude)
    phase_val = _check_finite_real("phase", phase)

    if method not in ("linear", "logarithmic"):
        raise ValueError(f"method must be 'linear' or 'logarithmic', got {method!r}")

    nyquist = fs_val / 2.0
    if as_complex:
        if not (abs(f0_val) <= nyquist and abs(f1_val) <= nyquist):
            raise ValueError(f"for complex chirps, |f0| and |f1| must be <= fs/2 = {nyquist}")
    else:
        if not (0.0 <= f0_val <= nyquist and 0.0 <= f1_val <= nyquist):
            raise ValueError(f"for real chirps, 0 <= f0 and 0 <= f1 must be <= fs/2 = {nyquist}")

    if method == "logarithmic":
        if not (f0_val > 0.0 and f1_val > 0.0 and f0_val != f1_val):
            raise ValueError(
                "for a logarithmic chirp, f0 and f1 must both be positive and different"
            )

    duration = n_val / fs_val
    t = np.arange(n_val, dtype=np.float64) / fs_val

    if method == "linear":
        theta = (
            2.0 * np.pi * (f0_val * t + (f1_val - f0_val) * t * t / (2.0 * duration)) + phase_val
        )
        params = {
            "duration": float(duration),
            "rate": float((f1_val - f0_val) / duration),
        }
    else:
        ratio = f1_val / f0_val
        theta = (
            2.0 * np.pi * f0_val * duration / math.log(ratio) * (ratio ** (t / duration) - 1.0)
            + phase_val
        )
        params = {"duration": float(duration)}

    if as_complex:
        x = amplitude_val * np.exp(1j * theta)
        power = float(amplitude_val * amplitude_val)
    else:
        x = amplitude_val * np.cos(theta)
        power = float(amplitude_val * amplitude_val / 2.0)

    info = SignalInfo(
        kind="chirp",
        fs=fs_val,
        n_samples=n_val,
        freqs=(float(f0_val), float(f1_val)),
        amplitudes=(float(amplitude_val),),
        power=power,
        params=params,
    )
    return x, info


def impulse(
    n: int,
    *,
    fs: float = 1.0,
    index: int = 0,
    amplitude: float = 1.0,
    as_complex: bool = False,
) -> tuple[NDArray[np.float64] | NDArray[np.complex128], SignalInfo]:
    r"""Generate a discrete impulse signal.

    The signal is zero everywhere except at sample index ``index``, where it equals
    ``amplitude``.

    Parameters
    ----------
    n : int
        Number of samples. Must be positive.
    fs : float, default 1.0
        Sampling frequency in Hz. Must be finite and positive.
    index : int, default 0
        Sample index of the impulse. Must satisfy :math:`0 \le \texttt{index} < n`.
    amplitude : float, default 1.0
        Impulse amplitude. Must be finite.
    as_complex : bool, default False
        If True, generate a complex-valued impulse.

    Returns
    -------
    tuple
        A tuple ``(x, info)`` where ``x`` is the signal array and ``info`` is a
        :class:`SignalInfo` record.

    Raises
    ------
    TypeError
        If ``n`` or ``index`` are not integers, or ``fs`` or ``amplitude`` are not
        real numbers.
    ValueError
        If ``n <= 0``, ``fs`` is not positive, ``amplitude`` is not finite, or
        ``index`` is outside ``[0, n)``.
    """
    n_val = _check_integer("n", n, positive=True)
    fs_val = _check_finite_real("fs", fs, positive=True)
    amplitude_val = _check_finite_real("amplitude", amplitude)

    if isinstance(index, bool | np.bool_):
        raise TypeError(f"index must be an integer, got {type(index).__name__}")
    try:
        index_val = operator.index(cast("SupportsIndex", index))
    except TypeError:
        raise TypeError(f"index must be an integer, got {type(index).__name__}") from None
    if index_val < 0 or index_val >= n_val:
        raise ValueError(f"index must satisfy 0 <= index < n, got {index_val} with n={n_val}")

    if as_complex:
        x = np.zeros(n_val, dtype=np.complex128)
    else:
        x = np.zeros(n_val, dtype=np.float64)
    x[index_val] = amplitude_val

    info = SignalInfo(
        kind="impulse",
        fs=fs_val,
        n_samples=n_val,
        freqs=(),
        amplitudes=(float(amplitude_val),),
        power=float(amplitude_val * amplitude_val / n_val),
        params={"index": float(index_val), "energy": float(amplitude_val * amplitude_val)},
    )
    return x, info
