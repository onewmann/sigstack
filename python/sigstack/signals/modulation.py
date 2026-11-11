"""AM and FM signal generators."""

from __future__ import annotations

import math
import numbers
import operator

import numpy as np
from numpy.typing import NDArray

from sigstack.signals.info import SignalInfo

__all__ = ["am", "fm"]


def _check_n(n: object) -> int:
    if isinstance(n, bool | np.bool_):
        raise TypeError("n must be an integer, got bool")
    try:
        length = operator.index(n)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(f"n must be an integer, got {type(n).__name__}") from None
    if length < 0:
        raise ValueError(f"n must be non-negative, got {length}")
    return length


def _check_fs(fs: object) -> float:
    if isinstance(fs, bool) or not isinstance(fs, numbers.Real):
        raise TypeError(f"fs must be a real number, got {type(fs).__name__}")
    value = float(fs)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"fs must be finite and positive, got {fs!r}")
    return value


def _check_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _check_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return bool(value)


def am(
    n: int,
    fs: float,
    fc: float,
    f_mod: float,
    *,
    depth: float = 0.5,
    amplitude: float = 1.0,
    as_complex: bool = False,
) -> tuple[NDArray[np.float64] | NDArray[np.complex128], SignalInfo]:
    r"""Generate an AM (amplitude-modulated) signal.

    Parameters
    ----------
    n : int
        Number of samples.
    fs : float
        Sampling frequency in Hz.
    fc : float
        Carrier frequency in Hz.
    f_mod : float
        Modulation frequency in Hz.
    depth : float, default 0.5
        Modulation index (0 to 1).
    amplitude : float, default 1.0
        Carrier amplitude.
    as_complex : bool, default False
        If True, return the complex analytic signal.

    Returns
    -------
    tuple[NDArray[np.float64] | NDArray[np.complex128], SignalInfo]
        The signal array and its metadata.
    """
    length = _check_n(n)
    sample_rate = _check_fs(fs)
    carrier = _check_real(fc, "fc")
    mod_freq = _check_real(f_mod, "f_mod")
    mod_depth = _check_real(depth, "depth")
    amp = _check_real(amplitude, "amplitude")
    is_complex = _check_bool(as_complex, "as_complex")

    if not (0.0 <= mod_depth <= 1.0):
        raise ValueError(f"depth must be in [0, 1], got {mod_depth!r}")
    if mod_freq <= 0.0:
        raise ValueError(f"f_mod must be positive, got {mod_freq!r}")

    if is_complex:
        if abs(carrier) + mod_freq >= sample_rate / 2.0:
            raise ValueError(
                f"for complex AM, |fc| + f_mod must be < fs/2; "
                f"|{carrier}| + {mod_freq} = {abs(carrier) + mod_freq}, fs/2 = {sample_rate / 2.0}"
            )
    else:
        if not (0.0 < mod_freq < carrier):
            raise ValueError(
                f"for real AM, require 0 < f_mod < fc; got f_mod={mod_freq}, fc={carrier}"
            )
        if carrier + mod_freq >= sample_rate / 2.0:
            raise ValueError(
                f"for real AM, fc + f_mod must be < fs/2; "
                f"{carrier} + {mod_freq} = {carrier + mod_freq}, fs/2 = {sample_rate / 2.0}"
            )

    t = np.arange(length, dtype=np.float64) / sample_rate
    envelope = 1.0 + mod_depth * np.cos(2.0 * math.pi * mod_freq * t)

    if is_complex:
        x = amp * envelope * np.exp(1j * 2.0 * math.pi * carrier * t).astype(np.complex128)
        power = amp**2 * (1.0 + mod_depth**2 / 2.0)
    else:
        x = (amp * envelope * np.cos(2.0 * math.pi * carrier * t)).astype(np.float64)
        power = amp**2 / 2.0 * (1.0 + mod_depth**2 / 2.0)

    info = SignalInfo(
        kind="am",
        fs=sample_rate,
        n_samples=length,
        freqs=(carrier - mod_freq, carrier, carrier + mod_freq),
        amplitudes=(amp * mod_depth / 2.0, amp, amp * mod_depth / 2.0),
        power=power,
        params={"depth": float(mod_depth), "f_mod": float(mod_freq)},
    )
    return x, info


def fm(
    n: int,
    fs: float,
    fc: float,
    f_mod: float,
    *,
    deviation: float,
    amplitude: float = 1.0,
    as_complex: bool = False,
) -> tuple[NDArray[np.float64] | NDArray[np.complex128], SignalInfo]:
    r"""Generate an FM (frequency-modulated) signal.

    Parameters
    ----------
    n : int
        Number of samples.
    fs : float
        Sampling frequency in Hz.
    fc : float
        Carrier frequency in Hz.
    f_mod : float
        Modulation frequency in Hz.
    deviation : float
        Frequency deviation in Hz (keyword-only, required).
    amplitude : float, default 1.0
        Carrier amplitude.
    as_complex : bool, default False
        If True, return the complex analytic signal.

    Returns
    -------
    tuple[NDArray[np.float64] | NDArray[np.complex128], SignalInfo]
        The signal array and its metadata.
    """
    length = _check_n(n)
    sample_rate = _check_fs(fs)
    carrier = _check_real(fc, "fc")
    mod_freq = _check_real(f_mod, "f_mod")
    dev = _check_real(deviation, "deviation")
    amp = _check_real(amplitude, "amplitude")
    is_complex = _check_bool(as_complex, "as_complex")

    if mod_freq <= 0.0:
        raise ValueError(f"f_mod must be positive, got {mod_freq!r}")
    if dev < 0.0:
        raise ValueError(f"deviation must be non-negative, got {dev!r}")

    if is_complex:
        if abs(carrier) + dev >= sample_rate / 2.0:
            raise ValueError(
                f"for complex FM, |fc| + deviation must be < fs/2; "
                f"|{carrier}| + {dev} = {abs(carrier) + dev}, fs/2 = {sample_rate / 2.0}"
            )
    else:
        if carrier - dev <= 0.0:
            raise ValueError(
                f"for real FM, fc - deviation must be > 0; {carrier} - {dev} = {carrier - dev}"
            )
        if carrier + dev >= sample_rate / 2.0:
            raise ValueError(
                f"for real FM, fc + deviation must be < fs/2; "
                f"{carrier} + {dev} = {carrier + dev}, fs/2 = {sample_rate / 2.0}"
            )

    beta = dev / mod_freq
    t = np.arange(length, dtype=np.float64) / sample_rate
    theta = 2.0 * math.pi * carrier * t + beta * np.sin(2.0 * math.pi * mod_freq * t)

    if is_complex:
        x = (amp * np.exp(1j * theta)).astype(np.complex128)
        power = amp**2
    else:
        x = (amp * np.cos(theta)).astype(np.float64)
        power = amp**2 / 2.0

    info = SignalInfo(
        kind="fm",
        fs=sample_rate,
        n_samples=length,
        freqs=(carrier,),
        amplitudes=(amp,),
        power=power,
        params={
            "deviation": float(dev),
            "f_mod": float(mod_freq),
            "beta": float(beta),
            "carson_bandwidth": float(2.0 * dev + 2.0 * mod_freq),
        },
    )
    return x, info
