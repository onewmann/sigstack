"""Metadata record for generated test signals.

This module defines :class:`SignalInfo`, a frozen dataclass that stores the
parameters and theoretical properties of a generated signal. It is returned
alongside the signal array by every generator in :mod:`sigstack.signals`.

The record is immutable to prevent accidental modification after generation.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class SignalInfo:
    r"""Metadata for a generated test signal.

    This frozen dataclass stores the parameters and theoretical properties of a
    generated signal. It is returned alongside the signal array by every generator
    in :mod:`sigstack.signals`.

    Parameters
    ----------
    kind : str
        Type of the signal, e.g. ``"multitone"``, ``"chirp"``, ``"impulse"``,
        ``"white_noise"``, ``"pink_noise"``, ``"am"``, or ``"fm"``.
    fs : float
        Sampling frequency in Hz. Must be positive.
    n_samples : int
        Number of samples in the signal. Must be non-negative.
    freqs : tuple[float, ...]
        Frequencies present in the signal in Hz. For a multitone signal, these are
        the carrier frequencies. For noise signals, this is typically empty or
        contains the center frequency if applicable.
    amplitudes : tuple[float, ...]
        Amplitudes corresponding to each frequency in ``freqs``. For noise signals,
        this may be empty or contain a single value representing the amplitude scale.
    power : float
        Theoretical mean power of the signal. For a multitone with amplitudes $A_i$,
        the power is $\sum_i A_i^2 / 2$. For white noise with variance $\sigma^2$,
        the power is $\sigma^2$.
    params : dict[str, float], optional
        Additional parameters specific to the signal type. For example, for a chirp
        this might contain ``"f0"`` and ``"f1"``, or for AM/FM it might contain
        modulation indices. Defaults to an empty dictionary. The record keeps its own
        copy, and ``params`` is not part of the hash.

    Attributes
    ----------
    kind : str
        Type of the signal.
    fs : float
        Sampling frequency in Hz.
    n_samples : int
        Number of samples.
    freqs : tuple[float, ...]
        Frequencies in Hz.
    amplitudes : tuple[float, ...]
        Amplitudes corresponding to ``freqs``.
    power : float
        Theoretical mean power.
    params : dict[str, float]
        Additional parameters (a copy of the argument; excluded from the hash).

    Examples
    --------
    >>> info = SignalInfo("multitone", 1000.0, 100, (100.0,), (1.0,), 0.5)
    >>> info.kind
    'multitone'
    >>> info.power
    0.5

    The record is frozen:

    >>> try:
    ...     info.power = 1.0
    ... except dataclasses.FrozenInstanceError:
    ...     print("Frozen")
    Frozen
    """

    kind: str
    fs: float
    n_samples: int
    freqs: tuple[float, ...]
    amplitudes: tuple[float, ...]
    power: float
    params: dict[str, float] = dataclasses.field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        """Keep an own copy of params, so the caller's dict cannot change the record."""
        object.__setattr__(self, "params", dict(self.params))
