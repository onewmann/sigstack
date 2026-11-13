"""Signal generation utilities for sigstack.

This subpackage provides seeded test signal generators with known ground truth.
Every generator returns a NumPy array and a :class:`SignalInfo` record containing
the frequencies, amplitudes, and theoretical mean power of the generated signal.
The same seed produces identical output.

Examples
--------
>>> from sigstack.signals import SignalInfo
>>> info = SignalInfo("impulse", 1.0, 4, (), (), 0.25)
>>> info.kind
'impulse'
"""

from __future__ import annotations

from .info import SignalInfo
from .modulation import am, fm
from .noise import pink_noise, white_noise
from .tones import chirp, impulse, multitone

__all__ = ["SignalInfo", "am", "chirp", "fm", "impulse", "multitone", "pink_noise", "white_noise"]
