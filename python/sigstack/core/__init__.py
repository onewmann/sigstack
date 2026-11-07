"""Core building blocks: the block interface, buffers, dtypes and window functions."""

from sigstack.core.block import BaseBlock, Block, Gain
from sigstack.core.windows import (
    blackman,
    blackman_harris,
    coherent_gain,
    enbw,
    flattop,
    general_cosine,
    get_window,
    hamming,
    hann,
    kaiser,
    rectangular,
)

__all__ = [
    "BaseBlock",
    "Block",
    "Gain",
    "blackman",
    "blackman_harris",
    "coherent_gain",
    "enbw",
    "flattop",
    "general_cosine",
    "get_window",
    "hamming",
    "hann",
    "kaiser",
    "rectangular",
]
