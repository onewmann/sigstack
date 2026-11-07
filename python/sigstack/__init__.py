"""sigstack: block-based signal processing toolkit."""

from sigstack.core import BaseBlock, Block, Gain, coherent_gain, enbw, get_window

__version__ = "0.0.1"

__all__ = ["BaseBlock", "Block", "Gain", "coherent_gain", "enbw", "get_window"]
