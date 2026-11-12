"""FIR filter design by the window method, least squares and Remez exchange."""

from sigstack.filters.firls import firls
from sigstack.filters.firwin import firwin
from sigstack.filters.remez import remez

__all__ = ["firls", "firwin", "remez"]
