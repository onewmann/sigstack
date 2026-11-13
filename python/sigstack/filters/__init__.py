"""FIR and IIR filter design for sigstack.

This subpackage provides FIR filter design (window method, least squares,
Remez exchange) and IIR filter design in second-order sections (Butterworth,
Chebyshev I and II, elliptic).  All designs return zero-pole-gain or
second-order-section representations that can be used with the block-processing
model.

Public API
----------
butter : IIR Butterworth filter design (second-order sections).
cheby1 : IIR Chebyshev type-I filter design (second-order sections).
cheby2 : IIR Chebyshev type-II filter design (second-order sections).
ellip  : IIR elliptic filter design (second-order sections).
firls  : FIR least-squares filter design.
firwin : FIR window-method filter design.
remez  : FIR Remez-exchange filter design.
sos_response : Frequency response of a second-order-section filter.
"""

from sigstack.filters.firls import firls
from sigstack.filters.firwin import firwin
from sigstack.filters.iir import butter, cheby1, cheby2, ellip
from sigstack.filters.remez import remez
from sigstack.filters.zpk import sos_response

__all__: list[str] = [
    "butter",
    "cheby1",
    "cheby2",
    "ellip",
    "firls",
    "firwin",
    "remez",
    "sos_response",
]
