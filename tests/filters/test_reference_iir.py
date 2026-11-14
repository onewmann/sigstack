"""Acceptance (never shown to the models): specifications checked with sigstack.metrics,
harder SciPy cases.
"""

import math
import unittest

import numpy as np
import scipy.signal as ss
from numpy.testing import assert_allclose

import sigstack.filters as flt
from sigstack.metrics import passband_ripple, stopband_attenuation


class Accept(unittest.TestCase):
    def test_butterworth_cutoff(self) -> None:
        _, h = flt.sos_response(flt.butter(4, 0.2), n=1000)
        # -3.01 dB at Wn
        assert_allclose(20 * np.log10(abs(h[200])), -10 * math.log10(2.0), atol=1e-6)

    def test_chebyshev_ripple_and_stopband(self) -> None:
        f, h = flt.sos_response(flt.cheby1(4, 1.0, 0.3), n=6000)
        assert_allclose(passband_ripple(f, h, (0.0, 0.3 * math.pi)), 1.0, atol=1e-3)
        f, h = flt.sos_response(flt.cheby2(5, 40.0, 0.3), n=6000)
        assert_allclose(stopband_attenuation(f, h, (0.3 * math.pi, math.pi)), 40.0, atol=1e-3)

    def test_elliptic_ripple_and_stopband(self) -> None:
        f, h = flt.sos_response(flt.ellip(4, 0.5, 40.0, 0.3), n=6000)
        assert_allclose(passband_ripple(f, h, (0.0, 0.3 * math.pi)), 0.5, atol=1e-3)
        assert_allclose(stopband_attenuation(f, h, (0.8 * math.pi, math.pi)), 40.0, atol=1e-3)

    def test_harder_cases(self) -> None:
        for sos, ref, rtol in [
            (flt.butter(10, 0.05), ss.butter(10, 0.05, output="sos"), 1e-7),
            (
                flt.ellip(8, 0.1, 80.0, [0.3, 0.32], "band"),
                ss.ellip(8, 0.1, 80.0, [0.3, 0.32], "bandpass", output="sos"),
                1e-5,
            ),
            (
                flt.cheby2(7, 60.0, [0.1, 0.6], "stop"),
                ss.cheby2(7, 60.0, [0.1, 0.6], "bandstop", output="sos"),
                1e-7,
            ),
        ]:
            h = flt.sos_response(sos, n=2048)[1]
            assert_allclose(h, ss.sosfreqz(ref, worN=2048)[1], rtol=rtol, atol=1e-8)
            p = np.concatenate([np.roots(s[3:]) for s in sos])
            self.assertTrue(np.all(np.abs(p) < 1.0))
