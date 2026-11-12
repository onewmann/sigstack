"""Reference tests: specifications checked with sigstack.metrics, more SciPy cases."""

import math
import unittest

import numpy as np
import scipy.signal as ss
from numpy.testing import assert_allclose

import sigstack.filters as flt
from sigstack.metrics import freq_response, group_delay, passband_ripple, stopband_attenuation


class Accept(unittest.TestCase):
    def test_kaiser_design_meets_spec(self) -> None:
        # Kaiser's formulas: 60 dB, transition 0.25..0.35 (x pi): beta = 0.1102 (A - 8.7), N = 73
        beta = 0.1102 * (60.0 - 8.7)
        h = flt.firwin(73, 0.3, window="kaiser", beta=beta)
        f, hr = freq_response(h, n=8192)
        self.assertGreaterEqual(stopband_attenuation(f, hr, (0.35 * math.pi, math.pi)), 58.9)
        self.assertLessEqual(passband_ripple(f, hr, (0.0, 0.25 * math.pi)), 0.021)
        _, gd = group_delay(h, n=512)
        _, hr512 = freq_response(h, n=512)
        ok = np.isfinite(gd) & (np.abs(hr512) > 1e-4)  # gd is ill-conditioned at the stopband zeros
        assert_allclose(gd[ok], 36.0, atol=1e-6)  # linear phase: (N - 1) / 2

    def test_remez_ripple_ratio(self) -> None:
        # weight 10 in the stopband: passband deviation = 10 x stopband deviation (equiripple)
        h = flt.remez(41, [0, 0.3, 0.4, 1.0], [1, 0], weight=[1, 10])
        f, hr = freq_response(h, n=16384)
        ds = 10 ** (-stopband_attenuation(f, hr, (0.4 * math.pi, math.pi)) / 20)
        r = passband_ripple(f, hr, (0.0, 0.3 * math.pi))
        dp = (10 ** (r / 20) - 1) / (10 ** (r / 20) + 1)
        assert_allclose(dp / ds, 10.0, rtol=0.02)

    def test_more_scipy_cases(self) -> None:
        for n, c, pz in [(101, [0.05, 0.15], False), (65, 0.45, True), (63, [0.3, 0.7], True)]:
            assert_allclose(
                flt.firwin(n, c, window="blackman", pass_zero=pz),
                ss.firwin(n, c, window="blackman", pass_zero=pz),
                rtol=1e-10,
                atol=1e-12,
            )
        assert_allclose(
            flt.firls(61, [0, 0.1, 0.2, 0.6, 0.7, 1.0], [0, 0, 1, 1, 0, 0]),
            ss.firls(61, [0, 0.1, 0.2, 0.6, 0.7, 1.0], [0, 0, 1, 1, 0, 0]),
            atol=1e-8,
        )
        bands, des, w = [0, 0.1, 0.15, 0.5, 0.55, 1.0], [1, 0, 1], [1, 5, 1]
        got = flt.remez(63, bands, des, weight=w)
        ref = ss.remez(63, bands, des, weight=w, fs=2.0)
        assert_allclose(got, ref, atol=1e-3)  # grids differ (SciPy 16 points per coefficient)
        k = np.arange(63)

        def werr(h: np.ndarray) -> float:  # max weighted error, dense grid
            e = 0.0
            for i in range(3):
                f = np.linspace(bands[2 * i], bands[2 * i + 1], 4000)
                m = np.abs(np.exp(-1j * np.pi * np.outer(f, k)) @ h)
                e = max(e, float(np.max(w[i] * np.abs(m - des[i]))))
            return e

        self.assertLessEqual(werr(got), werr(ref) * 1.001)  # minimax: no worse than SciPy
