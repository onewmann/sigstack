"""Reference tests: SciPy as the test reference and textbook filter properties."""

import math
import unittest

import numpy as np
import scipy.signal as ss
from numpy.testing import assert_allclose

import sigstack.metrics as m
from sigstack.signals import multitone, white_noise

B = np.random.default_rng(0).standard_normal(7)
A = np.real(np.poly([0.5, -0.3 + 0.4j, -0.3 - 0.4j]))  # stable: poles inside the unit circle


class Accept(unittest.TestCase):
    def test_freq_response_matches_scipy(self) -> None:
        for whole in (False, True):
            f, h = m.freq_response(B, A, n=256, fs=48000.0, whole=whole)
            fr, hr = ss.freqz(B, A, worN=256, fs=48000.0, whole=whole)
            assert_allclose(f, fr, rtol=1e-12, atol=1e-9)
            assert_allclose(h, hr, rtol=1e-9, atol=1e-12)  # independent implementation, rounding

    def test_group_delay_matches_scipy(self) -> None:
        f, gd = m.group_delay(B, A, n=256, fs=48000.0)
        fr, gr = ss.group_delay((B, A), w=256, fs=48000.0)
        assert_allclose(f, fr, rtol=1e-12, atol=1e-9)
        assert_allclose(gd, gr, rtol=1e-7, atol=1e-8)

    def test_butterworth_cutoff(self) -> None:
        b, a = ss.butter(4, 0.2)  # -3.01 dB at w = 0.2 pi
        f, h = m.freq_response(b, a, n=1000)
        assert_allclose(20 * np.log10(abs(h[200])), -10 * math.log10(2.0), atol=1e-6)
        att = m.stopband_attenuation(
            f, h, (0.2 * math.pi - 1e-9, 0.2 * math.pi + 1e-9)
        )  # one grid point
        assert_allclose(att, 10 * math.log10(2.0), atol=1e-6)

    def test_chebyshev_ripple(self) -> None:
        b, a = ss.cheby1(4, 1.0, 0.3)  # equiripple passband, 1 dB
        f, h = m.freq_response(b, a, n=6000)
        assert_allclose(m.passband_ripple(f, h, (0.0, 0.3 * math.pi)), 1.0, atol=1e-3)

    def test_elliptic_stopband(self) -> None:
        b, a = ss.ellip(4, 0.5, 40.0, 0.3)  # even order: |H(pi)| sits on the 40 dB stopband ripple
        f, h = m.freq_response(b, a, n=6000)
        assert_allclose(m.stopband_attenuation(f, h, (0.8 * math.pi, math.pi)), 40.0, atol=1e-3)
        assert_allclose(m.passband_ripple(f, h, (0.0, 0.3 * math.pi)), 0.5, atol=1e-3)

    def test_linear_phase_fir(self) -> None:
        h = ss.firwin(31, 0.3)  # symmetric: delay 15 samples where H != 0
        _, gd = m.group_delay(h, n=500)
        ok = np.isfinite(gd)
        self.assertGreater(int(ok.sum()), 400)
        assert_allclose(gd[ok], 15.0, atol=1e-6)

    def test_snr_with_generated_signals(self) -> None:
        x, info = multitone(100_000, 1000.0, [50.0, 120.0], amplitudes=[1.0, 0.5])
        n, ninfo = white_noise(100_000, power=0.01, rng=3)
        expect = 10 * math.log10(info.power / ninfo.power)
        assert_allclose(
            m.snr_db_ref(x, x + n), expect, atol=0.05
        )  # sample noise power, 100k samples
        assert_allclose(m.snr_db(x, n), m.snr_db_ref(x, x + n), rtol=1e-9)
