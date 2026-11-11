"""Reference tests: line spectra, Bessel lines and SciPy as the test reference (ground rules)."""

import math
import unittest
from typing import Literal

import numpy as np
import scipy.signal
import scipy.special
from numpy.testing import assert_allclose

import sigstack.signals as sig


def lines(x: np.ndarray, n: int) -> np.ndarray:
    return np.abs(np.fft.rfft(x)) * 2.0 / n


class Accept(unittest.TestCase):
    def test_multitone_spectrum(self) -> None:
        x, info = sig.multitone(64, 64.0, [5.0, 12.0], amplitudes=[1.0, 0.5], phases=[0.3, 1.1])
        m = lines(x, 64)
        assert_allclose(m[[5, 12]], [1.0, 0.5], atol=1e-12)  # whole periods: exact lines
        self.assertLess(float(np.delete(m, [5, 12]).max()), 1e-12)
        assert_allclose(np.mean(x**2), info.power, atol=1e-12)

    def test_chirp_matches_scipy(self) -> None:
        n, fs = 200, 100.0
        t = np.arange(n) / fs
        cases: list[tuple[Literal["linear", "logarithmic"], float, float]] = [
            ("linear", 2.0, 30.0),
            ("linear", 40.0, 5.0),
            ("logarithmic", 1.0, 40.0),
        ]
        for method, f0, f1 in cases:
            x, _ = sig.chirp(n, fs, f0, f1, method=method, amplitude=1.5, phase=0.4)
            ref = 1.5 * scipy.signal.chirp(t, f0, n / fs, f1, method=method, phi=math.degrees(0.4))
            assert_allclose(x, ref, atol=1e-9)  # independent implementation, float rounding
        z, _ = sig.chirp(n, fs, 2.0, 30.0, as_complex=True)
        r, _ = sig.chirp(n, fs, 2.0, 30.0)
        assert_allclose(z.real, r, atol=1e-12)  # analytic variant: real part is the real chirp

    def test_am_spectrum(self) -> None:
        x, info = sig.am(64, 64.0, 16.0, 4.0, depth=0.6)
        assert_allclose(lines(x, 64)[[12, 16, 20]], [0.3, 1.0, 0.3], atol=1e-12)
        self.assertEqual(info.freqs, (12.0, 16.0, 20.0))

    def test_fm_bessel_lines(self) -> None:
        x, info = sig.fm(256, 256.0, 64.0, 8.0, deviation=16.0)
        m = lines(x, 256)
        for k in range(-3, 4):  # line at fc + k f_mod has amplitude |J_k(beta)|
            assert_allclose(
                m[64 + 8 * k], abs(scipy.special.jv(k, 2.0)), atol=1e-4
            )  # tiny aliasing of |k| >= 8
        self.assertEqual(info.params["beta"], 2.0)

    def test_pink_slope(self) -> None:
        n = 2**14
        p = np.mean(
            [np.abs(np.fft.rfft(sig.pink_noise(n, rng=s)[0])) ** 2 for s in range(8)], axis=0
        )
        f = np.arange(p.size)
        sel = slice(10, 2000)
        slope = np.polyfit(np.log10(f[sel]), np.log10(p[sel]), 1)[0]
        self.assertAlmostEqual(slope, -1.0, delta=0.1)  # 1/f power spectrum

    def test_white_noise_statistics(self) -> None:
        x, _ = sig.white_noise(200_000, power=2.0, rng=1)
        assert_allclose(np.mean(x**2), 2.0, rtol=0.02)  # sample variance, 200k samples
        z, _ = sig.white_noise(200_000, power=2.0, as_complex=True, rng=2)
        assert_allclose([np.mean(z.real**2), np.mean(z.imag**2)], [1.0, 1.0], rtol=0.02)

    def test_determinism_all(self) -> None:
        for f in [
            lambda r: sig.white_noise(50, rng=r),
            lambda r: sig.pink_noise(50, rng=r),
            lambda r: sig.multitone(50, 50.0, [3.0, 7.0], rng=r),
        ]:
            self.assertTrue(np.array_equal(f(42)[0], f(42)[0]))
