"""Acceptance (never shown to the models).

Perfect reconstruction, energy, vanishing moments, more PyWavelets cases.
"""

import math
import unittest

import numpy as np
import pywt
from numpy.testing import assert_allclose

import sigstack.wavelets as wv

MODES = ("symmetric", "zero", "periodization")


class Accept(unittest.TestCase):
    def test_perfect_reconstruction_all(self) -> None:
        # AC: reconstruction error < 1e-10, every wavelet, every mode, lengths 1..70, full depth
        rng = np.random.default_rng(10)
        for name in wv.WAVELETS:
            for n in (1, 2, 3, 4, 9, 16, 33, 70):
                x = rng.standard_normal(n)
                for mode in MODES:
                    with self.subTest(name=name, n=n, mode=mode):
                        rec = wv.waverec(wv.wavedec(x, name, mode=mode), name, mode=mode)
                        self.assertLess(float(np.max(np.abs(rec[:n] - x))), 1e-10)

    def test_energy_preserved_periodization(self) -> None:
        # orthonormal filters: sum x^2 = sum of all squared coefficients (periodization, n = 2^k)
        rng = np.random.default_rng(11)
        x = rng.standard_normal(256)
        for name in ("haar", "db4", "db10", "sym7"):
            with self.subTest(name=name):
                c = wv.wavedec(x, name, mode="periodization", level=3)
                self.assertAlmostEqual(sum(float(v @ v) for v in c), float(x @ x), places=8)

    def test_vanishing_moments_kill_polynomials(self) -> None:
        # dbN / symN: a polynomial of degree < N gives zero detail coefficients
        # away from the boundary
        t = np.arange(200, dtype=np.float64) / 200.0
        for name, deg in (("db2", 1), ("db4", 3), ("sym5", 4), ("db6", 5)):
            with self.subTest(name=name):
                x = np.polyval(np.arange(1, deg + 2, dtype=np.float64), t)
                _, cd = wv.dwt(x, name, mode="zero")
                inner = cd[len(cd) // 4 : 3 * len(cd) // 4]
                self.assertLess(float(np.max(np.abs(inner))), 1e-9 * float(np.max(np.abs(x))))

    def test_long_signal_matches_pywavelets(self) -> None:
        rng = np.random.default_rng(12)
        x = rng.standard_normal(4099)
        for name in ("db8", "sym9", "sym10"):
            for mode in MODES:
                with self.subTest(name=name, mode=mode):
                    got = wv.wavedec(x, name, mode=mode)
                    ref = pywt.wavedec(x, name, mode=mode)
                    self.assertEqual(len(got), len(ref))
                    for g, r in zip(got, ref, strict=True):
                        assert_allclose(g, r, rtol=0, atol=1e-9)

    def test_all_filters_tight(self) -> None:
        for name in wv.WAVELETS:
            with self.subTest(name=name):
                ref = pywt.Wavelet(name)
                assert_allclose(wv.wavelet_filters(name).dec_hi, ref.dec_hi, rtol=0, atol=1e-10)

    def test_haar_exact(self) -> None:
        s = 1.0 / math.sqrt(2.0)
        f = wv.wavelet_filters("haar")
        assert_allclose(f.rec_lo, [s, s], rtol=0, atol=1e-15)
        assert_allclose(f.rec_hi, [s, -s], rtol=0, atol=1e-15)
        ca, cd = wv.dwt(np.array([4.0, 2.0, 5.0, 5.0]), "haar", mode="periodization")
        assert_allclose(ca, [6.0 * s, 10.0 * s], rtol=0, atol=1e-14)
        assert_allclose(cd, [2.0 * s, 0.0], rtol=0, atol=1e-14)

    def test_odd_lengths_through_levels(self) -> None:
        rng = np.random.default_rng(13)
        for n in (5, 13, 77):
            x = rng.standard_normal(n)
            for mode in MODES:
                with self.subTest(n=n, mode=mode):
                    got = wv.wavedec(x, "db2", mode=mode, level=3)
                    ref = pywt.wavedec(x, "db2", mode=mode, level=3)
                    self.assertEqual([len(c) for c in got], [len(c) for c in ref])
                    want = pywt.waverec(ref, "db2", mode=mode)
                    assert_allclose(wv.waverec(got, "db2", mode=mode), want, rtol=0, atol=1e-10)
