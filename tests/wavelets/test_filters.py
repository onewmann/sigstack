import math
import unittest

import numpy as np
import pywt
from numpy.testing import assert_allclose

from sigstack.wavelets.filters import WAVELETS, WaveletFilters, wavelet_filters

EXPECTED_NAMES = ("haar", *(f"db{n}" for n in range(1, 11)), *(f"sym{n}" for n in range(2, 11)))


class TestWaveletFilters(unittest.TestCase):
    def test_names(self) -> None:
        # haar, db1..db10, sym2..sym10, in this order, as a tuple of str
        self.assertEqual(WAVELETS, EXPECTED_NAMES)

    def test_match_pywavelets(self) -> None:
        # spectral factorisation of P(y); dbN: all zeros inside the unit circle;
        # symN: by the selection table
        for name in WAVELETS:
            with self.subTest(name=name):
                f = wavelet_filters(name)
                self.assertIsInstance(f, WaveletFilters)
                self.assertEqual(f._fields, ("dec_lo", "dec_hi", "rec_lo", "rec_hi"))
                ref = pywt.Wavelet(name)
                refs = (ref.dec_lo, ref.dec_hi, ref.rec_lo, ref.rec_hi)
                for got, want in zip(f, refs, strict=True):
                    self.assertEqual((got.shape, got.dtype), ((len(want),), np.float64))
                    assert_allclose(got, want, rtol=0, atol=1e-10)

    def test_orthonormal(self) -> None:
        # sum h = sqrt(2), sum h^2 = 1, sum h[n] h[n + 2k] = 0 for k != 0
        for name in WAVELETS:
            with self.subTest(name=name):
                h = wavelet_filters(name).rec_lo
                self.assertAlmostEqual(float(h.sum()), math.sqrt(2.0), places=12)
                for k in range(len(h) // 2):
                    want = 1.0 if k == 0 else 0.0
                    self.assertAlmostEqual(float(h[: len(h) - 2 * k] @ h[2 * k :]), want, places=12)

    def test_vanishing_moments(self) -> None:
        # order N: sum_n n^p g[n] = 0 for p = 0..N-1 (g = rec_hi), relative to the scale of the sum
        for name in WAVELETS:
            order = 1 if name == "haar" else int(name.lstrip("dbsym"))
            with self.subTest(name=name):
                g = wavelet_filters(name).rec_hi
                n = np.arange(len(g), dtype=np.float64)
                for p in range(order):
                    scale = float(np.sum(np.abs(n**p * g)))
                    self.assertLess(abs(float(np.sum(n**p * g))), 1e-9 * scale)

    def test_fresh_arrays(self) -> None:
        a = wavelet_filters("db4")
        a.rec_lo[:] = 0.0
        self.assertGreater(float(np.abs(wavelet_filters("db4").rec_lo).sum()), 1.0)

    def test_unknown_name(self) -> None:
        for name in ("db0", "db11", "sym1", "sym11", "coif1", "Haar", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                wavelet_filters(name)
