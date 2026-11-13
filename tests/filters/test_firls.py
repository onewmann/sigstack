import unittest
from typing import Any

import numpy as np
import scipy.signal as ss
from numpy.testing import assert_allclose

from sigstack.filters.firls import firls


class TestFirls(unittest.TestCase):
    def test_matches_scipy(self) -> None:
        # minimise sum_b w_b int_band (A(f) - D(f))^2 df, A = amplitude of the odd-length
        # linear-phase filter, D linear between the two edge values of each band;
        # bands as edge pairs in [0, fs / 2]
        cases: list[Any] = [
            (31, [0, 0.3, 0.4, 1.0], [1, 1, 0, 0], None),
            (31, [0, 0.3, 0.4, 1.0], [1, 1, 0, 0], [1, 10]),
            (41, [0, 0.2, 0.3, 0.5, 0.6, 1.0], [0, 0, 1, 1, 0, 0], [5, 1, 5]),
            (25, [0, 1.0], [0, 1.0], None),  # linear ramp
            (33, [0, 0.25, 0.25, 0.5, 0.6, 1.0], [1, 1, 0.5, 0.5, 0, 0], None),  # shared edge
        ]
        for numtaps, bands, desired, weight in cases:
            with self.subTest(numtaps=numtaps, bands=bands, weight=weight):
                got = firls(numtaps, bands, desired, weight=weight)
                ref = ss.firls(numtaps, bands, desired, weight=weight)
                self.assertEqual((got.shape, got.dtype), ((numtaps,), np.float64))
                assert_allclose(got, ref, rtol=0, atol=1e-8)  # same optimum; linear-solve rounding

    def test_fs_and_symmetry(self) -> None:
        got = firls(31, [0, 1000, 1500, 4000], [1, 1, 0, 0], fs=8000.0)
        ref = ss.firls(31, [0, 1000, 1500, 4000], [1, 1, 0, 0], fs=8000.0)
        assert_allclose(got, ref, rtol=0, atol=1e-8)
        assert_allclose(got, got[::-1], rtol=0, atol=1e-14)  # linear phase: symmetric

    def test_validation(self) -> None:
        cases: list[Any] = [
            ((30, [0, 0.3, 0.4, 1.0], [1, 1, 0, 0]), {}),  # numtaps must be odd
            ((31, [0, 0.3, 0.4], [1, 1, 0]), {}),  # an even number of band edges
            ((31, [0, 0.4, 0.3, 1.0], [1, 1, 0, 0]), {}),  # edges non-decreasing
            ((31, [0, 0.3, 0.4, 1.2], [1, 1, 0, 0]), {}),  # edges within [0, fs / 2]
            ((31, [0, 0.3, 0.4, 1.0], [1, 1, 0]), {}),  # one desired value per edge
            ((31, [0, 0.3, 0.4, 1.0], [1, 1, 0, 0]), {"weight": [1, 2, 3]}),  # one weight per band
            ((31, [0, 0.3, 0.4, 1.0], [1, 1, 0, 0]), {"weight": [1, -1]}),  # weights > 0
        ]
        for args, kw in cases:
            with self.subTest(args=args, kw=kw), self.assertRaises(ValueError):
                firls(*args, **kw)
