import math
import unittest
from typing import Any

import numpy as np
from numpy.testing import assert_allclose

from sigstack.metrics.bands import passband_ripple, stopband_attenuation

F = np.array([0.0, 1.0, 2.0, 3.0])


class TestBands(unittest.TestCase):
    def test_ripple(self) -> None:
        # 20 log10(max |H|) - 20 log10(min |H|) over the grid points with band[0] <= f <= band[1]
        h = np.array([1.0, 0.5, 2.0, 0.1])
        assert_allclose(passband_ripple(F, h, (0.5, 2.0)), 20 * math.log10(4.0), rtol=1e-12)
        assert_allclose(passband_ripple(F, h, (1.0, 2.0)), 20 * math.log10(4.0), rtol=1e-12)
        assert_allclose(passband_ripple(F, -1j * h, (0.0, 0.0)), 0.0, atol=1e-12)  # |H| only
        self.assertEqual(passband_ripple(F, np.array([1.0, 0.0, 1.0, 1.0]), (0.0, 1.0)), math.inf)

    def test_attenuation(self) -> None:
        # 20 log10(reference) - 20 log10(max |H| in the band); reference defaults to 1
        h = np.array([1.0, 0.01, 0.001, 0.1])
        assert_allclose(stopband_attenuation(F, h, (1.0, 2.0)), 40.0, rtol=1e-12)
        assert_allclose(stopband_attenuation(F, h, (1.0, 2.0), reference=10.0), 60.0, rtol=1e-12)
        self.assertEqual(stopband_attenuation(F, np.zeros(4), (1.0, 3.0)), math.inf)
        self.assertIsInstance(stopband_attenuation(F, h, (0.0, 3.0)), float)

    def test_validation(self) -> None:
        cases: list[Any] = [
            (passband_ripple, (F, np.ones(4), (1.2, 1.8)), {}),  # no grid point in the band
            (passband_ripple, (F, np.ones(4), (2.0, 1.0)), {}),  # band[0] > band[1]
            (passband_ripple, (F, np.ones(3), (0.0, 1.0)), {}),  # shapes differ
            (
                stopband_attenuation,
                (F, np.ones(4), (1.0, 2.0)),
                {"reference": 0.0},
            ),  # reference > 0
            (stopband_attenuation, (F, np.ones(4), (5.0, 6.0)), {}),  # no grid point in the band
        ]
        for fn, args, kw in cases:
            with self.subTest(fn=fn.__name__, args=args[2], kw=kw), self.assertRaises(ValueError):
                fn(*args, **kw)
