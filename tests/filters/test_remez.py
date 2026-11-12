import unittest
from typing import Any

import numpy as np
import scipy.signal as ss
from numpy.testing import assert_allclose

from sigstack.filters.remez import remez


def weighted_error(h: Any, bands: Any, desired: Any, weight: Any) -> float:
    """Max over all bands of weight * | |H(f)| - desired | on a dense grid (fs = 2)."""
    w = [1.0] * len(desired) if weight is None else weight
    worst = 0.0
    k = np.arange(len(h))
    for i in range(len(desired)):
        f = np.linspace(bands[2 * i], bands[2 * i + 1], 2000)
        mag = np.abs(np.exp(-1j * np.pi * np.outer(f, k)) @ h)
        worst = max(worst, float(np.max(w[i] * np.abs(mag - desired[i]))))
    return worst


class TestRemez(unittest.TestCase):
    def test_matches_scipy(self) -> None:
        # equiripple (minimax) linear-phase filter by the Remez exchange;
        # desired and weight constant
        # per band; bands as edge pairs in [0, fs / 2]
        cases: list[Any] = [
            (31, [0, 0.3, 0.4, 1.0], [1, 0], None),  # lowpass
            (31, [0, 0.3, 0.4, 1.0], [1, 0], [1, 10]),  # weighted
            (32, [0, 0.3, 0.4, 1.0], [1, 0], None),  # even length (zero at Nyquist)
            (41, [0, 0.2, 0.3, 0.5, 0.6, 1.0], [0, 1, 0], None),  # bandpass
            (45, [0, 0.25, 0.35, 1.0], [0, 1], None),  # highpass, odd length
        ]
        for numtaps, bands, desired, weight in cases:
            with self.subTest(numtaps=numtaps, bands=bands, weight=weight):
                got = remez(numtaps, bands, desired, weight=weight)
                ref = ss.remez(numtaps, bands, desired, weight=weight, fs=2.0)
                self.assertEqual((got.shape, got.dtype), ((numtaps,), np.float64))
                assert_allclose(got, got[::-1], rtol=0, atol=1e-12)  # linear phase: symmetric
                assert_allclose(got, ref, rtol=0, atol=1e-4)  # same optimum; grids differ slightly
                e_got = weighted_error(got, bands, desired, weight)
                e_ref = weighted_error(ref, bands, desired, weight)
                self.assertLessEqual(e_got, e_ref * 1.001)  # minimax: no worse than SciPy's

    def test_fs(self) -> None:
        got = remez(31, [0, 1200, 1600, 4000], [1, 0], fs=8000.0)
        ref = ss.remez(31, [0, 1200, 1600, 4000], [1, 0], fs=8000.0)
        assert_allclose(got, ref, rtol=0, atol=1e-4)

    def test_validation(self) -> None:
        cases: list[Any] = [
            ((31, [0, 0.3, 0.4], [1, 0]), {}),  # an even number of band edges
            ((31, [0, 0.4, 0.3, 1.0], [1, 0]), {}),  # edges increasing
            ((31, [0, 0.3, 0.25, 1.0], [1, 0]), {}),  # bands must not overlap
            ((31, [0, 0.3, 0.4, 1.5], [1, 0]), {}),  # edges within [0, fs / 2]
            ((31, [0, 0.3, 0.4, 1.0], [1]), {}),  # one desired value per band
            ((31, [0, 0.3, 0.4, 1.0], [1, 0]), {"weight": [1]}),  # one weight per band
            ((31, [0, 0.3, 0.4, 1.0], [1, 0]), {"weight": [1, 0]}),  # weights > 0
            ((32, [0, 0.3, 0.4, 1.0], [0, 1]), {}),  # even length cannot pass Nyquist
            ((2, [0, 0.3, 0.4, 1.0], [1, 0]), {}),  # numtaps >= 3
        ]
        for args, kw in cases:
            with self.subTest(args=args, kw=kw), self.assertRaises(ValueError):
                remez(*args, **kw)
