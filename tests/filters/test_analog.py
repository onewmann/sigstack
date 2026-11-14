import unittest
from typing import Any

import numpy as np
import scipy.signal as ss
from numpy.testing import assert_allclose

from sigstack.filters.analog import buttap, cheb1ap, cheb2ap, ellipap


def same_set(tc: unittest.TestCase, got: Any, ref: Any, atol: float) -> None:
    """Equal multisets of complex numbers (order free)."""
    got, ref = np.asarray(got, dtype=complex), np.asarray(ref, dtype=complex)
    # SciPy returns a 0-d pole array for n = 1 (fix 2026-11-13)
    got, ref = np.atleast_1d(got), np.atleast_1d(ref)
    tc.assertEqual(got.shape, ref.shape)
    for a, b in ((got, ref), (ref, got)):
        for v in a:
            tc.assertLess(float(np.min(np.abs(b - v))), atol, (v, b))


class TestAnalog(unittest.TestCase):
    # analog lowpass prototypes (z, p, k) with SciPy's conventions: cutoff 1 rad/s
    # (Butterworth -3 dB, Chebyshev I and elliptic: passband edge at -rp dB,
    # Chebyshev II: stopband edge at -rs dB)
    def check(self, got: Any, ref: Any, atol: float) -> None:
        z, p, k = got
        self.assertEqual((np.asarray(z).dtype, np.asarray(p).dtype), (np.complex128, np.complex128))
        self.assertIsInstance(k, float)
        same_set(self, z, ref[0], atol)
        same_set(self, p, ref[1], atol)
        assert_allclose(k, ref[2], rtol=1e-9)
        self.assertTrue(np.all(np.real(p) < 0))  # stable

    def test_buttap(self) -> None:
        for n in range(1, 9):
            with self.subTest(n=n):
                self.check(buttap(n), ss.buttap(n), 1e-12)
                assert_allclose(np.abs(buttap(n)[1]), 1.0, atol=1e-12)  # poles on the unit circle

    def test_cheb1ap(self) -> None:
        for n in range(1, 8):
            for rp in (0.5, 1.0, 3.0):
                with self.subTest(n=n, rp=rp):
                    self.check(cheb1ap(n, rp), ss.cheb1ap(n, rp), 1e-10)

    def test_cheb2ap(self) -> None:
        for n in range(1, 8):
            for rs in (20.0, 40.0, 60.0):
                with self.subTest(n=n, rs=rs):
                    self.check(cheb2ap(n, rs), ss.cheb2ap(n, rs), 1e-9)

    def test_ellipap(self) -> None:
        for n in range(1, 8):
            for rp, rs in ((0.5, 40.0), (1.0, 60.0), (3.0, 30.0)):
                with self.subTest(n=n, rp=rp, rs=rs):
                    self.check(ellipap(n, rp, rs), ss.ellipap(n, rp, rs), 1e-7)

    def test_validation(self) -> None:
        cases: list[Any] = [
            (buttap, (0,)),
            (cheb1ap, (3, 0.0)),
            (cheb2ap, (3, -1.0)),
            (ellipap, (3, 1.0, 0.5)),  # rs > rp for ellipap
            (ellipap, (0, 1.0, 40.0)),
        ]
        for fn, args in cases:
            with self.subTest(fn=fn.__name__, args=args), self.assertRaises(ValueError):
                fn(*args)

    # review: restores the shape assertion that the n = 1 atleast_1d fix in same_set weakened
    def test_order_one_is_one_dimensional(self) -> None:
        # SciPy returns a 0-d pole array for ellipap(1, ...) (why same_set applies atleast_1d);
        # sigstack keeps its documented contract: 1-D complex arrays for every n, n = 1 included
        cases: list[Any] = [
            (buttap, (1,)),
            (cheb1ap, (1, 1.0)),
            (cheb2ap, (1, 40.0)),
            (ellipap, (1, 1.0, 40.0)),
        ]
        for fn, args in cases:
            with self.subTest(fn=fn.__name__):
                z, p, k = fn(*args)
                self.assertEqual(
                    (z.shape, p.shape, z.dtype, p.dtype), ((0,), (1,), np.complex128, np.complex128)
                )
                self.assertIsInstance(k, float)
