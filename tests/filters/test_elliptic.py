import math
import unittest

import numpy as np
import scipy.special as sp
from numpy.testing import assert_allclose

from sigstack.filters.elliptic import ellipj, ellipk


class TestElliptic(unittest.TestCase):
    def test_ellipk(self) -> None:
        # complete elliptic integral of the first kind K(m), parameter m = k^2 (SciPy's
        # convention), 0 <= m < 1
        for m in [0.0, 0.1, 0.5, 0.9, 0.99, 1 - 1e-9]:
            with self.subTest(m=m):
                got = ellipk(m)
                self.assertIsInstance(got, float)
                assert_allclose(got, sp.ellipk(m), rtol=1e-12)  # AGM converges to rounding
        assert_allclose(ellipk(0.0), math.pi / 2, rtol=1e-15)
        for m in [1.0, -0.1, 1.5]:
            with self.subTest(m=m), self.assertRaises(ValueError):
                ellipk(m)

    def test_ellipj(self) -> None:
        # (sn, cn, dn)(u | m) for real u (scalar or array), 0 <= m < 1; same shapes as u
        u = np.linspace(-3.0, 3.0, 13)
        for m in [0.0, 0.3, 0.8, 0.999]:
            with self.subTest(m=m):
                sn, cn, dn = ellipj(u, m)
                ref = sp.ellipj(u, m)
                for got, want in zip((sn, cn, dn), ref[:3], strict=True):
                    self.assertEqual(got.shape, u.shape)
                    assert_allclose(got, want, rtol=0, atol=1e-12)  # Landen converges to rounding
        sn, cn, dn = ellipj(u, 0.0)
        assert_allclose([sn, cn, dn], [np.sin(u), np.cos(u), np.ones_like(u)], rtol=0, atol=1e-15)
        s, c, d = ellipj(0.5, 0.5)
        assert_allclose([s, c, d], sp.ellipj(0.5, 0.5)[:3], rtol=0, atol=1e-12)  # scalar input
        with self.assertRaises(ValueError):
            ellipj(u, 1.0)
