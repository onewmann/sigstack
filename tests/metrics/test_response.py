import math
import unittest
from typing import Any

import numpy as np
from numpy.testing import assert_allclose

from sigstack.metrics.response import freq_response, group_delay

MA4 = [0.25, 0.25, 0.25, 0.25]


class TestFreqResponse(unittest.TestCase):
    def test_first_order_iir(self) -> None:
        # H = B(e^{jw}) / A(e^{jw}), B(z) = sum_k b_k z^-k; grid w_k = pi k / n, k = 0 .. n - 1
        f, h = freq_response([1.0], [1.0, -0.5], n=2)
        assert_allclose(f, [0.0, math.pi / 2], rtol=0, atol=1e-15)  # default fs = 2 pi: rad/sample
        assert_allclose(h, [2.0, 0.8 - 0.4j], rtol=0, atol=1e-12)  # 1 / (1 + 0.5 j), rounding only
        self.assertEqual((f.dtype, h.dtype), (np.float64, np.complex128))

    def test_moving_average_and_fs(self) -> None:
        f, h = freq_response(MA4, n=4, fs=1000.0)
        assert_allclose(f, [0.0, 125.0, 250.0, 375.0], rtol=0, atol=1e-12)  # f = w fs / (2 pi)
        k = np.arange(4)
        ref = np.array([np.sum(0.25 * np.exp(-1j * w * k)) for w in np.pi * k / 4])
        assert_allclose(h, ref, rtol=0, atol=1e-12)  # same sum, rounding only
        assert_allclose(abs(h[2]), 0.0, atol=1e-15)  # zero of the moving average at w = pi / 2

    def test_whole(self) -> None:
        # whole=True: w_k = 2 pi k / n
        f, h = freq_response([1.0], [1.0, -0.5], n=4, whole=True)
        assert_allclose(f, [0.0, math.pi / 2, math.pi, 3 * math.pi / 2], rtol=0, atol=1e-15)
        assert_allclose(h[2], 1 / 1.5, rtol=1e-12)
        assert_allclose(h[3], np.conj(h[1]), rtol=1e-12)  # real coefficients: conjugate symmetry

    def test_a0_scaling(self) -> None:
        _, h1 = freq_response([1.0, 1.0], [2.0, -1.0], n=8)
        _, h2 = freq_response([0.5, 0.5], [1.0, -0.5], n=8)
        assert_allclose(h1, h2, rtol=1e-12)  # only the ratio B / A matters

    def test_validation(self) -> None:
        cases: list[Any] = [
            (([],), {}),  # b empty
            (([1.0], []), {}),  # a empty
            (([1.0], [0.0, 1.0]), {}),  # a[0] == 0
            (([1.0],), {"n": 0}),  # n >= 1
            (([1.0],), {"fs": 0.0}),  # fs > 0 and finite
            (([math.nan],), {}),  # finite coefficients
            (([[1.0, 2.0]],), {}),  # 1-D coefficients
        ]
        for args, kw in cases:
            with self.subTest(args=args, kw=kw), self.assertRaises(ValueError):
                freq_response(*args, **kw)
        with self.assertRaises(TypeError):
            freq_response([1.0], n=2.0)  # type: ignore[arg-type]


class TestGroupDelay(unittest.TestCase):
    def test_first_order_iir(self) -> None:
        # gd = -d arg(H) / dw in samples; for 1 / (1 - a e^{-jw}):
        # gd = (a cos w - a^2) / (1 - 2 a cos w + a^2)
        f, gd = group_delay([1.0], [1.0, -0.5], n=4, whole=True)
        assert_allclose(f, [0.0, math.pi / 2, math.pi, 3 * math.pi / 2], rtol=0, atol=1e-15)
        assert_allclose(gd, [1.0, -0.2, -1 / 3, -0.2], rtol=0, atol=1e-12)  # rounding only
        self.assertEqual(gd.dtype, np.float64)

    def test_linear_phase_fir_and_zeros(self) -> None:
        # symmetric FIR of length N: constant delay (N - 1) / 2; where H = 0 the delay is nan
        _, gd = group_delay(MA4, n=8)
        expect = np.full(8, 1.5)
        expect[4] = np.nan  # w = pi / 2 is a zero of the moving average
        assert_allclose(gd, expect, rtol=0, atol=1e-9)  # nan only where expected

    def test_pure_delay_and_fs(self) -> None:
        f, gd = group_delay([0.0, 0.0, 1.0], n=5, fs=10.0)
        assert_allclose(f, [0.0, 1.0, 2.0, 3.0, 4.0], rtol=0, atol=1e-12)
        assert_allclose(gd, np.full(5, 2.0), rtol=0, atol=1e-12)  # in samples, whatever fs is
