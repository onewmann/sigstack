import math
import unittest
from typing import Any

import numpy as np
from numpy.testing import assert_allclose

from sigstack.signals.modulation import am, fm


class TestAM(unittest.TestCase):
    def test_real_values_and_info(self) -> None:
        # x = A (1 + m cos(2 pi f_mod t)) cos(2 pi fc t), t = k / fs
        x, info = am(8, 8.0, 2.0, 1.0, depth=0.5)
        assert_allclose(x, [1.5, 0, -1, 0, 0.5, 0, -1, 0], rtol=0, atol=1e-12)  # rounding only
        self.assertEqual((x.dtype, info.kind, info.fs, info.n_samples), (np.float64, "am", 8.0, 8))
        # carrier and the two side lines; power = A^2 / 2 (1 + m^2 / 2)
        self.assertEqual((info.freqs, info.amplitudes), ((1.0, 2.0, 3.0), (0.25, 1.0, 0.25)))
        self.assertEqual((info.power, info.params), (0.5625, {"depth": 0.5, "f_mod": 1.0}))
        assert_allclose(np.mean(x**2), info.power, rtol=0, atol=1e-12)  # whole periods: exact

    def test_complex(self) -> None:
        # complex: x = A (1 + m cos(2 pi f_mod t)) exp(j 2 pi fc t); power = A^2 (1 + m^2 / 2)
        x, info = am(8, 8.0, 2.0, 1.0, depth=0.5, amplitude=2.0, as_complex=True)
        k = np.arange(8)
        ref = 2.0 * (1 + 0.5 * np.cos(2 * np.pi * k / 8)) * np.exp(2j * np.pi * 2.0 * k / 8)
        assert_allclose(x, ref, rtol=0, atol=1e-12)
        self.assertEqual(
            (x.dtype, info.power, info.amplitudes), (np.complex128, 4.5, (0.5, 2.0, 0.5))
        )

    def test_validation(self) -> None:
        am_cases: list[Any] = [
            ((8, 8.0, 2.0, 3.0), {}),  # real: 0 < f_mod < fc
            ((8, 8.0, 3.0, 1.0), {}),  # real: fc + f_mod < fs/2
            ((8, 8.0, 2.0, 0.0), {}),
            ((8, 8.0, 2.0, 1.0), {"depth": 1.5}),  # 0 <= depth <= 1
            ((8, 8.0, -3.0, 1.0), {"as_complex": True}),  # complex: |fc| + f_mod < fs/2
        ]
        for args, kw in am_cases:
            with self.subTest(args=args, kw=kw), self.assertRaises(ValueError):
                am(*args, **kw)
        am(8, 8.0, -2.0, 1.0, as_complex=True)


class TestFM(unittest.TestCase):
    def test_real_values_and_info(self) -> None:
        # x = A cos(2 pi fc t + beta sin(2 pi f_mod t)), beta = deviation / f_mod
        x, info = fm(4, 4.0, 1.0, 1.0, deviation=0.5)
        s = math.sin(0.5)
        assert_allclose(x, [1, -s, -1, -s], rtol=0, atol=1e-12)  # rounding only
        self.assertEqual(
            (x.dtype, info.kind, info.freqs, info.amplitudes, info.power),
            (np.float64, "fm", (1.0,), (1.0,), 0.5),
        )
        self.assertEqual(
            info.params, {"deviation": 0.5, "f_mod": 1.0, "beta": 0.5, "carson_bandwidth": 3.0}
        )

    def test_complex(self) -> None:
        x, info = fm(8, 8.0, -1.0, 1.0, deviation=1.0, amplitude=3.0, as_complex=True)
        t = np.arange(8) / 8.0
        ref = 3.0 * np.exp(1j * (2 * np.pi * -1.0 * t + 1.0 * np.sin(2 * np.pi * t)))
        assert_allclose(x, ref, rtol=0, atol=1e-12)
        self.assertEqual((x.dtype, info.power), (np.complex128, 9.0))

    def test_validation(self) -> None:
        fm_cases: list[Any] = [
            ((8, 8.0, 1.0, 1.0), {"deviation": 1.0}),  # real: fc - deviation > 0
            ((8, 8.0, 3.0, 1.0), {"deviation": 1.0}),  # real: fc + deviation < fs/2
            ((8, 8.0, 2.0, 0.0), {"deviation": 0.5}),  # f_mod > 0
            ((8, 8.0, 2.0, 1.0), {"deviation": -0.5}),  # deviation >= 0
            ((8, 8.0, 3.5, 1.0), {"deviation": 0.5, "as_complex": True}),
        ]  # complex: |fc| + dev < fs/2
        for args, kw in fm_cases:
            with self.subTest(args=args, kw=kw), self.assertRaises(ValueError):
                fm(*args, **kw)
        with self.assertRaises(TypeError):
            fm(8, 8.0, 2.0, 1.0)  # type: ignore[call-arg]   # deviation is keyword-only and required
