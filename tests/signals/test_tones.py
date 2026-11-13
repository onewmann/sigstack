import math
import unittest
from typing import Any

import numpy as np
from numpy.testing import assert_allclose

from sigstack.signals.tones import chirp, impulse, multitone

S2 = math.sqrt(2.0)


class TestMultitone(unittest.TestCase):
    def test_single_real_tone(self) -> None:
        # x[k] = sum_i a_i cos(2 pi f_i k / fs + phi_i); phases default to 0 when rng is None
        x, info = multitone(8, 8.0, [1.0], amplitudes=[2.0])
        self.assertEqual((x.shape, x.dtype), ((8,), np.float64))
        # exact up to cos() rounding
        assert_allclose(x, [2, S2, 0, -S2, -2, -S2, 0, S2], rtol=0, atol=1e-12)
        self.assertEqual((info.kind, info.fs, info.n_samples), ("multitone", 8.0, 8))
        self.assertEqual(
            (info.freqs, info.amplitudes, info.power, info.params), ((1.0,), (2.0,), 2.0, {})
        )

    def test_complex_tone(self) -> None:
        # as_complex: x[k] = sum_i a_i exp(j (2 pi f_i k / fs + phi_i)); power = sum a_i^2
        x, info = multitone(4, 4.0, [1.0], as_complex=True)
        self.assertEqual(x.dtype, np.complex128)
        assert_allclose(x, [1, 1j, -1, -1j], rtol=0, atol=1e-12)  # cos/sin rounding only
        self.assertEqual((info.amplitudes, info.power), ((1.0,), 1.0))
        y, _ = multitone(
            4, 4.0, [-1.0], as_complex=True
        )  # negative frequencies are allowed when complex
        assert_allclose(y, [1, -1j, -1, 1j], rtol=0, atol=1e-12)

    def test_power_two_tones(self) -> None:
        # real power = sum a_i^2 / 2; the sample mean power equals it over whole periods
        x, info = multitone(8, 8.0, [1.0, 2.0], amplitudes=[1.0, 3.0])
        self.assertEqual(info.power, 5.0)
        assert_allclose(
            np.mean(x**2), 5.0, rtol=0, atol=1e-12
        )  # whole periods: exact up to rounding
        self.assertAlmostEqual(float(x[0]), 4.0)  # phases 0: x[0] = sum of amplitudes

    def test_explicit_phases(self) -> None:
        x, _ = multitone(4, 4.0, [1.0], phases=[math.pi / 2])
        assert_allclose(
            x, [0, -1, 0, 1], rtol=0, atol=1e-12
        )  # cos(pi k / 2 + pi / 2) = -sin(pi k / 2)

    def test_seeded_random_phases(self) -> None:
        # phases None and rng given: phases = rng.uniform(0, 2 pi, size=len(freqs)),
        # rng resolved with numpy.random.default_rng
        a, _ = multitone(16, 16.0, [1.0, 3.0], rng=7)
        b, _ = multitone(16, 16.0, [1.0, 3.0], rng=7)
        c, _ = multitone(16, 16.0, [1.0, 3.0], rng=8)
        assert_allclose(a, b, rtol=0, atol=0)  # same seed: identical output
        self.assertFalse(np.allclose(a, c))
        ph = np.random.default_rng(7).uniform(0.0, 2 * np.pi, size=2)
        k = np.arange(16)
        ref = np.cos(2 * np.pi * 1.0 * k / 16 + ph[0]) + np.cos(2 * np.pi * 3.0 * k / 16 + ph[1])
        assert_allclose(a, ref, rtol=0, atol=1e-12)  # same formula, rounding only
        d, _ = multitone(16, 16.0, [1.0, 3.0], rng=np.random.default_rng(7))
        assert_allclose(a, d, rtol=0, atol=0)  # a Generator works like its seed

    def test_validation(self) -> None:
        cases1: list[Any] = [
            ((8, 8.0, [0.0]), {}, ValueError),  # real: 0 < f < fs/2
            ((8, 8.0, [4.0]), {}, ValueError),
            ((8, 8.0, [4.0]), {"as_complex": True}, ValueError),  # complex: |f| < fs/2
            ((8, 8.0, [1.0]), {"amplitudes": [1.0, 2.0]}, ValueError),
            ((8, 8.0, [1.0]), {"phases": [0.0, 1.0]}, ValueError),
            ((8, 8.0, []), {}, ValueError),
            ((0, 8.0, [1.0]), {}, ValueError),
            ((True, 8.0, [1.0]), {}, TypeError),
            ((8.0, 8.0, [1.0]), {}, TypeError),
            ((8, 0.0, [1.0]), {}, ValueError),
            ((8, float("inf"), [1.0]), {}, ValueError),
        ]
        for args, kw, exc in cases1:
            with self.subTest(args=args, kw=kw), self.assertRaises(exc):
                multitone(*args, **kw)


class TestChirp(unittest.TestCase):
    def test_linear_values(self) -> None:
        # T = n / fs, t = k / fs, theta = 2 pi (f0 t + (f1 - f0) t^2 / (2 T)) + phase;
        # x = A cos(theta)
        x, info = chirp(4, 4.0, 0.0, 2.0)
        c = math.cos(math.pi / 8)
        assert_allclose(x, [1, c, 0, -c], rtol=0, atol=1e-12)  # rounding only
        self.assertEqual(
            (info.kind, info.freqs, info.amplitudes, info.power), ("chirp", (0.0, 2.0), (1.0,), 0.5)
        )
        self.assertEqual(info.params, {"duration": 1.0, "rate": 2.0})

    def test_complex_amplitude_phase(self) -> None:
        x, info = chirp(4, 4.0, 0.0, 2.0, amplitude=2.0, phase=math.pi / 2, as_complex=True)
        k = np.arange(4) / 4.0
        assert_allclose(x, 2.0 * np.exp(1j * (2 * np.pi * k**2 + np.pi / 2)), rtol=0, atol=1e-12)
        self.assertEqual((x.dtype, info.power, info.amplitudes), (np.complex128, 4.0, (2.0,)))

    def test_logarithmic(self) -> None:
        # method "logarithmic": theta = 2 pi f0 T / ln(f1/f0) ((f1/f0)^(t/T) - 1) + phase;
        # params {"duration": T}
        x, info = chirp(8, 8.0, 1.0, 3.0, method="logarithmic")
        t = np.arange(8) / 8.0
        ref = np.cos(2 * np.pi * 1.0 * 1.0 / np.log(3.0) * (3.0**t - 1.0))
        assert_allclose(x, ref, rtol=0, atol=1e-12)  # same formula, rounding only
        self.assertEqual(info.params, {"duration": 1.0})

    def test_validation(self) -> None:
        cases2: list[Any] = [
            ((8, 8.0, 0.0, 5.0), {}),  # real: 0 <= f <= fs/2
            ((8, 8.0, -1.0, 2.0), {}),
            ((8, 8.0, -5.0, 2.0), {"as_complex": True}),  # complex: |f| <= fs/2
            ((8, 8.0, 1.0, 2.0), {"method": "quadratic"}),
            ((8, 8.0, 0.0, 2.0), {"method": "logarithmic"}),  # log: f0, f1 > 0 and f0 != f1
            ((8, 8.0, 2.0, 2.0), {"method": "logarithmic"}),
        ]
        for args, kw in cases2:
            with self.subTest(args=args, kw=kw), self.assertRaises(ValueError):
                chirp(*args, **kw)
        chirp(8, 8.0, -4.0, 4.0, as_complex=True)  # the band edges themselves are allowed


class TestImpulse(unittest.TestCase):
    def test_impulse(self) -> None:
        x, info = impulse(4, index=1, amplitude=3.0)
        assert_allclose(x, [0, 3, 0, 0], rtol=0, atol=0)
        self.assertEqual(
            (x.dtype, info.kind, info.fs, info.freqs, info.amplitudes),
            (np.float64, "impulse", 1.0, (), (3.0,)),
        )
        self.assertEqual(
            (info.power, info.params), (9.0 / 4, {"index": 1.0, "energy": 9.0})
        )  # power = A^2 / n
        y, info2 = impulse(2, fs=10.0, as_complex=True)
        self.assertEqual((y.dtype, y.tolist(), info2.fs), (np.complex128, [1, 0], 10.0))

    def test_validation(self) -> None:
        cases3: list[Any] = [{"index": 4}, {"index": -1}]
        for kw in cases3:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                impulse(4, **kw)
