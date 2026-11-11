import unittest
from typing import Any

import numpy as np
from numpy.testing import assert_allclose

from sigstack.signals.noise import pink_noise, white_noise


class TestWhiteNoise(unittest.TestCase):
    def test_real_exact_draws(self) -> None:
        # real: x = sqrt(power) * g.standard_normal(n), g = numpy.random.default_rng(rng)
        x, info = white_noise(6, power=2.0, rng=5)
        ref = np.sqrt(2.0) * np.random.default_rng(5).standard_normal(6)
        assert_allclose(x, ref, rtol=0, atol=0)  # same draws, same arithmetic
        self.assertEqual(
            (x.dtype, info.kind, info.fs, info.n_samples), (np.float64, "white_noise", 1.0, 6)
        )
        self.assertEqual((info.freqs, info.amplitudes, info.power, info.params), ((), (), 2.0, {}))

    def test_complex_exact_draws(self) -> None:
        # complex: z = g.standard_normal((2, n)); x = sqrt(power / 2) * (z[0] + 1j z[1])
        x, info = white_noise(5, power=4.0, fs=100.0, as_complex=True, rng=np.random.default_rng(3))
        z = np.random.default_rng(3).standard_normal((2, 5))
        assert_allclose(
            x, np.sqrt(2.0) * (z[0] + 1j * z[1]), rtol=0, atol=1e-15
        )  # rounding of the product only
        self.assertEqual((x.dtype, info.power, info.fs), (np.complex128, 4.0, 100.0))

    def test_same_seed_same_output(self) -> None:
        a, _ = white_noise(32, rng=11)
        b, _ = white_noise(32, rng=11)
        self.assertTrue(np.array_equal(a, b))

    def test_validation(self) -> None:
        cases1: list[Any] = [{"power": -1.0}, {"power": float("nan")}, {"fs": 0.0}]
        for kw in cases1:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                white_noise(8, **kw)
        with self.assertRaises(TypeError):
            white_noise(2.5)  # type: ignore[arg-type]


class TestPinkNoise(unittest.TestCase):
    def test_exact_construction(self) -> None:
        # w = g.standard_normal(n); X = rfft(w); f = rfftfreq(n); X[0] = 0; X[1:] /= sqrt(f[1:]);
        # y = irfft(X, n); y *= sqrt(power / mean(y^2))
        y, info = pink_noise(64, power=0.5, rng=9)
        w = np.random.default_rng(9).standard_normal(64)
        spec = np.fft.rfft(w)
        f = np.fft.rfftfreq(64)
        spec[0] = 0.0
        spec[1:] = spec[1:] / np.sqrt(f[1:])
        ref = np.fft.irfft(spec, 64)
        ref = ref * np.sqrt(0.5 / np.mean(ref**2))
        assert_allclose(y, ref, rtol=1e-12, atol=0)  # same steps; FFT rounding only
        assert_allclose(np.mean(y**2), 0.5, rtol=1e-12)  # normalised to the requested power
        assert_allclose(np.mean(y), 0.0, atol=1e-12)  # DC removed
        self.assertEqual(
            (y.dtype, info.kind, info.power, info.freqs, info.params),
            (np.float64, "pink_noise", 0.5, (), {}),
        )

    def test_odd_length_and_seed(self) -> None:
        a, _ = pink_noise(33, rng=1)
        b, _ = pink_noise(33, rng=1)
        self.assertEqual(a.shape, (33,))
        self.assertTrue(np.array_equal(a, b))

    def test_validation(self) -> None:
        for n in [0, 1]:
            with self.subTest(n=n), self.assertRaises(ValueError):
                pink_noise(n)
        with self.assertRaises(ValueError):
            pink_noise(8, power=-0.1)
