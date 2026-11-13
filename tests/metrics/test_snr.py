import math
import unittest

import numpy as np
from numpy.testing import assert_allclose

from sigstack.metrics.snr import snr_db, snr_db_ref


class TestSnr(unittest.TestCase):
    def test_real(self) -> None:
        # 10 log10(P_s / P_n), P = mean |x|^2 over all elements
        got = snr_db([1.0, 1.0, 1.0, 1.0], [0.1, -0.1, 0.1, -0.1])
        assert_allclose(got, 20.0, rtol=1e-12)  # power ratio 100, rounding only
        self.assertIsInstance(got, float)

    def test_complex_and_multichannel(self) -> None:
        assert_allclose(snr_db([1j, -1j], [0.5, 0.5]), 10 * math.log10(4.0), rtol=1e-12)
        s, n = np.ones((2, 3)), np.full((2, 3), 0.5)  # powers over all elements
        assert_allclose(snr_db(s, n), 10 * math.log10(4.0), rtol=1e-12)

    def test_zero_powers(self) -> None:
        self.assertEqual(snr_db([1.0], [0.0]), math.inf)
        self.assertEqual(snr_db([0.0], [1.0]), -math.inf)
        with self.assertRaises(ValueError):
            snr_db([0.0], [0.0])  # both powers zero
        with self.assertRaises(ValueError):
            snr_db([], [1.0])  # empty input

    def test_integer_input_does_not_overflow(self) -> None:
        # int16 PCM: |x|^2 in int16 would overflow; the result must equal the float64 computation
        s = (np.sin(np.arange(1000) * 0.05) * 20000).astype(np.int16)
        n = (np.random.default_rng(0).standard_normal(1000) * 200).astype(np.int16)
        sf, nf = s.astype(np.float64), n.astype(np.float64)
        self.assertAlmostEqual(snr_db(s, n), snr_db(sf, nf), places=10)
        obs = (s.astype(np.int32) + n).clip(-32768, 32767).astype(np.int16)
        want = snr_db_ref(sf, obs.astype(np.float64))
        self.assertAlmostEqual(snr_db_ref(s, obs), want, places=10)
        # uint8: 20^2 would wrap, and observed - reference = -2 would wrap to 254
        ref8 = np.array([20, 30, 40], dtype=np.uint8)
        obs8 = np.array([18, 30, 40], dtype=np.uint8)
        self.assertAlmostEqual(snr_db_ref(ref8, obs8), 10 * math.log10(2900 / 4), places=12)

    def test_reference(self) -> None:
        # noise = observed - reference, then snr_db(reference, noise); shapes must match
        got = snr_db_ref([1.0, 2.0], [1.1, 1.9])
        assert_allclose(got, 10 * math.log10(250.0), rtol=1e-9)  # 1.1 - 1.0 is not exact in binary
        with self.assertRaises(ValueError):
            snr_db_ref([1.0, 2.0], [1.0])
