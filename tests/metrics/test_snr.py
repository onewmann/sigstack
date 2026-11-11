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

    def test_reference(self) -> None:
        # noise = observed - reference, then snr_db(reference, noise); shapes must match
        got = snr_db_ref([1.0, 2.0], [1.1, 1.9])
        assert_allclose(got, 10 * math.log10(250.0), rtol=1e-9)  # 1.1 - 1.0 is not exact in binary
        with self.assertRaises(ValueError):
            snr_db_ref([1.0, 2.0], [1.0])
