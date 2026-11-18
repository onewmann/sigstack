import math
import unittest
from typing import Any

import numpy as np
import scipy.signal as ss
from numpy.testing import assert_allclose

from sigstack.signals import SignalInfo, chirp, multitone


class TestDistinctTones(unittest.TestCase):
    def test_duplicate_frequencies_rejected(self) -> None:
        # +1 and -1 at the same frequency cancel: the closed-form power would be wrong
        for freqs in ([1.0, 1.0], [1.0, 2.0, 1.0]):
            with self.subTest(freqs=freqs), self.assertRaises(ValueError):
                multitone(64, 16.0, freqs, amplitudes=[1.0] * len(freqs))
        with self.assertRaises(ValueError):
            multitone(64, 16.0, [2.0, 2.0], as_complex=True)

    def test_distinct_tones_power_matches(self) -> None:
        # distinct tones on the FFT grid: the reported power equals the measured mean power
        x, info = multitone(64, 16.0, [1.0, 3.0], amplitudes=[1.0, 0.5])
        self.assertAlmostEqual(float(np.mean(x**2)), info.power, places=12)
        z, zinfo = multitone(64, 16.0, [-2.0, 2.0], amplitudes=[1.0, 0.5], as_complex=True)
        self.assertAlmostEqual(float(np.mean(np.abs(z) ** 2)), zinfo.power, places=12)


class TestSignalInfoContract(unittest.TestCase):
    def test_documented_constraints_enforced(self) -> None:
        bad: list[tuple[str, Any, Any]] = [
            ("fs negative", -1.0, 8),
            ("fs zero", 0.0, 8),
            ("fs nan", float("nan"), 8),
            ("fs inf", float("inf"), 8),
            ("fs bool", True, 8),
            ("fs str", "8", 8),
            ("n negative", 8.0, -5),
            ("n float", 8.0, 8.0),
            ("n bool", 8.0, True),
            ("fs huge int", 10**400, 8),
            ("fs 0-d array (unhashable record)", np.array(8.0), 8),
        ]
        for name, fs, n in bad:
            with self.subTest(case=name), self.assertRaises(ValueError):
                SignalInfo("x", fs, n, (), (), 0.0)

    def test_valid_records(self) -> None:
        fs: Any = np.float32(8.0)
        n: Any = np.int64(0)
        info = SignalInfo("x", fs, n, (), (), 0.0)
        self.assertEqual((info.n_samples, info.fs), (0, 8.0))


class TestChirpEndTime(unittest.TestCase):
    def test_f1_reached_at_n_over_fs(self) -> None:
        # documented convention: T = n / fs, like scipy.signal.chirp with t1 = n / fs
        n, fs = 200, 100.0
        x, _ = chirp(n, fs, 5.0, 20.0)
        t = np.arange(n) / fs
        assert_allclose(
            x, ss.chirp(t, f0=5.0, t1=n / fs, f1=20.0, method="linear", phi=0.0), atol=1e-9
        )
        self.assertTrue(math.isclose(t[-1], (n - 1) / fs))
