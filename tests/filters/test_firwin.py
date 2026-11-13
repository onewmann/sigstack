import unittest
from typing import Any

import numpy as np
import scipy.signal as ss
from numpy.testing import assert_allclose

from sigstack.filters.firwin import firwin

SCIPY_WINDOW = {
    "rectangular": "boxcar",
    "hann": "hann",
    "hamming": "hamming",
    "blackman": "blackman",
}


class TestFirwin(unittest.TestCase):
    def test_matches_scipy(self) -> None:
        # h = symmetric window * ideal response (sum of sinc differences);
        # cutoffs relative to fs / 2;
        # pass_zero: the band starting at 0 is a passband; bands alternate
        cases: list[Any] = [
            (31, 0.3, True),  # lowpass
            (31, 0.3, False),  # highpass
            (30, 0.3, True),  # even length lowpass
            (41, [0.2, 0.5], False),  # bandpass
            (41, [0.2, 0.5], True),  # bandstop
            (51, [0.1, 0.3, 0.6], True),  # multiband
        ]
        for numtaps, cutoff, pass_zero in cases:
            for name, sname in SCIPY_WINDOW.items():
                with self.subTest(numtaps=numtaps, cutoff=cutoff, pass_zero=pass_zero, window=name):
                    got = firwin(numtaps, cutoff, window=name, pass_zero=pass_zero)
                    ref = ss.firwin(numtaps, cutoff, window=sname, pass_zero=pass_zero)
                    self.assertEqual((got.shape, got.dtype), ((numtaps,), np.float64))
                    assert_allclose(got, ref, rtol=1e-10, atol=1e-12)  # same formula, rounding

    def test_kaiser_and_fs(self) -> None:
        # window parameters are passed on to get_window; cutoffs in the units of fs
        got = firwin(61, 1000.0, window="kaiser", beta=8.0, fs=8000.0)
        ref = ss.firwin(61, 1000.0, window=("kaiser", 8.0), fs=8000.0)
        assert_allclose(got, ref, rtol=1e-10, atol=1e-12)

    def test_scale(self) -> None:
        # scale=True: |H| = 1 at the centre of the first passband (0 if it starts at 0,
        # Nyquist if it
        # ends there); scale=False: the windowed ideal response as is
        k = np.arange(31)
        assert_allclose(np.sum(firwin(31, 0.3, window="hann")), 1.0, rtol=1e-12)
        h = firwin(31, 0.3, window="hann", pass_zero=False)
        assert_allclose(abs(np.sum(h * (-1.0) ** k)), 1.0, rtol=1e-12)
        h = firwin(31, [0.2, 0.5], window="hann", pass_zero=False)
        assert_allclose(abs(np.sum(h * np.exp(-1j * np.pi * 0.35 * k))), 1.0, rtol=1e-12)
        got = firwin(31, 0.3, window="rectangular", scale=False)
        assert_allclose(
            got, ss.firwin(31, 0.3, window="boxcar", scale=False), rtol=1e-10, atol=1e-12
        )

    def test_validation(self) -> None:
        cases: list[Any] = [
            ((30, 0.3), {"pass_zero": False}),  # even length cannot pass Nyquist
            ((30, [0.2, 0.5]), {}),  # bandstop passes Nyquist: even length refused too
            ((31, 0.0), {}),  # cutoffs strictly inside (0, fs / 2)
            ((31, 1.0), {}),
            ((31, [0.5, 0.2]), {}),  # strictly increasing
            ((31, []), {}),
            ((31, 0.3), {"window": "nope"}),
            ((0, 0.3), {}),
            ((31, 300.0), {"fs": 500.0}),  # above fs / 2
        ]
        for args, kw in cases:
            with self.subTest(args=args, kw=kw), self.assertRaises(ValueError):
                firwin(*args, **kw)
        with self.assertRaises(TypeError):
            firwin(31.0, 0.3)  # type: ignore[arg-type]
