import unittest
from typing import Any

import numpy as np
import scipy.signal as ss
from numpy.testing import assert_allclose

from sigstack.filters.analog import buttap, cheb1ap
from sigstack.filters.zpk import (
    bilinear_zpk,
    lp2bp_zpk,
    lp2bs_zpk,
    lp2hp_zpk,
    lp2lp_zpk,
    sos_response,
    zpk2sos,
)

Z, P, K = ss.ellipap(4, 1.0, 40.0)


def same_set(tc: unittest.TestCase, got: Any, ref: Any, atol: float) -> None:
    got, ref = np.asarray(got, dtype=complex), np.asarray(ref, dtype=complex)
    tc.assertEqual(got.shape, ref.shape)
    for a, b in ((got, ref), (ref, got)):
        for v in a:
            tc.assertLess(float(np.min(np.abs(b - v))), atol, (v, b))


def match_multiset(tc: unittest.TestCase, got: Any, ref: Any, atol: float) -> None:
    """Equal multisets: every reference value is used exactly once (multiplicities count)."""
    got, ref = (
        np.atleast_1d(np.asarray(got, dtype=complex)),
        np.atleast_1d(np.asarray(ref, dtype=complex)),
    )
    tc.assertEqual(got.shape, ref.shape)
    free = list(got)
    for v in ref:
        d = [abs(v - g) for g in free]
        i = int(np.argmin(d)) if d else 0
        tc.assertTrue(d and d[i] < atol, (v, free))
        free.pop(i)


class TestZpk(unittest.TestCase):
    def test_transforms_match_scipy(self) -> None:
        # same conventions as scipy.signal.lp2*_zpk and bilinear_zpk (prewarping is the
        # caller's job)
        cases: list[Any] = [
            (lp2lp_zpk, ss.lp2lp_zpk, {"wo": 2.5}),
            (lp2hp_zpk, ss.lp2hp_zpk, {"wo": 2.5}),
            (lp2bp_zpk, ss.lp2bp_zpk, {"wo": 2.0, "bw": 0.7}),
            (lp2bs_zpk, ss.lp2bs_zpk, {"wo": 2.0, "bw": 0.7}),
            (bilinear_zpk, ss.bilinear_zpk, {"fs": 3.0}),
        ]
        for fn, ref_fn, kw in cases:
            with self.subTest(fn=fn.__name__):
                z, p, k = fn(Z, P, K, **kw)
                zr, pr, kr = ref_fn(Z, P, K, **kw)
                same_set(self, z, zr, 1e-9)
                same_set(self, p, pr, 1e-9)
                assert_allclose(k, kr, rtol=1e-9)
                self.assertIsInstance(k, float)

    def test_zpk2sos_and_response(self) -> None:
        # sections [b0, b1, b2, 1, a1, a2], real coefficients, ceil(order / 2) rows; the
        # cascade has the same response as the zpk (section order and pairing are free)
        z, p, k = ss.bilinear_zpk(*ss.lp2bp_zpk(Z, P, K, wo=1.0, bw=0.5), fs=2.0)
        sos = zpk2sos(z, p, k)
        self.assertEqual((sos.shape, sos.dtype), ((4, 6), np.float64))
        assert_allclose(sos[:, 3], 1.0, rtol=0, atol=0)
        _, h = sos_response(sos, n=256)
        _, hr = ss.freqz_zpk(z, p, k, worN=256)
        assert_allclose(h, hr, rtol=1e-9, atol=1e-12)
        z3, p3, k3 = ss.bilinear_zpk(*ss.buttap(3), fs=2.0)  # odd order: one first-order section
        sos3 = zpk2sos(z3, p3, k3)
        self.assertEqual(sos3.shape, (2, 6))
        assert_allclose(
            sos_response(sos3, n=64)[1], ss.freqz_zpk(z3, p3, k3, worN=64)[1], rtol=1e-9, atol=1e-12
        )

    def test_sos_response_matches_sosfreqz(self) -> None:
        # (f, H) on the grid of sigstack.metrics.freq_response: w_k = pi k / n (or 2 pi k / n),
        # f = w fs / (2 pi)
        sos = ss.butter(6, [0.2, 0.4], btype="bandpass", output="sos")
        for kw in ({}, {"fs": 48000.0}, {"whole": True}):
            with self.subTest(kw=kw):
                f, h = sos_response(sos, n=128, **kw)
                fr, hr = ss.sosfreqz(sos, worN=128, **{"fs": 2 * np.pi, **kw})
                assert_allclose(f, fr, rtol=1e-12, atol=1e-12)
                assert_allclose(h, hr, rtol=1e-10, atol=1e-13)

    # review: closes the d > 0 gap (the ellipap(4) prototype above has d = 0) with exact
    # multiplicities; kills the surviving 'real -> abs' mutants in bilinear / lp2hp / lp2bs
    def test_transforms_with_zeros_at_infinity(self) -> None:
        # Butterworth / Chebyshev I prototypes have no finite zeros (d = len(p) - len(z) = n > 0):
        # the d new zeros are -1 (bilinear), 0 (lp2hp, lp2bp), +-j wo (lp2bs), gain factors wo**d,
        # bw**d; exact multiplicities; same closed forms as SciPy: atol 1e-9, k rtol 1e-12
        transforms: list[Any] = [
            (lp2lp_zpk, ss.lp2lp_zpk, {"wo": 2.5}),
            (lp2hp_zpk, ss.lp2hp_zpk, {"wo": 2.5}),
            (lp2bp_zpk, ss.lp2bp_zpk, {"wo": 2.0, "bw": 0.7}),
            (lp2bs_zpk, ss.lp2bs_zpk, {"wo": 2.0, "bw": 0.7}),
            (bilinear_zpk, ss.bilinear_zpk, {"fs": 2.0}),
        ]
        for n, proto in [(3, buttap(3)), (4, cheb1ap(4, 1.0)), (5, buttap(5))]:
            z, p, k = proto
            for fn, ref_fn, kw in transforms:
                with self.subTest(n=n, fn=fn.__name__):
                    zg, pg, kg = fn(z, p, k, **kw)
                    zr, pr, kr = ref_fn(z, p, k, **kw)
                    match_multiset(self, zg, zr, 1e-9)
                    match_multiset(self, pg, pr, 1e-9)
                    assert_allclose(kg, kr, rtol=1e-12)
                    self.assertEqual(
                        (zg.dtype, pg.dtype, zg.ndim, pg.ndim), (np.complex128, np.complex128, 1, 1)
                    )
        zb = bilinear_zpk(*buttap(4), fs=2.0)[0]
        self.assertEqual(int(np.sum(np.abs(zb + 1.0) < 1e-12)), 4)  # four zeros at z = -1
        zs = lp2bs_zpk(*buttap(3), wo=2.0, bw=0.7)[0]
        self.assertEqual(
            (int(np.sum(np.abs(zs - 2j) < 1e-12)), int(np.sum(np.abs(zs + 2j) < 1e-12))), (3, 3)
        )
        # right-half-plane zero / negative gain: the sign of Re(prod) must be kept (kills
        # real -> abs); 6 > 2 fs = 4: 2 fs - z < 0
        z, p, k = np.array([6.0 + 0j]), np.array([-1.0 + 0j, -2.0 + 0j]), 1.5
        rhp_cases: list[Any] = [
            (bilinear_zpk, ss.bilinear_zpk, {"fs": 2.0}),
            (lp2hp_zpk, ss.lp2hp_zpk, {"wo": 2.0}),
            (lp2bs_zpk, ss.lp2bs_zpk, {"wo": 2.0, "bw": 0.5}),
        ]
        for fn, ref_fn, kw in rhp_cases:
            with self.subTest(rhp=fn.__name__):
                assert_allclose(fn(z, p, k, **kw)[2], ref_fn(z, p, k, **kw)[2], rtol=1e-12)
