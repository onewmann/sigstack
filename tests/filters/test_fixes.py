"""Tests for the S3.2 review fixes (pairing/ordering, dn at K, ellipap at large rs, validation).

One test method per fix group.

SciPy is the reference only. Each tolerance carries the measured deviation it was derived from.
"""

import math
import unittest
from typing import Any

import numpy as np
import scipy.signal as ss
import scipy.special as sp
from numpy.testing import assert_allclose

from sigstack.filters.analog import buttap, cheb1ap, cheb2ap, ellipap
from sigstack.filters.elliptic import ellipj
from sigstack.filters.iir import butter, cheby1, cheby2, ellip
from sigstack.filters.zpk import (
    bilinear_zpk,
    lp2bp_zpk,
    lp2bs_zpk,
    lp2hp_zpk,
    lp2lp_zpk,
    sos_response,
    zpk2sos,
)

SCIPY_BTYPE = {"low": "lowpass", "high": "highpass", "band": "bandpass", "stop": "bandstop"}
# (sigstack design, SciPy design, arguments between the order and Wn), one row per family
FAMILIES: list[Any] = [
    (butter, ss.butter, ()),
    (cheby1, ss.cheby1, (1.0,)),
    (cheby2, ss.cheby2, (40.0,)),
    (ellip, ss.ellip, (0.5, 40.0)),
]
SWEEP_BANDS: list[Any] = [
    ("low", 0.3),
    ("high", 0.3),
    ("band", [0.2, 0.5]),
    ("stop", [0.1, 0.9]),
    ("band", [0.3, 0.32]),
]
TRANSFORMS: list[Any] = [
    (lp2lp_zpk, {"wo": 2.0}),
    (lp2hp_zpk, {"wo": 2.0}),
    (lp2bp_zpk, {"wo": 2.0, "bw": 0.7}),
    (lp2bs_zpk, {"wo": 2.0, "bw": 0.7}),
    (bilinear_zpk, {"fs": 3.0}),
]


def section_radii(sos: Any) -> Any:
    """Largest pole modulus of every section (a first-order section has a2 = 0, one root at 0)."""
    return np.array([np.max(np.abs(np.roots(row[3:]))) for row in sos])


def partial_cascade_gains(sos: Any, n: int = 2048) -> Any:
    """max_w |H_1(e^jw) ... H_i(e^jw)| for i = 1..m: the gain seen at the output of section i."""
    w = np.pi * np.arange(n) / n
    e1 = np.exp(-1j * w)
    e2 = e1 * e1
    h = np.ones(n, dtype=complex)
    gains = []
    for b0, b1, b2, _, a1, a2 in sos:
        h = h * (b0 + b1 * e1 + b2 * e2) / (1.0 + a1 * e1 + a2 * e2)
        gains.append(float(np.max(np.abs(h))))
    return np.array(gains)


def match_multiset(tc: unittest.TestCase, got: Any, ref: Any, rtol: float) -> None:
    """Equal multisets: each reference value is used exactly once, |d| <= rtol * max(1, |ref|)."""
    got = np.atleast_1d(np.asarray(got, dtype=complex))
    ref = np.atleast_1d(np.asarray(ref, dtype=complex))
    tc.assertEqual(got.shape, ref.shape)
    free = list(got)
    for v in ref:
        d = [abs(v - g) for g in free]
        i = int(np.argmin(d))
        tc.assertLessEqual(d[i], rtol * max(1.0, abs(v)), (v, free))
        free.pop(i)


class TestFixes(unittest.TestCase):
    def test_zpk2sos_pairing_and_ordering(self) -> None:
        # B1, Jackson's rules (Oppenheim and Schafer 2010, Sec. 6.9.3, p. 452): the pole closest to
        # the unit circle takes the zero closest to it, repeated until all are paired; sections
        # ordered by increasing pole radius (highest-Q section last). (a) The float64 impulse
        # response of the cascade (scipy.signal.sosfilt on both) matches SciPy's SOS for the
        # high-order designs that the index-order pairing broke (ellip 8 stop 2.7e-3 at peak 0.17,
        # ellip 9 stop 0.22, ellip 11 stop tail 4e6, ellip 12 band tail 19). Two well-conditioned
        # cascades of the same filter differ only by rounding: measured max |y - yr| <= 4.3e-14 of
        # the peak after the fix, so 1e-6 * peak keeps seven decades of margin and is three decades
        # below the smallest failure (2.7e-3 / 0.17 = 1.6e-2).
        imp = np.zeros(2000)
        imp[0] = 1.0
        cases: list[Any] = [
            (ellip, ss.ellip, (8, 0.5, 40.0), [0.1, 0.9], "stop"),
            (ellip, ss.ellip, (9, 0.5, 40.0), [0.1, 0.9], "stop"),
            (ellip, ss.ellip, (11, 0.5, 40.0), [0.1, 0.9], "stop"),
            (ellip, ss.ellip, (10, 0.5, 40.0), [0.3, 0.32], "band"),
            (ellip, ss.ellip, (12, 0.5, 40.0), [0.2, 0.5], "band"),
            (cheby1, ss.cheby1, (10, 1.0), [0.1, 0.9], "stop"),
            (butter, ss.butter, (12,), [0.1, 0.9], "stop"),
        ]
        for fn, ref_fn, args, wn, btype in cases:
            with self.subTest(fn=fn.__name__, order=args[0], btype=btype):
                got = fn(*args, wn, btype)
                ref = ref_fn(*args, wn, SCIPY_BTYPE[btype], output="sos")
                y, yr = ss.sosfilt(got, imp), ss.sosfilt(ref, imp)
                assert_allclose(y, yr, rtol=0, atol=1e-6 * np.max(np.abs(yr)))
        # (b) Order sweep 1..12, four families, five bands: sections in non-decreasing pole radius
        # (equal radii, e.g. the two images of one prototype pole pair in a bandstop, may come in
        # any order: slack 1e-12 for np.roots), and the gain at the output of every partial cascade
        # bounded. Measured maximum over the sweep: 1.99 on SciPy's SOS and 1.99 on sigstack's (both
        # cheby1 12 stop); the bound 10 is five times that and twelve decades below the 5.6e12 of
        # the index-order pairing.
        for order in range(1, 13):
            for btype, wn in SWEEP_BANDS:
                for fn, _, pre in FAMILIES:
                    with self.subTest(fn=fn.__name__, order=order, btype=btype, wn=wn):
                        sos = fn(order, *pre, wn, btype)
                        radii = section_radii(sos)
                        self.assertTrue(np.all(np.diff(radii) >= -1e-12), radii)
                        gains = partial_cascade_gains(sos)
                        self.assertLessEqual(float(np.max(gains)), 10.0, gains)

    def test_zpk2sos_rejects_unpaired_complex_roots(self) -> None:
        # N3 + N10: a complex root without a conjugate partner within 100 eps max(1, |r|) is a
        # ValueError (was a bare StopIteration, or silently accepted up to 1e-9 with the imaginary
        # residue dropped). The designs of this package produce conjugates to 9.7e-16 relative
        # (measured over the order sweep), so 100 eps = 2.2e-14 has a margin of 20 and still rejects
        # a 1e-10 mismatch.
        cases: list[Any] = [
            ([], [0.5 + 0.5j, 0.2]),
            ([0.5 + 0.5j], [0.5]),
            ([-1.0, -1.0], [0.5 + 0.5j, 0.5 - 0.5000000001j]),
        ]
        for z, p in cases:
            with self.subTest(z=z, p=p), self.assertRaisesRegex(ValueError, "conjugate"):
                zpk2sos(np.array(z, dtype=complex), np.array(p, dtype=complex), 1.0)
        # one ulp of asymmetry is accepted; the section is formed from the exact conjugate pair, so
        # the coefficients equal those of the exact pair (a1 = -1, a2 = 0.5) to rounding
        exact = np.array([0.5 + 0.5j, 0.5 - 0.5j])
        ulp = np.array([0.5 + 0.5j, 0.5 - (0.5 + 2.0**-53) * 1j])
        sos = zpk2sos(np.array([-1.0, -1.0]), ulp, 1.0)
        assert_allclose(sos, zpk2sos(np.array([-1.0, -1.0]), exact, 1.0), rtol=0, atol=1e-15)
        assert_allclose(sos, [[1.0, 2.0, 1.0, 1.0, -1.0, 0.5]], rtol=0, atol=1e-15)

    def test_improper_and_2d_inputs(self) -> None:
        # N5: more zeros than poles (an improper transfer function) is a ValueError naming the poles
        # in every transform and in zpk2sos (lp2lp_zpk returned k * wo**d with d < 0 silently, the
        # others leaked numpy's 'negative dimensions' or a broadcast error). N9: 2-D z or p is a
        # ValueError, not ravelled: np.atleast_1d let a (1, n) row through and zpk2sos failed on the
        # array truth value; a nested or matrix-shaped input is a caller error, and ravel would
        # silently merge the rows of a (2, n) array into one root set. 0-d (scalar) input stays
        # accepted as a length-1 array.
        for fn, kw in TRANSFORMS:
            with (
                self.subTest(fn=fn.__name__, case="improper"),
                self.assertRaisesRegex(ValueError, "pole"),
            ):
                fn([-1.0, -2.0, -3.0], [-1.0], 1.0, **kw)
            with (
                self.subTest(fn=fn.__name__, case="2-D"),
                self.assertRaisesRegex(ValueError, "1-D"),
            ):
                fn([], [[-1.0, -2.0]], 1.0, **kw)
        with self.assertRaisesRegex(ValueError, "pole"):
            zpk2sos(np.array([-1.0, 1.0, 0.5]), np.array([0.5]), 1.0)
        with self.assertRaisesRegex(ValueError, "1-D"):
            zpk2sos(np.array([[-1.0, -1.0]]), np.array([[0.5, 0.6]]), 1.0)
        z, p, _ = lp2lp_zpk([], -1.0, 1.0, wo=2.0)
        self.assertEqual((z.shape, p.shape), ((0,), (1,)))
        assert_allclose(p, [-2.0], rtol=0, atol=0)  # exact: one multiplication
        self.assertEqual(zpk2sos(np.array([]), np.array([0.5, -0.5]), 1.0).shape, (1, 6))

    def test_sos_response_validation(self) -> None:
        # N11: a0 != 1 was silently treated as 1 (the docstring promises normalised sections), fs <=
        # 0 gave an all-zero or negative frequency axis, n < 1 leaked numpy's error, complex sos was
        # cast with a ComplexWarning. All are ValueError now; a valid call keeps the (n,) float /
        # complex shapes.
        good = ss.butter(4, 0.3, output="sos")
        f, h = sos_response(good, n=8)
        self.assertEqual(
            (f.shape, h.shape, f.dtype, h.dtype), ((8,), (8,), np.float64, np.complex128)
        )
        bad: list[Any] = [
            ([[1.0, 0.0, 0.0, 2.0, 0.0, 0.0]], {}),
            (good, {"n": 0}),
            (good, {"n": -1}),
            (good, {"fs": 0.0}),
            (good, {"fs": -1.0}),
            (good, {"fs": float("nan")}),
            (good.astype(complex), {}),
            (good[0], {}),
            (np.ones((2, 5)), {}),
        ]
        for sos, kw in bad:
            with self.subTest(sos=np.asarray(sos).shape, kw=kw), self.assertRaises(ValueError):
                sos_response(sos, **kw)

    def test_ellipj_dn_at_k(self) -> None:
        # N1: dn = cn / cos(phi_{n-1} - phi_n) is 0/0 at u = (2j + 1) K (gave 1.0 or -0.16 instead
        # of sqrt(1 - m)) and loses eps / |u - K| nearby (3.5e-8 at 1e-8). dn^2 = 1 - m sn^2 = (1 -
        # m) + m cn^2 has no cancellation. Dense grid plus the odd multiples of K and their
        # neighbourhood, atol 1e-12 as in test_elliptic (measured max |dn - scipy| 2.8e-14 after the
        # fix, 0 at u = K).
        for m in (0.3, 0.5, 0.9, 0.999):
            kk = float(sp.ellipk(m))
            special = [kk, 3.0 * kk, -kk, 5.0 * kk, kk + 1e-8, kk - 1e-8, kk + 1e-12, kk - 1e-12]
            u = np.concatenate((np.linspace(-20.0, 20.0, 20001), special))
            with self.subTest(m=m):
                sn, cn, dn = ellipj(u, m)
                ref = sp.ellipj(u, m)
                for got, want in zip((sn, cn, dn), ref[:3], strict=True):
                    self.assertEqual(got.shape, u.shape)
                    assert_allclose(got, want, rtol=0, atol=1e-12)
                self.assertTrue(np.all(dn >= math.sqrt(1.0 - m) - 1e-15))  # sqrt(1 - m) <= dn <= 1
                self.assertTrue(np.all(dn <= 1.0 + 1e-15))
                s, c, d = ellipj(kk, m)  # scalar u = K: sn = 1, cn = 0, dn = sqrt(1 - m)
                assert_allclose([s, c, d], [1.0, 0.0, math.sqrt(1.0 - m)], rtol=0, atol=1e-15)

    def test_ellipap_large_rs(self) -> None:
        # N2: 1 - m1 rounded to 1.0 for the discrimination parameter m1 = eps^2 / (10^(rs/10) - 1)
        # below 1.1e-16 (rs - rp beyond ~135 dB at rp = 0.01, ~155 dB at rp = 1) and ellipj raised
        # 'm must satisfy 0 <= m < 1'. The Landen chain started from b0 = sqrt(m1) (the
        # complementary parameter entry) keeps k1' exact. SciPy designs all of these; zeros up to
        # |z| = 2.7e6, so the roots are compared relative to max(1, |ref|): measured max 1.5e-12
        # (roots), rtol 1e-9 keeps nearly three decades; k rtol 1e-9 (measured 8.7e-13). Also the
        # design contract: Re p < 0, zeros on the imaginary axis, |H(0)| = 1 (odd n) or 10^(-rp/20)
        # (even n).
        cases: list[Any] = [
            (3, 1.0, 160.0),
            (3, 0.01, 140.0),
            (2, 1e-4, 150.0),
            (4, 0.1, 160.0),
            (3, 1.0, 400.0),
        ]
        for n, rp, rs in cases:
            with self.subTest(n=n, rp=rp, rs=rs):
                z, p, k = ellipap(n, rp, rs)
                zr, pr, kr = ss.ellipap(n, rp, rs)
                self.assertEqual((z.dtype, p.dtype), (np.complex128, np.complex128))
                self.assertIsInstance(k, float)
                match_multiset(self, z, zr, 1e-9)
                match_multiset(self, p, pr, 1e-9)
                assert_allclose(k, kr, rtol=1e-9)
                self.assertTrue(np.all(np.real(p) < 0))
                assert_allclose(np.real(z), 0.0, rtol=0, atol=1e-9 * np.max(np.abs(z)))
                h0 = abs(k * np.prod(-z) / np.prod(-p))
                assert_allclose(h0, 1.0 if n % 2 else 10 ** (-rp / 20), rtol=1e-9)

    def test_nonfinite_and_bad_types_rejected(self) -> None:
        # Validation in iir._validate and the analog prototypes. Before: cheby1(4, nan, .3) gave
        # all-NaN sections, cheby1(4, inf, .3) |p| = 1 + 2e-16, butter(4, 1000, fs=inf) poles on |z|
        # = 1, cheby2(4, inf, .3) a LinAlgError, cheb1ap(3, 1e-300) a ZeroDivisionError (10**(rp/10)
        # - 1 == 0), order 4.0 a TypeError only in butter, 4.5 a StopIteration (cheby1/2) or a
        # silent design (ellip). Contract: non-finite or unresolvable magnitudes -> ValueError; an
        # order that is not an integer (bool included) -> TypeError; numpy integers are integers.
        nan, inf = float("nan"), float("inf")
        value_errors: list[Any] = [
            (cheby1, (4, nan, 0.3)),
            (cheby1, (4, inf, 0.3)),
            (cheby2, (4, nan, 0.3)),
            (cheby2, (4, inf, 0.3)),
            (ellip, (4, nan, 40.0, 0.3)),
            (ellip, (4, inf, 40.0, 0.3)),
            (ellip, (4, 1.0, nan, 0.3)),
            (ellip, (4, 1.0, inf, 0.3)),
            (butter, (4, 1000.0, "low", nan)),
            (butter, (4, 1000.0, "low", inf)),
            (butter, (4, 1000.0, "low", 0.0)),
            (butter, (4, 1000.0, "low", -8000.0)),
            (butter, (4, nan)),
            (butter, (4, inf)),
            (butter, (4, [0.2, nan], "band")),
            (cheby1, (4, 1e-300, 0.3)),
            (cheby2, (4, 1e-300, 0.3)),
            (ellip, (4, 1e-300, 40.0, 0.3)),
            (cheb1ap, (3, nan)),
            (cheb1ap, (3, inf)),
            (cheb2ap, (3, nan)),
            (cheb2ap, (3, inf)),
            (ellipap, (3, nan, 40.0)),
            (ellipap, (3, 1.0, inf)),
            (ellipap, (3, inf, inf)),
            (cheb1ap, (3, 1e-300)),
            (cheb2ap, (3, 1e-300)),
            (ellipap, (3, 1e-300, 40.0)),
        ]
        for fn, args in value_errors:
            with self.subTest(fn=fn.__name__, args=args), self.assertRaises(ValueError):
                fn(*args)
        type_errors: list[Any] = [
            (butter, (4.0, 0.3)),
            (butter, (4.5, 0.3)),
            (butter, (True, 0.3)),
            (butter, ("4", 0.3)),
            (cheby1, (4.5, 1.0, 0.3)),
            (cheby2, (4.5, 40.0, 0.3)),
            (ellip, (4.5, 1.0, 40.0, 0.3)),
            (buttap, (4.0,)),
            (buttap, (True,)),
            (cheb1ap, (3.5, 1.0)),
            (cheb2ap, (3.0, 40.0)),
            (ellipap, (2.5, 1.0, 40.0)),
        ]
        for fn, args in type_errors:
            with self.subTest(fn=fn.__name__, args=args), self.assertRaises(TypeError):
                fn(*args)
        # numpy integers are accepted at run time (operator.index); the static signature says int
        self.assertEqual(butter(np.int64(4), 0.3).shape, (2, 6))  # type: ignore[arg-type]
        self.assertEqual(buttap(np.int32(3))[1].shape, (3,))  # type: ignore[arg-type]
