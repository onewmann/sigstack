"""Tests for the second review round of S3.2 (fair pairing order, trivial zpk, non-finite inputs).

zpk2sos serves every pole group in one pass, closest to the unit circle first (Jackson's rule 2
for all inputs, not only for the package's own designs); the trivial transfer function is one
section; non-finite roots and gains are rejected; ellipk/ellipj reject NaN.
"""

import unittest
from typing import Any

import numpy as np
import scipy.signal as ss
from numpy.testing import assert_allclose

from sigstack.filters.elliptic import ellipj, ellipk
from sigstack.filters.zpk import lp2lp_zpk, sos_response, zpk2sos


def section_rows(groups: list[tuple[Any, Any]], k: float) -> Any:
    """Expected SOS rows from (zeros, poles) per section, in order; the gain sits in row 0."""
    rows = []
    for i, (zeros, poles) in enumerate(groups):
        b = np.real(np.poly(np.asarray(zeros, dtype=complex))) if len(zeros) else np.array([1.0])
        a = np.real(np.poly(np.asarray(poles, dtype=complex)))
        b = np.concatenate([b, np.zeros(3 - len(b))]) * (k if i == 0 else 1.0)
        rows.append(np.concatenate([b, a, np.zeros(3 - len(a))]))
    return np.array(rows)


class TestS32d(unittest.TestCase):
    def test_zpk2sos_fair_order(self) -> None:
        # Jackson's rules for generic inputs: pole groups are served in one pass by decreasing |p|
        # (closest to the unit circle first), whatever their kind. A second-order group takes the
        # nearest zero group (a conjugate pair, or the nearest real zero plus the next-nearest real
        # zero to the same reference pole, whichever is nearer); a lone real pole, when its turn
        # comes, takes the nearest remaining real zero (a first-order section can never hold a
        # complex pair). Sections are then ordered by increasing |p|, the gain sits in row 0.
        # Before the fix a lone real pole was served before all second-order groups, so in case A
        # it grabbed 0.0307 although the pair at |p| = 0.855 is nearer to the circle (partial
        # cascade gain 125 instead of 7.7), and in case B it took 0.85 away from the pair at 0.949.
        pa = [-0.2464 + 0.8185j, -0.2464 - 0.8185j]
        pb = [-0.6977 + 0.2222j, -0.6977 - 0.2222j]
        zc = [-0.8912 + 0.0773j, -0.8912 - 0.0773j]
        case_a = (
            np.array([*zc, 0.5089, 0.0307, 0.7253]),
            np.array([*pa, *pb, -0.8475]),
            1.0,
            [(zc, pb), ([0.7253], [-0.8475]), ([0.0307, 0.5089], pa)],
        )
        pp = [0.9 + 0.3j, 0.9 - 0.3j]
        case_b = (
            np.array([0.85, 0.95, -0.5]),
            np.array([*pp, 0.5]),
            2.0,
            [([-0.5], [0.5]), ([0.85, 0.95], pp)],
        )
        zz = [0.95 + 0.2j, 0.95 - 0.2j]
        qq = [0.3 + 0.3j, 0.3 - 0.3j]
        # regression pin: the real pole is nearest to the circle and takes the only real zero
        case_c = (
            np.array([*zz, -1.0]),
            np.array([0.97, *qq]),
            1.0,
            [(zz, qq), ([-1.0], [0.97])],
        )
        cases: list[tuple[Any, Any, float, list[tuple[Any, Any]]]] = [case_a, case_b, case_c]
        for name, (z, p, k, groups) in zip("ABC", cases, strict=True):
            with self.subTest(case=name):
                sos = zpk2sos(z, p, k)
                expected = section_rows(groups, k)
                self.assertEqual(sos.shape, expected.shape)
                assert_allclose(sos, expected, rtol=0, atol=1e-12)
                # the product is unchanged whatever the pairing
                _, h = ss.sosfreqz(sos, worN=256)
                _, h_ref = ss.freqz_zpk(z, p, k, worN=256)
                assert_allclose(h, h_ref, rtol=1e-9, atol=1e-12)

    def test_zpk2sos_no_pair_left_for_lone_pole(self) -> None:
        # Found by review (263 of 3000 random sets): a second-order group took the last real zero
        # by distance, the remaining zeros were a conjugate pair and the lone real pole raised
        # IndexError. A pair is now forced when the free pairs would otherwise outnumber the
        # second-order groups left; the lone pole keeps a real zero.
        cases: list[tuple[list[complex], list[complex]]] = [
            ([0.9, 0.3 + 0.3j, 0.3 - 0.3j], [0.95 + 0.1j, 0.95 - 0.1j, 0.31]),
            ([-0.6252 + 0.4892j, -0.6252 - 0.4892j], [-0.4075, -0.1391, -0.4615]),
        ]
        for z, p in cases:
            with self.subTest(z=z, p=p):
                sos = zpk2sos(np.array(z), np.array(p), 1.0)
                self.assertEqual(sos.shape, (2, 6))
                _, h = ss.sosfreqz(sos, worN=256)
                # zeros at infinity are realised as zeros at the origin (z^d H(z), documented)
                padded = z + [0.0] * (len(p) - len(z))
                _, h_ref = ss.freqz_zpk(padded, p, 1.0, worN=256)
                assert_allclose(h, h_ref, rtol=1e-9, atol=1e-12)
                first_order = [row for row in sos if row[5] == 0.0 and row[2] == 0.0]
                self.assertEqual(len(first_order), 1)  # the lone pole's section holds one real zero

    def test_zpk2sos_empty_is_pure_gain(self) -> None:
        # no zeros and no poles is the constant transfer function k: one trivial section (SciPy
        # returns the same); before the fix an IndexError escaped from the gain multiplication
        assert_allclose(zpk2sos(np.array([]), np.array([]), 2.0), [[2.0, 0, 0, 1.0, 0, 0]])
        assert_allclose(zpk2sos(np.array([]), np.array([]), -0.5), [[-0.5, 0, 0, 1.0, 0, 0]])

    def test_zpk2sos_rejects_nonfinite(self) -> None:
        # a NaN or infinite root was classified as real and written into the coefficients (a NaN
        # zero even displaced finite zeros in the nearest-zero search); a NaN or infinite gain gave
        # NaN rows; a complex gain leaked a TypeError from float(). All are ValueError now, the
        # message names what is wrong; the transforms share the root check.
        nan, inf = float("nan"), float("inf")
        bad_roots: list[Any] = [
            ([nan], [0.5]),
            ([-1.0], [inf]),
            ([nan, -1.0, 0.5, 0.2], [0.5 + 0.5j, 0.5 - 0.5j, 0.3, 0.1]),
            ([-1.0], [-inf]),
        ]
        for z, p in bad_roots:
            with self.subTest(z=z, p=p), self.assertRaisesRegex(ValueError, "finite"):
                zpk2sos(np.array(z, dtype=complex), np.array(p, dtype=complex), 1.0)
        for k in (nan, inf, -inf):
            with self.subTest(k=k), self.assertRaisesRegex(ValueError, "finite"):
                zpk2sos(np.array([-1.0]), np.array([0.5]), k)
        with self.assertRaisesRegex(ValueError, "real"):
            zpk2sos(np.array([-1.0]), np.array([0.5]), 1.0 + 1.0j)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "finite"):
            lp2lp_zpk([nan], [-1.0], 1.0, wo=2.0)
        # a complex gain is rejected even with a zero imaginary part (it used to leak a TypeError
        # or a numpy ComplexWarning); numpy real scalars stay accepted
        complex_gains: list[Any] = [2.0 + 0.0j, np.complex128(2.0)]
        for gain in complex_gains:
            with self.subTest(k=gain), self.assertRaisesRegex(ValueError, "real"):
                zpk2sos(np.array([-1.0]), np.array([0.5]), gain)
        assert_allclose(
            zpk2sos(np.array([-1.0]), np.array([0.5]), np.float64(2.0)), [[2, 2, 0, 1, -0.5, 0]]
        )

    def test_sos_response_rejects_infinite_fs(self) -> None:
        # fs = inf passed 'fs > 0' and produced a frequency grid of [nan, inf, inf, ...]
        good = ss.butter(4, 0.3, output="sos")
        for fs in (float("inf"), -float("inf")):
            with self.subTest(fs=fs), self.assertRaises(ValueError):
                sos_response(good, fs=fs)

    def test_elliptic_rejects_nan(self) -> None:
        # 'm < 0 or m >= 1' is False for NaN, so NaN flowed through the AGM loops silently
        nan = float("nan")
        with self.assertRaises(ValueError):
            ellipk(nan)
        with self.assertRaises(ValueError):
            ellipj(1.0, nan)
        with self.assertRaises(ValueError):
            ellipj(np.array([0.0, 1.0]), nan)
