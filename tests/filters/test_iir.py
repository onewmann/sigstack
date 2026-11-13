import math
import unittest
from typing import Any

import numpy as np
import scipy.signal as ss
from numpy.testing import assert_allclose

from sigstack.filters.iir import butter, cheby1, cheby2, ellip
from sigstack.filters.zpk import sos_response

SCIPY_BTYPE = {"low": "lowpass", "high": "highpass", "band": "bandpass", "stop": "bandstop"}
# (name, sigstack fn, SciPy fn, leading args: order 4 and the family's ripple parameters)
FAMILIES: list[Any] = [
    ("butter", butter, ss.butter, (4,)),
    ("cheby1", cheby1, ss.cheby1, (4, 1.0)),
    ("cheby2", cheby2, ss.cheby2, (4, 40.0)),
    ("ellip", ellip, ss.ellip, (4, 0.5, 40.0)),
]
BANDS: list[Any] = [("low", 0.3), ("high", 0.3), ("band", [0.2, 0.5]), ("stop", [0.2, 0.5])]


def poles(sos: Any) -> Any:
    return np.concatenate([np.roots(s[3:]) for s in sos])


class TestIir(unittest.TestCase):
    # digital designs as second-order sections, SciPy's conventions (prewarped bilinear
    # transform, Wn relative to fs / 2, cheby2's Wn is the stopband edge); compared by their
    # frequency response (section order free)
    def compare(self, sos: Any, ref: Any, rtol: float) -> None:
        self.assertEqual((sos.shape, sos.dtype), (ref.shape, np.float64))
        h = sos_response(sos, n=512)[1]
        hr = ss.sosfreqz(ref, worN=512)[1]
        assert_allclose(h, hr, rtol=rtol, atol=1e-9)
        self.assertTrue(np.all(np.abs(poles(sos)) < 1.0))  # stable

    def test_designs_match_scipy(self) -> None:
        for btype, wn in BANDS:
            for order in (1, 4, 5):
                kind = SCIPY_BTYPE[btype]
                with self.subTest(btype=btype, order=order):
                    self.compare(
                        butter(order, wn, btype), ss.butter(order, wn, kind, output="sos"), 1e-9
                    )
                    self.compare(
                        cheby1(order, 1.0, wn, btype),
                        ss.cheby1(order, 1.0, wn, kind, output="sos"),
                        1e-8,
                    )
                    self.compare(
                        cheby2(order, 40.0, wn, btype),
                        ss.cheby2(order, 40.0, wn, kind, output="sos"),
                        1e-8,
                    )
                    self.compare(
                        ellip(order, 0.5, 40.0, wn, btype),
                        ss.ellip(order, 0.5, 40.0, wn, kind, output="sos"),
                        1e-6,
                    )

    def test_fs(self) -> None:
        self.compare(
            butter(4, 1000.0, "low", fs=8000.0), ss.butter(4, 1000.0, fs=8000.0, output="sos"), 1e-9
        )
        self.compare(
            ellip(3, 1.0, 50.0, [500.0, 1500.0], "band", fs=8000.0),
            ss.ellip(3, 1.0, 50.0, [500.0, 1500.0], "bandpass", fs=8000.0, output="sos"),
            1e-6,
        )

    def test_validation(self) -> None:
        cases: list[Any] = [
            (butter, (0, 0.3)),  # order >= 1
            (butter, (4, 1.0)),  # 0 < Wn < fs / 2
            (butter, (4, [0.5, 0.2], "band")),  # increasing pair
            (butter, (4, 0.3, "band")),  # band and stop need two edges
            (butter, (4, 0.3, "notch")),  # low, high, band or stop
            (cheby1, (4, 0.0, 0.3)),  # rp > 0
            (cheby2, (4, -10.0, 0.3)),  # rs > 0
            (ellip, (4, 1.0, 0.5, 0.3)),  # rs > rp
        ]
        for fn, args in cases:
            with self.subTest(fn=fn.__name__, args=args), self.assertRaises(ValueError):
                fn(*args)

    # review: kills the fs/3 mutants in cheby1 and cheby2 that the suite missed; adds highpass /
    # bandstop with fs and the fs-vs-normalised equivalence, which does not depend on SciPy
    def test_fs_every_family_and_band_type(self) -> None:
        # fs in Hz: Wn / (fs / 2) must equal the normalised design for all four functions (each
        # scales Wn in its own body) and match SciPy; measured worst rel. error 4e-12, rtol 1e-8
        # keeps three decades and still catches any scaling error (> 1e-3)
        fs = 8000.0
        edges: dict[str, Any] = {
            "low": 1000.0,
            "high": 1000.0,
            "band": [500.0, 1500.0],
            "stop": [500.0, 1500.0],
        }
        for name, fn, ref_fn, pre in FAMILIES:
            for btype, wn in edges.items():
                with self.subTest(fn=name, btype=btype):
                    got = fn(*pre, wn, btype, fs=fs)
                    ref = ref_fn(*pre, wn, SCIPY_BTYPE[btype], fs=fs, output="sos")
                    self.assertEqual((got.shape, got.dtype), (ref.shape, np.float64))
                    assert_allclose(
                        sos_response(got, n=512)[1],
                        ss.sosfreqz(ref, worN=512)[1],
                        rtol=1e-8,
                        atol=1e-10,
                    )
                    norm = np.asarray(wn) / (fs / 2.0)
                    same = fn(*pre, norm.tolist() if norm.ndim else float(norm), btype)
                    # same pipeline, only the scaling differs
                    assert_allclose(got, same, rtol=0, atol=1e-12)

    # review: kills 12 boundary / branch mutants of iir._validate and the pairing-tolerance mutant
    # (1e-9 -> 1e-3) via the 1e-3 inside-bound designs
    def test_validation_edges(self) -> None:
        # open interval 0 < Wn < fs / 2: the bounds themselves must raise (Wn = 0 -> poles at z = 1,
        # Wn = Nyquist -> wo = 6.5e16); every family, since each calls _validate with its own
        # rp / rs
        cases: list[Any] = [
            ("low", 0.0, None),
            ("low", -0.1, None),
            ("low", 1.0, None),
            ("low", 1.5, None),
            ("low", float("nan"), None),
            ("low", 0.0, 8000.0),
            ("low", 4000.0, 8000.0),
            ("low", 5000.0, 8000.0),
            ("high", 0.0, None),
            ("high", 1.0, None),
            ("band", [0.0, 0.5], None),
            ("band", [0.2, 1.0], None),
            ("band", [0.2, 1.5], None),
            ("band", [-0.1, 0.5], None),
            ("band", [0.3, 0.3], None),
            ("band", [0.2, float("nan")], None),
            ("band", [0.2, 0.5, 0.7], None),
            ("band", [0.2], None),
            ("stop", [0.0, 0.5], None),
            ("stop", [0.2, 1.0], None),
            ("stop", [1000.0, 4000.0], 8000.0),
            ("stop", [0.0, 1000.0], 8000.0),
            ("band", [1000.0, 5000.0], 8000.0),
            ("low", 0.3, 0.0),
            ("low", 0.3, -8000.0),
            ("low", [0.3], None),
            ("high", np.array([0.3]), None),
        ]
        for name, fn, _, pre in FAMILIES:
            for btype, wn, fs in cases:
                with (
                    self.subTest(fn=name, btype=btype, wn=wn, fs=fs),
                    self.assertRaises(ValueError),
                ):
                    fn(*pre, wn, btype, fs=fs)
        # just inside the bounds still designs, finite and stable
        for name, fn, _, pre in FAMILIES:
            for btype, wn in (("low", 1e-3), ("high", 1 - 1e-3), ("band", [1e-3, 1 - 1e-3])):
                with self.subTest(fn=name, inside=btype):
                    sos = fn(*pre, wn, btype)
                    self.assertTrue(np.all(np.isfinite(sos)))
                    self.assertTrue(np.all(np.abs(poles(sos)) < 1.0))
        param_cases: list[Any] = [
            (ellip, (4, 1.0, 1.0, 0.3)),
            (ellip, (4, 0.0, 40.0, 0.3)),
            (ellip, (4, 1.0, 0.0, 0.3)),
            (cheby1, (4, -1.0, 0.3)),
            (cheby2, (4, 0.0, 0.3)),
            (cheby1, (0, 1.0, 0.3)),
            (cheby2, (-3, 40.0, 0.3)),
            (ellip, (0, 1.0, 40.0, 0.3)),
        ]
        for fn, args in param_cases:
            with self.subTest(fn=fn.__name__, args=args), self.assertRaises(ValueError):
                fn(*args)

    # review: an oracle other than SciPy, so a shared convention mistake would be caught
    def test_textbook_examples_oppenheim_schafer(self) -> None:
        # Independent of SciPy. O&S (2010) Sec. 7.3, Example 7.3, Eq. (7.35): 6th-order Butterworth
        # by the bilinear transform (T_d = 1, Omega_c = 0.766 rad/s, i.e. here
        # Wn = (2/pi) atan(Omega_c/2)): H(z) = 0.0007378 (1+z^-1)^6 / [(1 - 1.2686z^-1 + 0.7051z^-2)
        # (1 - 1.0106z^-1 + 0.3583z^-2)(1 - 0.9044z^-1 + 0.2155z^-2)], -0.56 dB at 0.2 pi, exactly
        # -15 dB at 0.3 pi. The exact Omega_c is the one giving |H| = 0.17783 at 0.3 pi (Eq. 7.32b).
        # Coefficients printed with 4 decimals: atol 6e-5 (half a last digit plus input rounding;
        # measured worst 4.5e-5).
        a = 0.17783  # |H_c| at the stopband edge, 15 dB
        omega_c = 2 * math.tan(0.15 * math.pi) / (((1 / a) ** 2 - 1) ** (1 / 12))
        sos = butter(6, 2 / math.pi * math.atan(omega_c / 2))
        self.assertEqual(sos.shape, (3, 6))
        sos = sos[np.argsort(-sos[:, 5])]  # section order is free: sort by a2
        assert_allclose(sos[:, 4], [-1.2686, -1.0106, -0.9044], rtol=0, atol=6e-5)
        assert_allclose(sos[:, 5], [0.7051, 0.3583, 0.2155], rtol=0, atol=6e-5)
        # numerator at z = 1
        assert_allclose(np.prod(sos[:, 0:3].sum(axis=1)), 0.0007378 * 2**6, rtol=2e-3)
        h = sos_response(sos, n=1000)[1]  # grid pi k / 1000: k = 200, 300 are 0.2 pi, 0.3 pi
        assert_allclose(20 * np.log10(abs(h[[200, 300]])), [-0.56, -15.0], rtol=0, atol=6e-3)
        # Sec. 7.4, Example 7.6, Eq. (7.49): 4th-order Chebyshev I, 1 dB ripple, passband edge
        # 0.2 pi: 0.001836 (1+z^-1)^4 / [(1 - 1.5548z^-1 + 0.6493z^-2)(1 - 1.4996z^-1 + 0.8482z^-2)]
        c = cheby1(4, 1.0, 0.2)
        c = c[np.argsort(-c[:, 5])]
        assert_allclose(c[:, 4], [-1.4996, -1.5548], rtol=0, atol=6e-5)
        assert_allclose(c[:, 5], [0.8482, 0.6493], rtol=0, atol=6e-5)
        assert_allclose(np.prod(c[:, 0:3].sum(axis=1)), 0.001836 * 2**4, rtol=1e-3)
        assert_allclose(20 * np.log10(abs(sos_response(c, n=1000)[1][200])), -1.0, atol=1e-9)

    # review: covers orders 2, 3, 6-12 that no test touches; a0 == 1, finiteness and section count
    # for every design; Jury conditions instead of the np.roots-based stability check
    def test_order_sweep_stable_and_matches_scipy(self) -> None:
        # every order 1..12, narrow and wide bands; a0 == 1, finite, section count, Jury triangle
        # per section (stable <=> |a2| < 1 and |a1| < 1 + a2, independent of np.roots rounding);
        # worst measured deviation 3e-12 relative to the peak, so atol = 1e-8 * peak (relative to
        # the peak because near-zero stopband points would dominate rtol)
        designs: list[Any] = [
            ("low", 0.05),
            ("high", 0.9),
            ("band", [0.3, 0.32]),
            ("stop", [0.1, 0.9]),
        ]
        for name, fn, ref_fn, pre in FAMILIES:
            for btype, wn in designs:
                for order in range(1, 13):
                    with self.subTest(fn=name, btype=btype, order=order):
                        got = fn(order, *pre[1:], wn, btype)
                        ref = ref_fn(order, *pre[1:], wn, SCIPY_BTYPE[btype], output="sos")
                        self.assertEqual(got.shape, ref.shape)
                        self.assertEqual(
                            got.shape[0], order if btype in ("band", "stop") else (order + 1) // 2
                        )
                        assert_allclose(got[:, 3], 1.0, rtol=0, atol=0)
                        self.assertTrue(np.all(np.isfinite(got)))
                        a1, a2 = got[:, 4], got[:, 5]
                        self.assertTrue(np.all(np.abs(a2) < 1.0) and np.all(np.abs(a1) < 1.0 + a2))
                        h = sos_response(got, n=1024)[1]
                        hr = ss.sosfreqz(ref, worN=1024)[1]
                        assert_allclose(h, hr, rtol=0, atol=1e-8 * np.max(np.abs(hr)))

    # review: guards the time-domain contract (pole / zero pairing and section order) for ordinary
    # designs, which no frequency-response test can see
    def test_impulse_response_matches_scipy(self) -> None:
        # time domain, i.e. what runs in float64: the product response is insensitive to how poles
        # and zeros are paired and ordered into sections, the impulse response is not; orders 2..6,
        # ordinary edges; measured worst |difference| 2e-13 (peak 0.2..1.5), atol 1e-9 leaves four
        # decades; the cascade is evaluated by SciPy's sosfilt for both (test only)
        imp = np.zeros(1500)
        imp[0] = 1.0
        designs: list[Any] = [
            ("low", 0.2),
            ("high", 0.35),
            ("band", [0.2, 0.5]),
            ("stop", [0.2, 0.5]),
        ]
        for name, fn, ref_fn, pre in FAMILIES:
            for btype, wn in designs:
                for order in range(2, 7):
                    with self.subTest(fn=name, btype=btype, order=order):
                        got = fn(order, *pre[1:], wn, btype)
                        ref = ref_fn(order, *pre[1:], wn, SCIPY_BTYPE[btype], output="sos")
                        assert_allclose(
                            ss.sosfilt(got, imp), ss.sosfilt(ref, imp), rtol=0, atol=1e-9
                        )

    def test_impulse_response_high_order_pairing(self) -> None:
        # Failed before the pairing fix (review B1): ellip 8 stop (2.7e-3 at peak 0.167), ellip 9
        # stop (0.22), ellip 10 band; cheby1 10 stop and butter 12 stop passed (regression pins).
        # Reference: SciPy's pairing (nearest zeros to poles, ordered by radius); atol 1e-6 * peak
        # is ~4 decades below the old failures.
        imp = np.zeros(2000)
        imp[0] = 1.0
        cases: list[Any] = [
            (ellip, ss.ellip, (8, 0.5, 40.0), [0.1, 0.9], "stop"),
            (ellip, ss.ellip, (9, 0.5, 40.0), [0.1, 0.9], "stop"),
            (cheby1, ss.cheby1, (10, 1.0), [0.1, 0.9], "stop"),
            (butter, ss.butter, (12,), [0.1, 0.9], "stop"),
            (ellip, ss.ellip, (10, 0.5, 40.0), [0.3, 0.32], "band"),
        ]
        for fn, ref_fn, args, wn, btype in cases:
            with self.subTest(fn=fn.__name__, order=args[0], btype=btype):
                got = fn(*args, wn, btype)
                ref = ref_fn(*args, wn, SCIPY_BTYPE[btype], output="sos")
                y, yr = ss.sosfilt(got, imp), ss.sosfilt(ref, imp)
                assert_allclose(y, yr, rtol=0, atol=1e-6 * np.max(np.abs(yr)))
