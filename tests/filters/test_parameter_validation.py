import unittest
from typing import Any

import numpy as np
from numpy.testing import assert_array_equal

from sigstack.filters import butter, cheby1, cheby2, ellip
from sigstack.filters.analog import buttap
from sigstack.filters.zpk import bilinear_zpk, lp2bp_zpk, lp2bs_zpk, lp2hp_zpk, lp2lp_zpk

BAD_POSITIVE: list[Any] = [0.0, -1.0, float("nan"), float("inf"), True, np.True_, "1", 1.0 + 0j]


class TestZpkTransformParameters(unittest.TestCase):
    def setUp(self) -> None:
        z, p, k = buttap(3)
        self.zpk = (z, p, k)

    def test_bad_values_rejected(self) -> None:
        # wo, bw and fs must be real (not bool, str or complex), finite and > 0
        z, p, k = self.zpk
        calls: list[tuple[str, Any]] = [
            ("lp2lp_zpk wo", lambda v: lp2lp_zpk(z, p, k, v)),
            ("lp2hp_zpk wo", lambda v: lp2hp_zpk(z, p, k, v)),
            ("lp2bp_zpk wo", lambda v: lp2bp_zpk(z, p, k, v, 1.0)),
            ("lp2bp_zpk bw", lambda v: lp2bp_zpk(z, p, k, 2.0, v)),
            ("lp2bs_zpk wo", lambda v: lp2bs_zpk(z, p, k, v, 1.0)),
            ("lp2bs_zpk bw", lambda v: lp2bs_zpk(z, p, k, 2.0, v)),
            ("bilinear_zpk fs", lambda v: bilinear_zpk(z, p, k, v)),
        ]
        for name, call in calls:
            for value in BAD_POSITIVE:
                with self.subTest(arg=name, value=repr(value)), self.assertRaises(ValueError):
                    call(value)

    def test_numpy_scalars_accepted(self) -> None:
        z, p, k = self.zpk
        wo: Any = np.int64(2)
        bw: Any = np.float32(0.5)
        for got, want in [
            (lp2lp_zpk(z, p, k, wo), lp2lp_zpk(z, p, k, 2.0)),
            (lp2hp_zpk(z, p, k, wo), lp2hp_zpk(z, p, k, 2.0)),
            (lp2bp_zpk(z, p, k, wo, bw), lp2bp_zpk(z, p, k, 2.0, 0.5)),
            (lp2bs_zpk(z, p, k, wo, bw), lp2bs_zpk(z, p, k, 2.0, 0.5)),
            (bilinear_zpk(z, p, k, wo), bilinear_zpk(z, p, k, 2.0)),
        ]:
            assert_array_equal(got[0], want[0])
            assert_array_equal(got[1], want[1])
            self.assertEqual(got[2], want[2])


class TestDesignParameters(unittest.TestCase):
    def test_bad_types_are_value_errors(self) -> None:
        # Wn, rp, rs and fs: numbers.Real without bool / numpy.bool_; strings, bools and complex
        # values are ValueError like every other bad magnitude (order keeps its TypeError)
        # the functions are typed Any here: the arguments are deliberately of the wrong type
        bu: Any = butter
        c1: Any = cheby1
        c2: Any = cheby2
        el: Any = ellip
        bad: list[tuple[str, Any]] = [
            ("butter Wn str", lambda: bu(4, "0.2")),
            ("butter Wn complex", lambda: bu(4, 0.2 + 0j)),
            ("butter Wn bool", lambda: bu(4, True)),
            ("butter band Wn str", lambda: bu(4, [0.1, "0.2"], "band")),
            ("butter fs str", lambda: bu(4, 10.0, "low", "100")),
            ("butter fs bool", lambda: bu(4, 0.2, "low", True)),
            ("cheby1 rp bool", lambda: c1(4, True, 0.2)),
            ("cheby1 rp numpy bool", lambda: c1(4, np.True_, 0.2)),
            ("cheby1 rp str", lambda: c1(4, "1", 0.2)),
            ("cheby2 rs str", lambda: c2(4, "40", 0.2)),
            ("ellip rp complex", lambda: el(4, 1.0 + 0j, 40.0, 0.2)),
            ("ellip rs bool", lambda: el(4, 1.0, True, 0.2)),
        ]
        for name, call in bad:
            with self.subTest(case=name), self.assertRaises(ValueError):
                call()

    def test_numpy_scalars_accepted(self) -> None:
        wn: Any = np.float64(0.2)
        rp: Any = np.int64(1)
        fs: Any = np.int64(100)
        rs: Any = np.float32(40.0)
        assert_array_equal(butter(4, wn), butter(4, 0.2))
        assert_array_equal(cheby1(4, rp, 0.2), cheby1(4, 1.0, 0.2))
        assert_array_equal(butter(4, 10.0, "low", fs), butter(4, 10.0, "low", 100.0))
        assert_array_equal(
            ellip(4, rp, rs, [0.1, 0.3], "band"),
            ellip(4, 1.0, 40.0, [0.1, 0.3], "band"),
        )
