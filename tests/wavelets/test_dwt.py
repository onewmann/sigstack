import unittest

import numpy as np
import pywt
from numpy.testing import assert_allclose

from sigstack.wavelets.dwt import dwt, dwt_max_level, idwt, wavedec, waverec

MODES = ("symmetric", "zero", "periodization")
NAMES = ("haar", "db2", "db4", "sym5", "sym8", "db10")
LENGTHS = (1, 2, 5, 8, 31, 64)


class TestDwt(unittest.TestCase):
    def test_dwt_matches_pywavelets(self) -> None:
        # symmetric/zero: K = (n + L - 1) // 2, cA[k] = sum_m dec_lo[m] x~[2k + 1 - m];
        # periodization: odd n padded with x[n-1], cA[k] = sum_m dec_lo[m] xp[(2k + L/2 - m) mod P]
        rng = np.random.default_rng(0)
        for name in NAMES:
            for n in LENGTHS:
                x = rng.standard_normal(n)
                for mode in MODES:
                    with self.subTest(name=name, n=n, mode=mode):
                        ca, cd = dwt(x, name, mode=mode)
                        ra, rd = pywt.dwt(x, name, mode=mode)
                        self.assertEqual((ca.shape, cd.shape), (ra.shape, rd.shape))
                        self.assertEqual((ca.dtype, cd.dtype), (np.float64, np.float64))
                        assert_allclose(ca, ra, rtol=0, atol=1e-10)
                        assert_allclose(cd, rd, rtol=0, atol=1e-10)

    def test_idwt_matches_pywavelets(self) -> None:
        # symmetric/zero: length 2K - L + 2 (must be >= 1); periodization: length 2K
        rng = np.random.default_rng(1)
        for name in NAMES:
            for k in (1, 3, 4, 16):
                ca, cd = rng.standard_normal(k), rng.standard_normal(k)
                for mode in MODES:
                    if mode != "periodization" and 2 * k - pywt.Wavelet(name).dec_len + 2 < 1:
                        continue  # too short for the filter: ValueError (test_validation)
                    with self.subTest(name=name, k=k, mode=mode):
                        got = idwt(ca, cd, name, mode=mode)
                        ref = pywt.idwt(ca, cd, name, mode=mode)
                        self.assertEqual((got.shape, got.dtype), (ref.shape, np.float64))
                        assert_allclose(got, ref, rtol=0, atol=1e-10)

    def test_default_mode_is_symmetric(self) -> None:
        x = np.arange(10.0) ** 2
        ca, cd = dwt(x, "db3")
        ra, rd = pywt.dwt(x, "db3", mode="symmetric")
        assert_allclose(ca, ra, rtol=0, atol=1e-10)
        assert_allclose(cd, rd, rtol=0, atol=1e-10)
        assert_allclose(idwt(ca, cd, "db3"), pywt.idwt(ra, rd, "db3"), rtol=0, atol=1e-10)

    def test_single_level_reconstruction(self) -> None:
        # idwt(dwt(x)) gives x back
        # (symmetric/zero: the first n samples of a length n or n + 1 result)
        rng = np.random.default_rng(2)
        for name in NAMES:
            for n in (2, 7, 32):
                x = rng.standard_normal(n)
                for mode in MODES:
                    with self.subTest(name=name, n=n, mode=mode):
                        y = idwt(*dwt(x, name, mode=mode), name, mode=mode)
                        assert_allclose(y[:n], x, rtol=0, atol=1e-10)

    def test_wavedec_waverec_match_pywavelets(self) -> None:
        # [cA_level, cD_level, ..., cD_1]; level None = dwt_max_level(n, L);
        # waverec drops the last value of a when len(a) == len(d) + 1
        rng = np.random.default_rng(3)
        for name in ("haar", "db3", "sym6"):
            for n in (37, 64, 100):
                x = rng.standard_normal(n)
                for mode in MODES:
                    for level in (None, 1, 2):
                        with self.subTest(name=name, n=n, mode=mode, level=level):
                            got = wavedec(x, name, mode=mode, level=level)
                            ref = pywt.wavedec(x, name, mode=mode, level=level)
                            self.assertEqual([len(c) for c in got], [len(c) for c in ref])
                            for g, r in zip(got, ref, strict=True):
                                assert_allclose(g, r, rtol=0, atol=1e-10)
                            rec = waverec(got, name, mode=mode)
                            want = pywt.waverec(ref, name, mode=mode)
                            assert_allclose(rec, want, rtol=0, atol=1e-10)
                            assert_allclose(rec[:n], x, rtol=0, atol=1e-10)

    def test_level_zero_and_single_list(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        out = wavedec(x, "db2", level=0)
        self.assertEqual(len(out), 1)
        assert_allclose(out[0], x)
        assert_allclose(waverec([x], "db2"), x)

    def test_dwt_max_level(self) -> None:
        # largest j >= 0 with (L - 1) * 2**j <= n
        for n in (1, 2, 3, 5, 7, 8, 100, 1023, 1024, 4096):
            for filt in (2, 4, 6, 12, 20):
                with self.subTest(n=n, filt=filt):
                    self.assertEqual(dwt_max_level(n, filt), pywt.dwt_max_level(n, filt))
        self.assertIsInstance(dwt_max_level(1024, 4), int)

    def test_list_and_int_input(self) -> None:
        ca, cd = dwt([1, 2, 3, 4], "haar", mode="periodization")
        assert_allclose(ca, np.array([3.0, 7.0]) / np.sqrt(2.0), rtol=0, atol=1e-12)
        assert_allclose(cd, np.array([-1.0, -1.0]) / np.sqrt(2.0), rtol=0, atol=1e-12)
        self.assertEqual(ca.dtype, np.float64)

    def test_input_not_modified(self) -> None:
        x = np.linspace(0.0, 1.0, 9)
        keep = x.copy()
        wavedec(x, "sym4", mode="symmetric")
        assert_allclose(x, keep, rtol=0, atol=0)

    def test_validation(self) -> None:
        good = np.ones(8)
        bad_calls = [
            lambda: dwt(np.array([]), "db2"),
            lambda: dwt(np.ones((4, 2)), "db2"),
            lambda: dwt(np.array([1.0, np.nan]), "db2"),
            lambda: dwt(np.array([1.0 + 1.0j, 2.0]), "db2"),
            lambda: dwt(good, "db2", mode="reflect"),
            lambda: dwt(good, "coif1"),
            lambda: idwt(np.ones(3), np.ones(4), "db2"),
            lambda: idwt(np.array([]), np.array([]), "db2"),
            lambda: idwt(np.ones(3), np.ones(3), "db2", mode="smooth"),
            lambda: idwt(np.ones(1), np.ones(1), "db2"),  # 2K - L + 2 = 0
            lambda: idwt(np.ones(4), np.ones(4), "db10", mode="zero"),
            lambda: wavedec(good, "db2", level=-1),
            lambda: waverec([], "db2"),
            lambda: waverec([np.ones(3), np.ones(5)], "db2"),
            lambda: dwt_max_level(0, 4),
            lambda: dwt_max_level(8, 1),
        ]
        for i, call in enumerate(bad_calls):
            with self.subTest(case=i), self.assertRaises(ValueError):
                call()
