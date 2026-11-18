"""Visible tests for sigstack.spectral.shorttime (S4.1, offline part); SciPy is the reference."""

import unittest
import warnings
from typing import Any

import numpy as np
import scipy.signal as ss
from numpy.testing import assert_allclose, assert_array_equal

from sigstack.core import get_window
from sigstack.spectral import check_cola, check_nola, istft, stft

# (n, nperseg, hop, nfft, window, complex input, leading shape)
CONFIGS: list[tuple[int, int, int, int | None, Any, bool, tuple[int, ...]]] = [
    (1000, 64, 32, None, "hann", False, ()),
    (1001, 63, 20, 80, "hamming", False, (3,)),
    (517, 128, 128, None, "rectangular", False, (2, 3)),
    (300, 50, 7, 64, "blackman", True, ()),
    (64, 64, 16, None, ("kaiser", 8.0), False, ()),
    (65, 33, 11, None, "hann", True, (2,)),
    (200, 16, 5, 17, "flattop", False, ()),
]


SCIPY_NAMES = {"rectangular": "boxcar", "blackman_harris": "blackmanharris"}


def _signal(rng: np.random.Generator, shape: tuple[int, ...], cplx: bool) -> Any:
    x = rng.standard_normal(shape)
    if cplx:
        x = x + 1j * rng.standard_normal(shape)
    return x


def _scipy_window(window: Any, nperseg: int) -> Any:
    return ss.get_window(window, nperseg)


class TestStftAgainstScipy(unittest.TestCase):
    def test_matches_scipy_stft(self) -> None:
        rng = np.random.default_rng(1)
        for n, nperseg, hop, nfft, window, cplx, lead in CONFIGS:
            with self.subTest(n=n, nperseg=nperseg, hop=hop, window=window, cplx=cplx):
                x = _signal(rng, (*lead, n), cplx)
                w = _scipy_window(window, nperseg)
                arg = window if isinstance(window, str) else w
                f, t, z = stft(x, 2.5, window=arg, nperseg=nperseg, hop=hop, nfft=nfft)
                fr, tr, zr = ss.stft(
                    x,
                    2.5,
                    window=w,
                    nperseg=nperseg,
                    noverlap=nperseg - hop,
                    nfft=nfft,
                    return_onesided=not cplx,
                )
                self.assertEqual(z.shape, zr.shape)
                # same frames and FFT as SciPy; O(1) values, 1e-12 allows FFT rounding
                assert_allclose(f, fr, rtol=0, atol=1e-12)
                assert_allclose(t, tr, rtol=0, atol=1e-12)
                assert_allclose(z, zr, rtol=0, atol=1e-12)
                self.assertEqual(f.dtype, np.float64)
                self.assertEqual(t.dtype, np.float64)
                self.assertEqual(z.dtype, np.complex128)

    def test_frame_count_and_times(self) -> None:
        # K = (L - nperseg) // hop + 1 with L = n + 2 * (nperseg // 2) rounded up to the hop grid
        for n, nperseg, hop in [(10, 4, 2), (11, 4, 3), (1, 5, 2), (3, 7, 3), (8, 1, 1)]:
            with self.subTest(n=n, nperseg=nperseg, hop=hop):
                length = n + 2 * (nperseg // 2)
                length += (nperseg - length) % hop
                k = (length - nperseg) // hop + 1
                _, t, z = stft(np.ones(n), 4.0, window="hann", nperseg=nperseg, hop=hop)
                self.assertEqual(z.shape, (nperseg // 2 + 1, k))
                # exact: the same integer products divided by the same fs
                assert_allclose(t, np.arange(k) * hop / 4.0, rtol=0, atol=0)

    def test_defaults(self) -> None:
        x = np.random.default_rng(2).standard_normal(1000)
        f, t, z = stft(x)
        fr, tr, zr = ss.stft(x, 1.0, window="hann", nperseg=256, noverlap=128)
        # same frames and FFT as SciPy; O(1) values, 1e-12 allows FFT rounding
        assert_allclose(z, zr, rtol=0, atol=1e-12)
        assert_allclose(f, fr, rtol=0, atol=1e-12)
        assert_allclose(t, tr, rtol=0, atol=1e-12)
        # hop defaults to nperseg // 2, at least 1
        _, _, z1 = stft(np.arange(5.0), nperseg=1)
        self.assertEqual(z1.shape, (1, 5))

    def test_tone_amplitude_scaling(self) -> None:
        # spectrum scaling: Z = FFT(w * frame) / sum(w); a cosine of amplitude A at a bin
        # centre gives |Z| = A / 2 in the frames that lie fully inside the signal
        fs, nperseg, hop, a = 1000.0, 100, 50, 3.0
        n = 2000
        x = a * np.cos(2 * np.pi * 120.0 * np.arange(n) / fs)
        f, _, z = stft(x, fs, window="hann", nperseg=nperseg, hop=hop)
        b = int(np.argmin(np.abs(f - 120.0)))
        inner = z[b, 2:-2]
        # exact at a bin centre (Hann leakage of the -f image is zero); rounding only
        assert_allclose(np.abs(inner), a / 2, rtol=1e-12)

    def test_complex_two_sided_frequencies(self) -> None:
        x = np.exp(2j * np.pi * 0.1 * np.arange(64))
        f, _, z = stft(x, 1.0, nperseg=20, hop=10, nfft=25)
        # same formula as numpy; 1e-15 is one ulp for these values
        assert_allclose(f, np.fft.fftfreq(25, 1.0), rtol=0, atol=1e-15)
        self.assertEqual(z.shape[0], 25)

    def test_integer_and_bool_input(self) -> None:
        xi = np.arange(50)
        _, _, zi = stft(xi, nperseg=8)
        _, _, zf = stft(xi.astype(np.float64), nperseg=8)
        assert_array_equal(zi, zf)
        xb = np.arange(50) % 3 == 0
        _, _, zb = stft(xb, nperseg=8)
        _, _, zbf = stft(xb.astype(np.float64), nperseg=8)
        assert_array_equal(zb, zbf)
        _, _, zl = stft([1.0, 2.0, 3.0, 4.0], nperseg=2)
        self.assertEqual(zl.shape[0], 2)

    def test_input_not_modified(self) -> None:
        x = np.random.default_rng(3).standard_normal((2, 100))
        x0 = x.copy()
        w = np.hanning(16)
        w0 = w.copy()
        stft(x, window=w, nperseg=16, hop=4)
        assert_array_equal(x, x0)
        assert_array_equal(w, w0)


class TestIstft(unittest.TestCase):
    def test_matches_scipy_istft(self) -> None:
        rng = np.random.default_rng(4)
        for n, nperseg, hop, nfft, window, cplx, lead in CONFIGS:
            with self.subTest(n=n, nperseg=nperseg, hop=hop, window=window, cplx=cplx):
                x = _signal(rng, (*lead, n), cplx)
                w = _scipy_window(window, nperseg)
                arg = window if isinstance(window, str) else w
                _, _, z = stft(x, window=arg, nperseg=nperseg, hop=hop, nfft=nfft)
                y = istft(z, window=arg, nperseg=nperseg, hop=hop, nfft=nfft, onesided=not cplx)
                _, yr = ss.istft(
                    z,
                    1.0,
                    window=w,
                    nperseg=nperseg,
                    noverlap=nperseg - hop,
                    nfft=nfft,
                    input_onesided=not cplx,
                )
                self.assertEqual(y.shape, yr.shape)
                # same overlap-add as SciPy; O(1) values, 1e-12 allows FFT rounding
                assert_allclose(y, yr, rtol=0, atol=1e-12)
                self.assertEqual(y.dtype, np.complex128 if cplx else np.float64)

    def test_round_trip(self) -> None:
        rng = np.random.default_rng(5)
        for n, nperseg, hop, nfft, window, cplx, lead in CONFIGS:
            with self.subTest(n=n, nperseg=nperseg, hop=hop, window=window, cplx=cplx):
                x = _signal(rng, (*lead, n), cplx)
                arg = window if isinstance(window, str) else _scipy_window(window, nperseg)
                _, _, z = stft(x, window=arg, nperseg=nperseg, hop=hop, nfft=nfft)
                y = istft(
                    z, window=arg, nperseg=nperseg, hop=hop, nfft=nfft, onesided=not cplx, n=n
                )
                self.assertEqual(y.shape, x.shape)
                # AC: reconstruction error below 1e-10
                self.assertLess(float(np.max(np.abs(y - x))), 1e-10)

    def test_output_length(self) -> None:
        # without n: (K - 1) * hop + nperseg - 2 * (nperseg // 2) samples
        for n, nperseg, hop in [(10, 4, 2), (11, 4, 3), (1, 5, 2), (3, 7, 3), (8, 1, 1)]:
            with self.subTest(n=n, nperseg=nperseg, hop=hop):
                _, _, z = stft(np.ones(n), nperseg=nperseg, hop=hop, window="hamming")
                y = istft(z, nperseg=nperseg, hop=hop, window="hamming")
                k = z.shape[-1]
                self.assertEqual(y.shape, ((k - 1) * hop + nperseg - 2 * (nperseg // 2),))
                # constant input, reconstruction exact up to rounding
                assert_allclose(y[:n], np.ones(n), rtol=0, atol=1e-12)

    def test_nperseg_one(self) -> None:
        x = np.random.default_rng(6).standard_normal(7)
        _, _, z = stft(x, window="rectangular", nperseg=1)
        # one-sample rectangular frames: z equals x, inverse is exact up to one rounding
        assert_allclose(istft(z, window="rectangular", nperseg=1), x, rtol=0, atol=1e-15)

    def test_nola_violation_raises(self) -> None:
        # periodic Hann has w[0] = 0: with hop = nperseg sample 0 of every frame is lost
        _, _, z = stft(np.ones(64), window="hann", nperseg=16, hop=16)
        with self.assertRaises(ValueError):
            istft(z, window="hann", nperseg=16, hop=16)

    def test_n_out_of_range(self) -> None:
        _, _, z = stft(np.ones(20), nperseg=8, hop=4)
        full = istft(z, nperseg=8, hop=4).shape[-1]
        self.assertEqual(istft(z, nperseg=8, hop=4, n=full).shape, (full,))
        for bad in (0, -1, full + 1):
            with self.subTest(n=bad), self.assertRaises(ValueError):
                istft(z, nperseg=8, hop=4, n=bad)
        for bad_type in (2.0, True, "3"):
            with self.subTest(n=bad_type), self.assertRaises(TypeError):
                istft(z, nperseg=8, hop=4, n=bad_type)  # type: ignore[arg-type]

    def test_wrong_frequency_count(self) -> None:
        _, _, z = stft(np.ones(40), nperseg=8, hop=4)
        with self.assertRaises(ValueError):
            istft(z, nperseg=8, hop=4, nfft=10)
        with self.assertRaises(ValueError):
            istft(z, nperseg=8, hop=4, onesided=False)
        with self.assertRaises(ValueError):
            istft(z[0], nperseg=8, hop=4)
        with self.assertRaises(ValueError):
            istft(z[:, :0], nperseg=8, hop=4)


class TestColaNola(unittest.TestCase):
    def test_matches_scipy(self) -> None:
        names = ["hann", "hamming", "blackman", "rectangular", "flattop", "blackman_harris"]
        for name in names:
            for nperseg in (1, 2, 7, 16, 33, 64):
                for hop in range(1, nperseg + 1):
                    with self.subTest(name=name, nperseg=nperseg, hop=hop):
                        ours = get_window(name, nperseg)
                        ref = ss.get_window(SCIPY_NAMES.get(name, name), nperseg)
                        nov = nperseg - hop
                        self.assertEqual(
                            check_cola(name, nperseg, hop), ss.check_COLA(ours, nperseg, nov)
                        )
                        self.assertEqual(
                            check_nola(name, nperseg, hop), ss.check_NOLA(ours, nperseg, nov)
                        )
                        self.assertEqual(
                            check_cola(ref, nperseg, hop), ss.check_COLA(ref, nperseg, nov)
                        )
                        self.assertEqual(
                            check_nola(ref, nperseg, hop), ss.check_NOLA(ref, nperseg, nov)
                        )
                        self.assertIsInstance(check_cola(name, nperseg, hop), bool)
                        self.assertIsInstance(check_nola(ref, nperseg, hop), bool)

    def test_random_windows(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(200):
            nperseg = int(rng.integers(1, 40))
            hop = int(rng.integers(1, nperseg + 1))
            w = rng.standard_normal(nperseg)
            w[rng.random(nperseg) < 0.3] = 0.0
            noverlap = nperseg - hop
            self.assertEqual(check_cola(w, nperseg, hop), ss.check_COLA(w, nperseg, noverlap))
            self.assertEqual(check_nola(w, nperseg, hop), ss.check_NOLA(w, nperseg, noverlap))

    def test_known_cases(self) -> None:
        self.assertTrue(check_cola("hann", 64, 32))
        self.assertTrue(check_cola("hann", 64, 16))
        self.assertTrue(check_cola("hamming", 64, 32))
        self.assertTrue(check_cola("rectangular", 10, 5))
        self.assertFalse(check_cola("blackman", 64, 21))
        self.assertFalse(check_cola("hann", 65, 32))
        self.assertFalse(check_nola("hann", 16, 16))
        self.assertTrue(check_nola("hamming", 16, 16))

    def test_tolerance(self) -> None:
        w = np.ones(8)
        w[0] = 1.0 + 1e-6
        self.assertFalse(check_cola(w, 8, 4))
        self.assertTrue(check_cola(w, 8, 4, tol=1e-5))
        self.assertTrue(check_nola(np.full(4, 1e-3), 4, 4))
        self.assertFalse(check_nola(np.full(4, 1e-3), 4, 4, tol=1e-5))


class TestValidation(unittest.TestCase):
    def test_signal(self) -> None:
        for bad in ([], np.zeros((2, 0)), 3.0, np.array([1.0, np.nan]), [1.0, np.inf], ["a", "b"]):
            with self.subTest(x=bad), self.assertRaises(ValueError):
                stft(bad, nperseg=1)

    def test_integers(self) -> None:
        x = np.ones(32)
        for kw in ({"nperseg": 0}, {"hop": 0}, {"nperseg": 8, "hop": 9}, {"nperseg": 8, "nfft": 7}):
            with self.subTest(**kw), self.assertRaises(ValueError):
                stft(x, **kw)
        for kw2 in (
            {"nperseg": 8.0},
            {"nperseg": True},
            {"hop": 2.5},
            {"nfft": "16"},
            {"hop": False},
        ):
            args: dict[str, Any] = {"nperseg": 8, **kw2}
            with self.subTest(**kw2), self.assertRaises(TypeError):
                stft(x, **args)
        self.assertEqual(stft(x, nperseg=np.int64(8), hop=np.int32(4))[2].shape, (5, 9))

    def test_fs(self) -> None:
        x = np.ones(32)
        for bad in (0.0, -1.0, np.inf, np.nan):
            with self.subTest(fs=bad), self.assertRaises(ValueError):
                stft(x, bad, nperseg=8)
        for bad_type in (True, "1", None, 1j):
            with self.subTest(fs=bad_type), self.assertRaises(TypeError):
                stft(x, bad_type, nperseg=8)  # type: ignore[arg-type]
        f, _, _ = stft(x, np.float32(8.0), nperseg=8)  # type: ignore[arg-type]
        # fs / 2 = 4.0 exactly; default tolerance
        assert_allclose(f[-1], 4.0)
        f0, _, _ = stft(x, np.array(8.0), nperseg=8)  # type: ignore[arg-type]
        assert_allclose(f0[-1], 4.0)

    def test_window(self) -> None:
        x = np.ones(32)
        for bad in (
            np.ones(7),
            np.ones((2, 8)),
            np.array([1.0] * 7 + [np.nan]),
            np.zeros(8),
            "nope",
        ):
            with self.subTest(window=bad), self.assertRaises(ValueError):
                stft(x, window=bad, nperseg=8)
            with self.subTest(window=bad), self.assertRaises(ValueError):
                istft(np.ones((5, 9)), window=bad, nperseg=8)
        with self.assertRaises(ValueError):
            check_cola(np.ones(7), 8, 4)
        with self.assertRaises(ValueError):
            check_nola("nope", 8, 4)

    def test_istft_input(self) -> None:
        with self.assertRaises(ValueError):
            istft(np.array([[1.0, np.nan]] * 5), nperseg=8)
        with self.assertRaises(ValueError):
            istft(np.array([["a"] * 3] * 5), nperseg=8)
        with self.assertRaises(TypeError):
            istft(np.ones((5, 3)), nperseg=8, onesided=1)  # type: ignore[arg-type]

    def test_check_arguments(self) -> None:
        for args in ((8, 0), (8, 9), (0, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                check_cola("hann", *args)
        with self.assertRaises(TypeError):
            check_nola("hann", 8, 2.0)  # type: ignore[arg-type]
        for tol in (0.0, -1.0, np.nan):
            with self.subTest(tol=tol), self.assertRaises(ValueError):
                check_cola("hann", 8, 4, tol=tol)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unittest.main()
