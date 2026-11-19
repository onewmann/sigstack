"""Visible tests for sigstack.spectral.psd (S4.2); SciPy is the reference."""

import unittest
from typing import Any

import numpy as np
import scipy.signal as ss
from numpy.testing import assert_allclose, assert_array_equal

from sigstack.core import get_window
from sigstack.signals import multitone
from sigstack.spectral import band_power, welch

SCIPY_NAMES = {"rectangular": "boxcar", "blackman_harris": "blackmanharris"}

# (n, nperseg, hop, nfft, window, detrend, scaling, complex input, leading shape)
# detrend and scaling are Any: the grid also feeds them to SciPy
CONFIGS: list[tuple[int, int, int, int | None, str, Any, Any, bool, tuple[int, ...]]] = [
    (1000, 64, 32, None, "hann", "constant", "density", False, ()),
    (1001, 63, 20, 80, "hamming", "linear", "density", False, (3,)),
    (517, 128, 128, None, "rectangular", None, "spectrum", False, (2, 3)),
    (300, 50, 7, 64, "blackman", "constant", "spectrum", True, ()),
    (64, 64, 16, None, "kaiser8", None, "density", False, ()),
    (65, 33, 11, None, "hann", "linear", "density", True, (2,)),
    (200, 16, 5, 17, "flattop", "constant", "density", False, ()),
    (40, 1, 1, None, "rectangular", None, "density", False, ()),
    (99, 99, 99, 128, "blackman_harris", "linear", "spectrum", False, (4,)),
]


def _windows(name: str, nperseg: int) -> tuple[Any, Any]:
    """Window argument for welch and the same window as an array for SciPy."""
    if name == "kaiser8":
        w = get_window("kaiser", nperseg, beta=8.0)
        return w, w
    return name, ss.get_window(SCIPY_NAMES.get(name, name), nperseg)


def _signal(rng: np.random.Generator, shape: tuple[int, ...], cplx: bool) -> Any:
    x = rng.standard_normal(shape)
    if cplx:
        x = x + 1j * rng.standard_normal(shape)
    return x


def _scipy_welch(
    x: Any,
    fs: float,
    w: Any,
    nperseg: int,
    hop: int,
    nfft: int | None,
    detrend: str | None,
    scaling: str,
    cplx: bool,
) -> Any:
    return ss.welch(
        x,
        fs=fs,
        window=w,
        nperseg=nperseg,
        noverlap=nperseg - hop,
        nfft=nfft,
        detrend=False if detrend is None else detrend,
        return_onesided=not cplx,
        scaling=scaling,
        axis=-1,
    )


def _mask_sum(f: Any, p: Any, edges: Any) -> Any:
    """Band powers by explicit masks: bands [lo, hi), the last one [lo, hi]."""
    df = np.sort(f)[1] - np.sort(f)[0]
    out = np.zeros((*p.shape[:-1], len(edges) - 1))
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        upper = f <= hi if i == len(edges) - 2 else f < hi
        out[..., i] = p[..., (f >= lo) & upper].sum(axis=-1) * df
    return out


class TestWelchAgainstScipy(unittest.TestCase):
    def test_matches_scipy_welch(self) -> None:
        rng = np.random.default_rng(1)
        for n, nperseg, hop, nfft, name, detrend, scaling, cplx, lead in CONFIGS:
            with self.subTest(n=n, nperseg=nperseg, hop=hop, window=name, detrend=detrend):
                x = _signal(rng, (*lead, n), cplx)
                ours, ref = _windows(name, nperseg)
                f, p = welch(
                    x,
                    250.0,
                    window=ours,
                    nperseg=nperseg,
                    hop=hop,
                    nfft=nfft,
                    detrend=detrend,
                    scaling=scaling,
                )
                f_ref, p_ref = _scipy_welch(
                    x, 250.0, ref, nperseg, hop, nfft, detrend, scaling, cplx
                )
                # same formula for the frequency grid: only rounding differs
                assert_allclose(f, f_ref, rtol=1e-14, atol=0)
                self.assertEqual(p.shape, p_ref.shape)
                # FFT round-off relative to the largest bin; small bins are absolute-noise bound
                assert_allclose(p, p_ref, rtol=1e-10, atol=1e-12 * float(np.max(p_ref)))

    def test_defaults(self) -> None:
        x = np.random.default_rng(2).standard_normal(1000)
        f, p = welch(x)
        f_ref, p_ref = ss.welch(x)  # fs 1, periodic hann, 256 samples, half overlap, constant
        assert_allclose(f, f_ref, rtol=1e-14, atol=0)  # same formula
        assert_allclose(p, p_ref, rtol=1e-10, atol=1e-12 * float(np.max(p_ref)))  # FFT round-off

    def test_trailing_samples_ignored(self) -> None:
        # frames start at 0, 30, ..., 900 for both lengths: 31 frames, the last ends at 1000
        x = np.random.default_rng(3).standard_normal(1029)
        _, p_long = welch(x, nperseg=100, hop=30)
        _, p_short = welch(x[:1000], nperseg=100, hop=30)
        assert_allclose(p_long, p_short, rtol=1e-14, atol=0)  # identical frames

    def test_complex_two_sided(self) -> None:
        x = _signal(np.random.default_rng(4), (512,), True)
        f, p = welch(x, 1000.0, nperseg=64)
        assert_allclose(f, np.fft.fftfreq(64, 1 / 1000.0), rtol=1e-14, atol=0)  # same formula
        self.assertEqual(p.shape, (64,))
        _, p_ref = ss.welch(x, 1000.0, nperseg=64, return_onesided=False)
        assert_allclose(p, p_ref, rtol=1e-10, atol=1e-12 * float(np.max(p_ref)))  # round-off

    def test_one_sided_folds_two_sided(self) -> None:
        rng = np.random.default_rng(5)
        for nfft in (64, 65):
            with self.subTest(nfft=nfft):
                x = rng.standard_normal((2, 700))
                _, p1 = welch(x, nperseg=60, hop=25, nfft=nfft)
                _, p2 = welch(x.astype(np.complex128), nperseg=60, hop=25, nfft=nfft)
                folded = p2[..., : nfft // 2 + 1].copy()
                last = nfft // 2 if nfft % 2 == 0 else nfft // 2 + 1
                folded[..., 1:last] += p2[..., ::-1][..., : last - 1]
                # sums of the same FFT bins in a different order
                assert_allclose(p1, folded, rtol=1e-12, atol=1e-14 * float(np.max(p1)))

    def test_multichannel_matches_per_channel(self) -> None:
        x = np.random.default_rng(6).standard_normal((2, 3, 500))
        _, p = welch(x, nperseg=50, detrend="linear")
        self.assertEqual(p.shape, (2, 3, 26))
        for i in range(2):
            for j in range(3):
                _, pij = welch(x[i, j], nperseg=50, detrend="linear")
                assert_allclose(p[i, j], pij, rtol=1e-12, atol=0)  # same frames, same FFT

    def test_output_types(self) -> None:
        f, p = welch(np.arange(300, dtype=np.float32), nperseg=32)
        self.assertEqual(f.dtype, np.float64)
        self.assertEqual(p.dtype, np.float64)
        self.assertEqual(f.shape, (17,))
        self.assertEqual(p.shape, (17,))
        f, p = welch(np.ones(40, dtype=np.complex64), nperseg=8)
        self.assertEqual((f.dtype, p.dtype, p.shape), (np.float64, np.float64, (8,)))

    def test_integer_and_bool_input(self) -> None:
        xi = np.random.default_rng(7).integers(-5, 6, 400)
        _, p_int = welch(xi, nperseg=40)
        _, p_flt = welch(xi.astype(np.float64), nperseg=40)
        assert_array_equal(p_int, p_flt)
        xb = xi > 0
        _, p_bool = welch(xb, nperseg=40, detrend=None)
        _, p_ref = welch(xb.astype(np.float64), nperseg=40, detrend=None)
        assert_array_equal(p_bool, p_ref)

    def test_detrend_false_is_none(self) -> None:
        x = np.random.default_rng(8).standard_normal(300) + 3.0
        _, p_false = welch(x, nperseg=30, detrend=False)
        _, p_none = welch(x, nperseg=30, detrend=None)
        assert_array_equal(p_false, p_none)

    def test_input_not_modified(self) -> None:
        x = np.random.default_rng(9).standard_normal((2, 300))
        w = get_window("hamming", 30)
        x0, w0 = x.copy(), w.copy()
        welch(x, window=w, nperseg=30, detrend="linear")
        assert_array_equal(x, x0)
        assert_array_equal(w, w0)


class TestParseval(unittest.TestCase):
    def test_exact_identity(self) -> None:
        # sum(P) * df equals the mean over frames of sum(|w * frame|^2) / sum(w^2)
        rng = np.random.default_rng(10)
        cases = [
            (800, 64, 20, None, "hann", False),
            (801, 65, 65, 100, "hamming", False),
            (500, 40, 13, 41, "flattop", True),
            (90, 90, 1, None, "rectangular", False),
        ]
        for n, nperseg, hop, nfft, name, cplx in cases:
            with self.subTest(n=n, nperseg=nperseg, window=name, cplx=cplx):
                x = _signal(rng, (3, n), cplx)
                _, p = welch(
                    x, 48.0, window=name, nperseg=nperseg, hop=hop, nfft=nfft, detrend=None
                )
                w = get_window(name, nperseg)
                k = (n - nperseg) // hop + 1
                frames = np.stack([x[..., i * hop : i * hop + nperseg] for i in range(k)])
                expected = np.mean(np.sum(np.abs(frames * w) ** 2, axis=-1), axis=0) / np.sum(w**2)
                df = 48.0 / (nfft or nperseg)
                assert_allclose(np.sum(p, axis=-1) * df, expected, rtol=1e-12)  # round-off only

    def test_multitone_total_power(self) -> None:
        x, info = multitone(2**15, 1000.0, [50.0, 123.4, 300.7], amplitudes=[1.0, 0.5, 0.25], rng=7)
        for name in ("hann", "blackman"):
            with self.subTest(window=name):
                f, p = welch(x, 1000.0, window=name, nperseg=1024, detrend=None)
                # tone cross terms leak through the window: below 1e-11 here (SciPy calibration)
                assert_allclose(np.sum(p) * (f[1] - f[0]), info.power, rtol=1e-6)

    def test_spectrum_scaling_tone_power(self) -> None:
        # 100 Hz is bin 20 for fs 1000 and 200 samples; the periodic hann is zero at
        # every other integer bin offset, so the image at -100 Hz does not leak
        t = np.arange(4000) / 1000.0
        x = 2.0 * np.cos(2 * np.pi * 100.0 * t + 0.3)
        f, p = welch(x, 1000.0, nperseg=200, detrend=None, scaling="spectrum")
        self.assertEqual(f[20], 100.0)
        assert_allclose(p[20], 2.0, rtol=1e-10)  # A^2 / 2, round-off only


class TestBandPower(unittest.TestCase):
    def test_simple_sums(self) -> None:
        f = 0.5 * np.arange(10.0)
        p = np.arange(10.0) + 1.0
        # [1.0, 2.5) holds bins 2..4, the last band [2.5, 4.0] bins 5..8, df = 0.5
        assert_allclose(band_power(f, p, [1.0, 2.5, 4.0]), [6.0, 15.0], rtol=1e-15)  # exact sums
        # a single band is the last band: closed at both ends, bins 2..8
        assert_allclose(band_power(f, p, [1.0, 4.0]), [21.0], rtol=1e-15)  # exact sum

    def test_non_uniform_edges(self) -> None:
        x = np.random.default_rng(11).standard_normal((2, 5000))
        f, p = welch(x, 1000.0, nperseg=256)
        edges = [0.0, 10.0, 25.0, 70.0, 200.0, 500.0]
        bp = band_power(f, p, edges)
        self.assertEqual(bp.shape, (2, 5))
        assert_allclose(bp, _mask_sum(f, p, np.asarray(edges)), rtol=1e-12, atol=0)  # same sums

    def test_partition_gives_total(self) -> None:
        x = np.random.default_rng(12).standard_normal(4000)
        f, p = welch(x, 200.0, nperseg=128)
        bp = band_power(f, p, [0.0, 3.3, 7.0, 41.5, 100.0])
        assert_allclose(np.sum(bp), np.sum(p) * (f[1] - f[0]), rtol=1e-12)  # summation order

    def test_additivity(self) -> None:
        x = np.random.default_rng(13).standard_normal(3000)
        f, p = welch(x, 100.0, nperseg=200)
        split = band_power(f, p, [5.0, 12.25, 30.0])
        whole = band_power(f, p, [5.0, 30.0])
        assert_allclose(np.sum(split), whole[0], rtol=1e-12)  # summation order

    def test_multitone_band_power(self) -> None:
        freqs, amps = [50.0, 123.4, 300.7], [1.0, 0.5, 0.25]
        x, info = multitone(2**15, 1000.0, freqs, amplitudes=amps, rng=7)
        f, p = welch(x, 1000.0, nperseg=1024, detrend=None)
        half = 12 * (f[1] - f[0])
        edges = [0.0]
        for fr in info.freqs:
            edges += [fr - half, fr + half]
        edges.append(500.0)
        bp = band_power(f, p, edges)
        # Hann leakage outside +-12 bins: below 1e-7 relative (SciPy calibration)
        assert_allclose(bp[1::2], [a * a / 2 for a in info.amplitudes], rtol=1e-5)
        self.assertLess(float(np.max(bp[0::2])), 1e-6 * info.power)

    def test_two_sided_negative_frequencies(self) -> None:
        t = np.arange(8192) / 1000.0
        x = np.exp(-2j * np.pi * 100.0 * t)
        f, p = welch(x, 1000.0, nperseg=256, detrend=None)
        bp = band_power(f, p, [-500.0, 0.0, 500.0])
        # one complex tone of power 1 (Parseval); Hann leakage across the 25.6 bins to
        # 0 Hz is about 1.2e-9 (SciPy calibration)
        assert_allclose(bp[0], 1.0, rtol=1e-8)
        self.assertLess(float(bp[1]), 1e-8)

    def test_empty_and_outside_bands(self) -> None:
        f = np.arange(10.0)
        p = np.ones(10)
        assert_array_equal(band_power(f, p, [2.2, 2.8]), [0.0])
        assert_array_equal(band_power(f, p, [-5.0, -1.0, 20.0, 30.0]), [0.0, 10.0, 0.0])

    def test_lists_and_types(self) -> None:
        bp = band_power([0.0, 1.0, 2.0, 3.0], [[1, 2, 3, 4]], [0, 2])
        self.assertEqual(bp.dtype, np.float64)
        self.assertEqual(bp.shape, (1, 1))
        assert_allclose(bp, [[6.0]], rtol=1e-15)  # exact sum

    def test_input_not_modified(self) -> None:
        f = np.linspace(0.0, 50.0, 51)
        p = np.random.default_rng(14).random((3, 51))
        e = np.array([0.0, 7.5, 50.0])
        f0, p0, e0 = f.copy(), p.copy(), e.copy()
        band_power(f, p, e)
        assert_array_equal(f, f0)
        assert_array_equal(p, p0)
        assert_array_equal(e, e0)


class TestValidation(unittest.TestCase):
    def test_welch_signal(self) -> None:
        for bad in (
            np.array(["a"] * 300),
            np.float64(1.0),
            np.zeros((3, 0)),
            np.r_[np.zeros(299), np.nan],
            np.r_[np.zeros(299), np.inf],
        ):
            with self.subTest(bad=repr(bad)[:40]), self.assertRaises(ValueError):
                welch(bad, nperseg=16)
        with self.assertRaises(ValueError):
            welch(np.zeros(100), nperseg=101)

    def test_welch_integers(self) -> None:
        x = np.zeros(300)
        type_cases: list[dict[str, Any]] = [
            {"nperseg": True},
            {"nperseg": 16.0},
            {"hop": np.bool_(True)},
            {"nfft": "64"},
        ]
        for kw in type_cases:
            with self.subTest(kw=kw), self.assertRaises(TypeError):
                welch(x, **kw)
        value_cases: list[dict[str, Any]] = [
            {"nperseg": 0},
            {"nperseg": 16, "hop": 0},
            {"nperseg": 16, "hop": 17},
            {"nperseg": 16, "nfft": 15},
        ]
        for kw in value_cases:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                welch(x, **kw)
        _, p = welch(x, nperseg=np.int64(16), hop=np.int32(4), nfft=np.uint8(20))
        self.assertEqual(p.shape, (11,))

    def test_welch_fs(self) -> None:
        x = np.zeros(300)
        for bad in (True, "1", None, 1j):
            with self.subTest(fs=bad), self.assertRaises(TypeError):
                welch(x, bad, nperseg=16)  # type: ignore[arg-type]
        for bad in (0.0, -1.0, float("inf"), float("nan"), 10**400):
            with self.subTest(fs=bad), self.assertRaises(ValueError):
                welch(x, bad, nperseg=16)

    def test_welch_window(self) -> None:
        x = np.ones(300)
        for bad in (np.ones(15), np.ones((16, 1)), np.r_[np.ones(15), np.nan], np.zeros(16)):
            with self.subTest(window=repr(bad)[:40]), self.assertRaises(ValueError):
                welch(x, window=bad, nperseg=16)
        with self.assertRaises(ValueError):
            welch(x, window="no-such-window", nperseg=16)
        zero_sum = np.r_[np.ones(8), -np.ones(8)]
        welch(x, window=zero_sum, nperseg=16)  # density only needs sum(w^2) > 0
        with self.assertRaises(ValueError):
            welch(x, window=zero_sum, nperseg=16, scaling="spectrum")

    def test_welch_options(self) -> None:
        x = np.zeros(300)
        option_cases: list[dict[str, Any]] = [
            {"detrend": "quadratic"},
            {"detrend": True},
            {"detrend": 1},
            {"detrend": ["linear"]},
            {"scaling": "psd"},
            {"scaling": None},
        ]
        for kw in option_cases:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                welch(x, nperseg=16, **kw)

    def test_band_power_f(self) -> None:
        p = np.ones(4)
        for bad in (
            np.zeros((2, 2)),
            [1.0],
            [0.0, 1.0, 1.0, 2.0],
            [0.0, 1.0, 2.5, 3.0],
            [3.0, 2.0, 1.0, 1.0],
            [0.0, 1.0, np.nan, 3.0],
            ["a", "b", "c", "d"],
            [0j, 1j, 2j, 3j],
        ):
            with self.subTest(f=repr(bad)[:40]), self.assertRaises(ValueError):
                band_power(bad, p[: np.size(bad)] if np.ndim(bad) == 1 else p, [0.0, 1.0])
        # fftfreq order is fine: only the sorted copy must be uniform; [-4, 0) and [0, 4]
        f = np.fft.fftfreq(8, 0.125)
        assert_allclose(band_power(f, np.ones(8), [-4.0, 0.0, 4.0]), [4.0, 4.0], rtol=1e-15)

    def test_band_power_psd(self) -> None:
        f = np.arange(4.0)
        for bad in (
            np.ones(3),
            np.ones((2, 5)),
            np.r_[1.0, 1.0, -0.5, 1.0],
            np.r_[1.0, np.inf, 1.0, 1.0],
            np.ones(4, dtype=complex),
            np.float64(1.0),
        ):
            with self.subTest(psd=repr(bad)[:40]), self.assertRaises(ValueError):
                band_power(f, bad, [0.0, 2.0])

    def test_band_power_edges(self) -> None:
        f, p = np.arange(4.0), np.ones(4)
        for bad in (
            [1.0],
            [[0.0, 1.0]],
            [0.0, 0.0, 1.0],
            [2.0, 1.0],
            [0.0, np.inf],
            [0.0, np.nan],
            ["0", "1"],
            [0j, 1j],
        ):
            with self.subTest(edges=repr(bad)[:40]), self.assertRaises(ValueError):
                band_power(f, p, bad)


class TestReviewFixes(unittest.TestCase):
    def test_bins_on_an_edge_go_to_the_upper_band(self) -> None:
        # rfftfreq gives 4.999999999999999 for the 5 Hz bin here; it must start [5, 10)
        x = np.random.default_rng(20).standard_normal(5000)
        f, p = welch(x, 125.0, nperseg=700, hop=350)
        spike = np.zeros_like(p)
        spike[28] = 1.0
        df = 125.0 / 700
        assert_allclose(band_power(f, spike, [0.0, 5.0, 10.0, 62.5]), [0.0, df, 0.0], rtol=1e-15)
        for fs, nfft in ((125.0, 700), (1000.0, 999), (44100.0, 1000), (3.0, 7)):
            ff = np.fft.rfftfreq(nfft, 1 / fs)
            for k in range(1, nfft // 2):
                with self.subTest(fs=fs, nfft=nfft, k=k):
                    one = np.zeros(ff.shape[0])
                    one[k] = 1.0
                    edge = k * fs / nfft  # the bin's nominal frequency as an edge
                    bp = band_power(ff, one, [0.0, edge, fs / 2])
                    assert_allclose(bp, [0.0, fs / nfft], rtol=1e-12)  # spacing round-off

    def test_linspace_grid(self) -> None:
        f = np.linspace(0.0, 50.0, 101)
        bp = band_power(f, np.ones(101), [0.0, 10.0, 20.0, 50.0])
        assert_allclose(bp, [10.0, 10.0, 30.5], rtol=1e-12)  # 20, 20 and 61 bins of 0.5

    def test_default_hop_for_odd_nperseg(self) -> None:
        # default hop is nperseg // 2 = 32 here, i.e. SciPy's noverlap = 33
        x = np.random.default_rng(21).standard_normal(1000)
        _, p = welch(x, nperseg=65)
        _, p_ref = ss.welch(x, nperseg=65, noverlap=33)
        assert_allclose(p, p_ref, rtol=1e-10, atol=1e-12 * float(np.max(p_ref)))  # round-off

    def test_non_contiguous_and_fortran_input(self) -> None:
        base = np.random.default_rng(22).standard_normal((400, 3))
        _, p_ref = welch(np.ascontiguousarray(base.T), nperseg=40, detrend="linear")
        for x in (base.T, np.asfortranarray(base.T), base.T[:, ::-1][:, ::-1]):
            _, p = welch(x, nperseg=40, detrend="linear")
            assert_allclose(p, p_ref, rtol=1e-12, atol=0)  # same values, other memory layout

    def test_window_with_negative_values(self) -> None:
        x = np.random.default_rng(23).standard_normal(600)
        w = get_window("hann", 48) - 0.2
        for scaling in ("density", "spectrum"):
            with self.subTest(scaling=scaling):
                _, p = welch(x, 10.0, window=w, nperseg=48, scaling=scaling)
                _, p_ref = ss.welch(x, 10.0, window=w, nperseg=48, scaling=scaling)
                # FFT round-off relative to the largest bin
                assert_allclose(p, p_ref, rtol=1e-10, atol=1e-12 * float(np.max(p_ref)))

    def test_nperseg_one_linear_detrend(self) -> None:
        x = np.random.default_rng(24).standard_normal(50)
        _, p = welch(x, nperseg=1, detrend="linear")
        assert_array_equal(p, np.zeros(1))  # a one-sample frame minus its mean is zero

    def test_extreme_fs_does_not_overflow_the_scale(self) -> None:
        x = np.random.default_rng(25).standard_normal(500)
        _, p1 = welch(x, 1.0, nperseg=64)
        _, p_big = welch(x, 1e307, nperseg=64)  # 1e307 * sum(w**2) = 2.4e308 overflows
        assert_allclose(p_big, p1 / 1e307, rtol=1e-12)  # division round-off only
        self.assertTrue(np.all(p_big > 0))

    def test_detrend_numpy_false(self) -> None:
        x = np.random.default_rng(26).standard_normal(300) + 2.0
        _, p_np = welch(x, nperseg=30, detrend=np.False_)  # type: ignore[arg-type]
        _, p_none = welch(x, nperseg=30, detrend=None)
        assert_array_equal(p_np, p_none)
        with self.assertRaises(ValueError):
            welch(x, nperseg=30, detrend=np.True_)  # type: ignore[arg-type]

    def test_spectrum_scaling_complex_exponential(self) -> None:
        # bin 20 of 200 at fs 1000: 100 Hz; |A|^2 at its bin, nothing at -100 Hz
        n = np.arange(4000)
        x = 2.0 * np.exp(2j * np.pi * 100.0 * n / 1000.0)
        f, p = welch(x, 1000.0, nperseg=200, detrend=None, scaling="spectrum")
        self.assertEqual(f[20], 100.0)
        assert_allclose(p[20], 4.0, rtol=1e-10)  # A^2, round-off only
        self.assertLess(float(p[180]), 1e-20)  # periodic Hann: no leakage to -100 Hz


if __name__ == "__main__":
    unittest.main()
