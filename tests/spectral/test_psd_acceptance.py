"""Acceptance tests for S4.2 (Welch PSD and band power); never shown to a model."""

import unittest
from typing import Any

import numpy as np
import scipy.signal as ss
from numpy.testing import assert_allclose

from sigstack.core import get_window
from sigstack.signals import multitone
from sigstack.spectral import band_power, welch

NAMES = ["rectangular", "hann", "hamming", "blackman", "blackman_harris", "flattop"]
SCIPY_NAMES = {"rectangular": "boxcar", "blackman_harris": "blackmanharris"}


def _config(rng: np.random.Generator, case: int) -> dict[str, Any]:
    nperseg = int(rng.integers(2, 90))
    return {
        "nperseg": nperseg,
        "hop": int(rng.integers(1, nperseg + 1)),
        "nfft": None if rng.random() < 0.5 else nperseg + int(rng.integers(0, 40)),
        "n": nperseg + int(rng.integers(0, 400)),
        "name": NAMES[case % len(NAMES)],
        "detrend": [None, "constant", "linear"][case % 3],
        "scaling": ["density", "spectrum"][(case // 3) % 2],
        "cplx": case % 5 == 0,
        "lead": [(), (2,), (2, 2)][case % 3],
        "fs": float(rng.uniform(0.5, 2000.0)),
    }


def _signal(rng: np.random.Generator, shape: tuple[int, ...], cplx: bool) -> Any:
    x = rng.standard_normal(shape)
    return x + 1j * rng.standard_normal(shape) if cplx else x


class Accept(unittest.TestCase):
    def test_scipy_grid(self) -> None:
        rng = np.random.default_rng(100)
        for case in range(150):
            c = _config(rng, case)
            with self.subTest(case=case, **{k: v for k, v in c.items() if k != "lead"}):
                x = _signal(rng, (*c["lead"], c["n"]), c["cplx"])
                f, p = welch(
                    x,
                    c["fs"],
                    window=c["name"],
                    nperseg=c["nperseg"],
                    hop=c["hop"],
                    nfft=c["nfft"],
                    detrend=c["detrend"],
                    scaling=c["scaling"],
                )
                f_ref, p_ref = ss.welch(
                    x,
                    fs=c["fs"],
                    window=ss.get_window(SCIPY_NAMES.get(c["name"], c["name"]), c["nperseg"]),
                    nperseg=c["nperseg"],
                    noverlap=c["nperseg"] - c["hop"],
                    nfft=c["nfft"],
                    detrend=False if c["detrend"] is None else c["detrend"],
                    return_onesided=not c["cplx"],
                    scaling=c["scaling"],
                    axis=-1,
                )
                assert_allclose(f, f_ref, rtol=1e-14, atol=0)  # same formula
                # FFT round-off relative to the largest bin
                assert_allclose(p, p_ref, rtol=1e-9, atol=1e-12 * float(np.max(p_ref)))

    def test_parseval_identity_grid(self) -> None:
        rng = np.random.default_rng(101)
        for case in range(60):
            c = _config(rng, case)
            with self.subTest(case=case):
                x = _signal(rng, (*c["lead"], c["n"]), c["cplx"])
                nps, hop = c["nperseg"], c["hop"]
                _, p = welch(
                    x, c["fs"], window=c["name"], nperseg=nps, hop=hop, nfft=c["nfft"], detrend=None
                )
                w = get_window(c["name"], nps)
                k = (c["n"] - nps) // hop + 1
                frames = np.stack([x[..., i * hop : i * hop + nps] for i in range(k)])
                expected = np.mean(np.sum(np.abs(frames * w) ** 2, axis=-1), axis=0)
                expected = expected / np.sum(w**2)
                df = c["fs"] / (c["nfft"] or nps)
                assert_allclose(np.sum(p, axis=-1) * df, expected, rtol=1e-11)  # round-off

    def test_multitone_ground_truth_non_uniform_bands(self) -> None:
        # non-uniform bands, one tone near the geometric centre of each inner band
        edges = np.array([0.0, 20.0, 35.0, 60.0, 110.0, 200.0, 330.0, 470.0, 500.0])
        freqs = [26.3, 46.1, 81.7, 148.9, 257.3, 394.1]
        amps = [0.3, 1.5, 0.7, 1.0, 0.2, 0.9]
        cases = [(2**16, 2048, "hann"), (2**16, 2048, "blackman"), (40000, 1024, "blackman_harris")]
        for n, nperseg, name in cases:
            with self.subTest(n=n, nperseg=nperseg, window=name):
                x, info = multitone(n, 1000.0, freqs, amplitudes=amps, rng=3)
                f, p = welch(x, 1000.0, window=name, nperseg=nperseg, detrend=None)
                bp = band_power(f, p, edges)
                expected = np.zeros(len(edges) - 1)
                for fr, a in zip(info.freqs, info.amplitudes, strict=True):
                    expected[int(np.searchsorted(edges, fr, side="right")) - 1] += a * a / 2
                # leakage of the strongest tone into its neighbours stays below 1e-6 of it
                assert_allclose(bp, expected, rtol=1e-4, atol=1e-6 * info.power)
                # Parseval on the ground truth
                assert_allclose(np.sum(bp), info.power, rtol=1e-6)

    def test_white_noise_level(self) -> None:
        rng = np.random.default_rng(102)
        sigma, fs = 0.5, 200.0
        x = sigma * rng.standard_normal(2**16)
        f, p = welch(x, fs, nperseg=256)
        # one-sided density of white noise: 2 sigma^2 / fs; 511 averaged frames x 127
        # bins give a relative standard error well below 1 %
        assert_allclose(np.mean(p[1:-1]), 2 * sigma**2 / fs, rtol=0.02)
        assert_allclose(np.sum(band_power(f, p, [0.0, fs / 2])), sigma**2, rtol=0.02)

    def test_detrend_invariance(self) -> None:
        rng = np.random.default_rng(103)
        x = rng.standard_normal((2, 2000))
        t = np.arange(2000.0)
        _, p0 = welch(x, nperseg=128, detrend="constant")
        _, p1 = welch(x + 3.0, nperseg=128, detrend="constant")
        # the offset is removed per frame; round-off of the subtraction only
        assert_allclose(p1, p0, rtol=1e-9, atol=1e-12 * float(np.max(p0)))
        _, q0 = welch(x, nperseg=128, hop=50, detrend="linear")
        _, q1 = welch(x + 3.0 - 0.01 * t, nperseg=128, hop=50, detrend="linear")
        assert_allclose(q1, q0, rtol=1e-8, atol=1e-11 * float(np.max(q0)))  # ramp of size 20

    def test_band_power_partitions(self) -> None:
        rng = np.random.default_rng(104)
        for case in range(40):
            c = _config(rng, case)
            with self.subTest(case=case):
                x = _signal(rng, (*c["lead"], c["n"]), c["cplx"])
                f, p = welch(
                    x,
                    c["fs"],
                    window=c["name"],
                    nperseg=c["nperseg"],
                    hop=c["hop"],
                    nfft=c["nfft"],
                    scaling="density",
                )
                inner = np.sort(rng.uniform(-c["fs"] / 2, c["fs"] / 2, 5))
                edges = np.r_[-c["fs"], inner, c["fs"]]
                bp = band_power(f, p, edges)
                self.assertEqual(bp.shape, (*c["lead"], len(edges) - 1))
                df = c["fs"] / (c["nfft"] or c["nperseg"])
                # every bin lies in exactly one band
                assert_allclose(
                    np.sum(bp, axis=-1), np.sum(p, axis=-1) * df, rtol=1e-12, atol=1e-15
                )
                merged = band_power(f, p, np.r_[edges[:2], edges[3:]])
                assert_allclose(merged[..., 1], bp[..., 1] + bp[..., 2], rtol=1e-12, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
