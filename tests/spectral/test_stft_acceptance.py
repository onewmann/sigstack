"""Acceptance tests of S4.1: perfect reconstruction below 1e-10 and the COLA check."""

import itertools
import unittest
from typing import Any

import numpy as np

from sigstack.core import get_window
from sigstack.spectral import check_cola, check_nola, istft, stft

WINDOWS = ["hann", "hamming", "blackman", "blackman_harris", "flattop", "rectangular"]


class Accept(unittest.TestCase):
    def test_reconstruction_error_below_1e_10(self) -> None:
        # AC: reconstruction error < 1e-10 for every window and hop that satisfy NOLA
        rng = np.random.default_rng(41)
        cases = 0
        for name, nperseg in itertools.product(WINDOWS, (1, 2, 7, 16, 63, 128)):
            for hop in sorted(
                {1, max(1, nperseg // 4), max(1, nperseg // 2), nperseg - 1, nperseg}
            ):
                if hop < 1 or not check_nola(name, nperseg, hop):
                    continue
                for nfft in (nperseg, nperseg + 3):
                    for n in (1, nperseg // 2 + 1, 3 * nperseg + 5):
                        x = rng.standard_normal(n)
                        _, _, z = stft(x, window=name, nperseg=nperseg, hop=hop, nfft=nfft)
                        y = istft(z, window=name, nperseg=nperseg, hop=hop, nfft=nfft, n=n)
                        with self.subTest(name=name, nperseg=nperseg, hop=hop, nfft=nfft, n=n):
                            self.assertEqual(y.shape, x.shape)
                            self.assertLess(float(np.max(np.abs(y - x))), 1e-10)
                        cases += 1
        self.assertGreater(cases, 300)

    def test_reconstruction_complex_and_multichannel(self) -> None:
        rng = np.random.default_rng(42)
        shapes: list[tuple[int, ...]] = [(500,), (3, 500), (2, 3, 500)]
        for shape, cplx in itertools.product(shapes, (False, True)):
            x: Any = rng.standard_normal(shape)
            if cplx:
                x = x + 1j * rng.standard_normal(shape)
            w = get_window("kaiser", 48, beta=9.0)
            _, _, z = stft(x, 8000.0, window=w, nperseg=48, hop=12, nfft=64)
            self.assertEqual(z.shape[:-2], shape[:-1])
            y = istft(z, window=w, nperseg=48, hop=12, nfft=64, onesided=not cplx, n=500)
            with self.subTest(shape=shape, cplx=cplx):
                self.assertEqual(y.dtype, np.complex128 if cplx else np.float64)
                self.assertLess(float(np.max(np.abs(y - x))), 1e-10)
                # each channel is transformed on its own
                flat_x = x.reshape(-1, 500)
                flat_z = z.reshape(-1, *z.shape[-2:])
                for c in range(flat_x.shape[0]):
                    _, _, zc = stft(flat_x[c], 8000.0, window=w, nperseg=48, hop=12, nfft=64)
                    # identical arithmetic per channel; 1e-13 allows FFT batching differences
                    np.testing.assert_allclose(flat_z[c], zc, rtol=0, atol=1e-13)

    def test_cola_known_windows(self) -> None:
        # AC: COLA check; periodic Hann is COLA at hop N/2, N/3, N/4; Hamming at N/2;
        # Blackman at N/3; the rectangular window at every divisor of N
        for n in (48, 96):
            for k in (2, 3, 4):
                self.assertTrue(check_cola("hann", n, n // k), (n, k))
            self.assertTrue(check_cola("hamming", n, n // 2))
            self.assertTrue(check_cola("blackman", n, n // 3))
            self.assertFalse(check_cola("blackman", n, n // 2))
            for hop in range(1, n + 1):
                self.assertEqual(check_cola("rectangular", n, hop), n % hop == 0, (n, hop))

    def test_cola_means_constant_overlap_add(self) -> None:
        # the overlap-added windows are constant away from the edges exactly when check_cola holds
        for name, nperseg, hop in itertools.product(WINDOWS, (16, 30), (3, 5, 8, 10, 15)):
            w = get_window(name, nperseg)
            frames = 12
            total = np.zeros((frames - 1) * hop + nperseg)
            for k in range(frames):
                total[k * hop : k * hop + nperseg] += w
            inner = total[nperseg : (frames - 1) * hop]
            constant = bool(np.max(np.abs(inner - np.median(inner))) < 1e-10)
            with self.subTest(name=name, nperseg=nperseg, hop=hop):
                self.assertEqual(check_cola(name, nperseg, hop), constant)

    def test_cola_window_gives_plain_overlap_add(self) -> None:
        # with a COLA window the frames' overlap-add alone (no normalisation) gives the
        # signal times the constant sum(w) / hop in the interior
        x = np.random.default_rng(43).standard_normal(400)
        nperseg, hop = 64, 16
        _, _, z = stft(x, window="hann", nperseg=nperseg, hop=hop)
        w = get_window("hann", nperseg)
        frames = np.fft.irfft(z * np.sum(w), nperseg, axis=0)
        acc = np.zeros((z.shape[1] - 1) * hop + nperseg)
        for k in range(z.shape[1]):
            acc[k * hop : k * hop + nperseg] += frames[:, k]
        inner = acc[nperseg // 2 :][nperseg : len(x) - nperseg]
        # COLA gain sum(w) / hop; 1e-10 allows accumulated rounding over 4 frames
        np.testing.assert_allclose(
            inner, x[nperseg : len(x) - nperseg] * np.sum(w) / hop, rtol=0, atol=1e-10
        )

    def test_frames_are_windowed_dfts(self) -> None:
        # frame k covers x[k * hop - nperseg // 2 : k * hop - nperseg // 2 + nperseg]
        x = np.random.default_rng(44).standard_normal(300)
        nperseg, hop = 50, 20
        w = get_window("hamming", nperseg)
        _, _, z = stft(x, window="hamming", nperseg=nperseg, hop=hop)
        for k in range(2, 13):
            start = k * hop - nperseg // 2
            ref = np.fft.rfft(x[start : start + nperseg] * w) / np.sum(w)
            # direct windowed DFT of the same samples; 1e-13 allows FFT rounding
            np.testing.assert_allclose(z[:, k], ref, rtol=0, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
