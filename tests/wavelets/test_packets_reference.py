"""Acceptance (never shown to the models): energy preservation, best basis for known signals."""

import unittest
from typing import Any

import numpy as np
import pywt
from numpy.testing import assert_allclose

import sigstack.wavelets as wv


def _energy(c: Any) -> float:
    return float(np.sum(np.asarray(c) ** 2))


class Accept(unittest.TestCase):
    def test_energy_preserved_every_level_and_basis(self) -> None:
        # AC: orthonormal packets (periodization, n = 2^k) keep the energy at every level and basis
        x = np.random.default_rng(20).standard_normal(256)
        e = _energy(x)
        for name in ("haar", "db4", "sym8"):
            nodes = wv.wp_decompose(x, name, level=4)
            for level in range(5):
                with self.subTest(name=name, level=level):
                    total = sum(_energy(c) for p, c in nodes.items() if len(p) == level)
                    self.assertAlmostEqual(total / e, 1.0, places=10)
            basis = wv.best_basis(nodes)
            self.assertAlmostEqual(sum(_energy(nodes[p]) for p in basis) / e, 1.0, places=10)
            assert_allclose(
                wv.wp_reconstruct({p: nodes[p] for p in basis}, name, 256), x, atol=1e-10
            )

    def test_tone_lands_in_its_band(self) -> None:
        # AC: a pure tone at a band centre: the best basis puts most energy into a deep node, whose
        # band contains the tone (fine frequency resolution where the signal is narrowband)
        n, fs, level = 512, 2.0, 5
        for index in (3, 11, 20):
            f = (index + 0.5) * (fs / 2) / 2**level
            x = np.cos(2 * np.pi * f * np.arange(n) / fs)
            nodes = wv.wp_decompose(x, "db8", level=level)
            basis = wv.best_basis(nodes)
            top = max(basis, key=lambda p: _energy(nodes[p]))
            lo, hi = wv.wp_frequency_band(top, fs=fs)
            with self.subTest(f=f, top=top):
                self.assertGreaterEqual(len(top), level - 1)
                self.assertTrue(lo <= f <= hi)
                self.assertGreater(_energy(nodes[top]) / _energy(x), 0.5)

    def test_walsh_functions_all_leaves(self) -> None:
        level = 4
        leaves = [
            p.path
            for p in pywt.WaveletPacket(np.zeros(16), "haar", "periodization", level).get_level(
                level
            )
        ]
        for leaf in leaves[::3]:
            wp = pywt.WaveletPacket(None, "haar", mode="periodization", maxlevel=level)
            for q in leaves:
                wp[q] = [2.0 if q == leaf else 0.0]
            x = wp.reconstruct(update=False)
            want = sorted(
                [leaf] + [leaf[:k] + ("d" if leaf[k] == "a" else "a") for k in range(level)]
            )
            with self.subTest(leaf=leaf):
                self.assertEqual(wv.best_basis(wv.wp_decompose(x, "haar", level=level)), want)

    def test_random_bases_odd_length_all_modes(self) -> None:
        rng = np.random.default_rng(21)

        def random_basis(p: str, depth: int) -> list[str]:
            if depth == 0 or rng.random() < 0.35:
                return [p]
            return random_basis(p + "a", depth - 1) + random_basis(p + "d", depth - 1)

        for n in (45, 77):
            x = rng.standard_normal(n)
            for mode in ("periodization", "symmetric", "zero"):
                nodes = wv.wp_decompose(x, "db3", level=3, mode=mode)
                for _ in range(5):
                    basis = random_basis("", 3)
                    with self.subTest(n=n, mode=mode, basis=basis):
                        got = wv.wp_reconstruct({p: nodes[p] for p in basis}, "db3", n, mode=mode)
                        assert_allclose(got, x, rtol=0, atol=1e-10)

    def test_costs_are_additive_shannon(self) -> None:
        # the cost of a split node equals the sum over its children only with the root energy
        x = np.random.default_rng(22).standard_normal(64)
        nodes = wv.wp_decompose(x, "db2", level=1)
        e = _energy(x)
        parts = wv.wp_cost(nodes["a"], energy=e) + wv.wp_cost(nodes["d"], energy=e)
        joint = wv.wp_cost(np.concatenate([nodes["a"], nodes["d"]]), energy=e)
        self.assertAlmostEqual(parts, joint, places=12)
