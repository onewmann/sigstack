import math
import unittest
from typing import Any

import numpy as np
import pywt
from numpy.testing import assert_allclose

from sigstack.wavelets.dwt import dwt_max_level
from sigstack.wavelets.packets import (
    best_basis,
    wp_cost,
    wp_decompose,
    wp_frequency_band,
    wp_frequency_order,
    wp_reconstruct,
)

MODES = ("periodization", "symmetric", "zero")


def _walsh_signal(leaf: str, level: int) -> Any:
    # Haar packets at full depth are a Walsh basis: one unit coefficient at one leaf
    wp = pywt.WaveletPacket(None, "haar", mode="periodization", maxlevel=level)
    for node in pywt.WaveletPacket(np.zeros(2**level), "haar", "periodization", level).get_level(
        level
    ):
        wp[node.path] = [1.0 if node.path == leaf else 0.0]
    return wp.reconstruct(update=False)


class TestDecompose(unittest.TestCase):
    def test_matches_pywavelets(self) -> None:
        # node[p + "a"], node[p + "d"] = dwt(node[p]) down to `level`
        rng = np.random.default_rng(0)
        for name in ("haar", "db2", "sym4"):
            for n in (16, 37, 64):
                x = rng.standard_normal(n)
                for mode in MODES:
                    with self.subTest(name=name, n=n, mode=mode):
                        nodes = wp_decompose(x, name, level=3, mode=mode)
                        ref = pywt.WaveletPacket(x, name, mode=mode, maxlevel=3)
                        self.assertEqual(len(nodes), 1 + 2 + 4 + 8)
                        for path, c in nodes.items():
                            want = x if path == "" else ref[path].data
                            self.assertEqual(c.dtype, np.float64)
                            assert_allclose(c, want, rtol=0, atol=1e-10)

    def test_key_order_and_default_level(self) -> None:
        nodes = wp_decompose(np.arange(32.0), "db2", level=2)
        self.assertEqual(list(nodes), ["", "a", "d", "aa", "ad", "da", "dd"])
        deep = wp_decompose(np.ones(64), "db2")
        self.assertEqual(max(len(p) for p in deep), dwt_max_level(64, 4))
        self.assertEqual(list(wp_decompose([1, 2, 3], "haar", level=0)), [""])

    def test_validation(self) -> None:
        bad: list[Any] = [
            lambda: wp_decompose(np.array([]), "haar"),
            lambda: wp_decompose(np.ones((4, 2)), "haar"),
            lambda: wp_decompose(np.ones(8), "haar", level=-1),
            lambda: wp_decompose(np.ones(8), "haar", level=True),
            lambda: wp_decompose(np.ones(8), "haar", mode="reflect"),
            lambda: wp_decompose(np.ones(8), "coif1"),
        ]
        for i, call in enumerate(bad):
            with self.subTest(case=i), self.assertRaises(ValueError):
                call()


class TestFrequencyOrder(unittest.TestCase):
    def test_matches_pywavelets(self) -> None:
        self.assertEqual(wp_frequency_order(0), [""])
        for level in range(1, 6):
            ref = pywt.WaveletPacket(np.zeros(64), "db2", "periodization", 5).get_level(
                level, "freq"
            )
            self.assertEqual(wp_frequency_order(level), [node.path for node in ref])

    def test_bands(self) -> None:
        # band of the node at index i of the frequency order: [i, i + 1] * (fs / 2) / 2**level
        self.assertEqual(wp_frequency_band("a"), (0.0, 0.5))
        self.assertEqual(wp_frequency_band("d", fs=100.0), (25.0, 50.0))
        self.assertEqual(wp_frequency_band("da"), (0.75, 1.0))  # order: aa, ad, dd, da
        self.assertEqual(wp_frequency_band(""), (0.0, 1.0))
        for bad in ("x", "ab"):
            with self.subTest(path=bad), self.assertRaises(ValueError):
                wp_frequency_band(bad)
        with self.assertRaises(ValueError):
            wp_frequency_band("a", fs=0.0)


class TestReconstruct(unittest.TestCase):
    def test_any_basis_gives_x(self) -> None:
        rng = np.random.default_rng(1)
        bases = [[""], ["a", "d"], ["a", "da", "ddd", "dda"], ["aaa", "aad", "ad", "d"]]
        for n in (32, 37):
            x = rng.standard_normal(n)
            for mode in MODES:
                nodes = wp_decompose(x, "db2", level=3, mode=mode)
                for basis in bases:
                    with self.subTest(n=n, mode=mode, basis=basis):
                        got = wp_reconstruct({p: nodes[p] for p in basis}, "db2", n, mode=mode)
                        self.assertEqual(got.shape, (n,))
                        assert_allclose(got, x, rtol=0, atol=1e-10)

    def test_matches_pywavelets_basis(self) -> None:
        x = np.random.default_rng(2).standard_normal(64)
        nodes = wp_decompose(x, "sym4", level=3)
        ref = pywt.WaveletPacket(None, "sym4", mode="periodization", maxlevel=3)
        basis = ["a", "da", "ddd", "dda"]
        for p in basis:
            ref[p] = nodes[p]
        want = ref.reconstruct(update=False)
        assert_allclose(wp_reconstruct({p: nodes[p] for p in basis}, "sym4", 64), want, atol=1e-10)

    def test_invalid_bases(self) -> None:
        nodes = wp_decompose(np.ones(16), "haar", level=2)
        bad = [
            ["a", "aa", "ad", "d"],  # overlap
            ["", "a"],  # root plus another key
            ["a", "da"],  # gap
            [],
        ]
        for basis in bad:
            with self.subTest(basis=basis), self.assertRaises(ValueError):
                wp_reconstruct({p: nodes[p] for p in basis}, "haar", 16)
        with self.assertRaises(ValueError):
            wp_reconstruct({"a": nodes["a"], "x": nodes["d"]}, "haar", 16)
        with self.assertRaises(ValueError):  # wrong node length for n = 16
            wp_reconstruct({"a": nodes["a"][:-1], "d": nodes["d"]}, "haar", 16)
        with self.assertRaises(ValueError):
            wp_reconstruct({"a": nodes["a"], "d": nodes["d"]}, "haar", 0)


class TestCost(unittest.TestCase):
    def test_values(self) -> None:
        q = np.array([9.0, 16.0]) / 25.0
        self.assertAlmostEqual(wp_cost(np.array([3.0, 4.0])), float(-np.sum(q * np.log(q))))
        self.assertAlmostEqual(
            wp_cost(np.array([3.0, 0.0]), energy=25.0), -(9 / 25) * math.log(9 / 25)
        )
        self.assertEqual(wp_cost(np.zeros(4)), 0.0)
        self.assertAlmostEqual(wp_cost(np.array([3.0, 0.0, 4.0]), "log_energy"), math.log(144.0))
        self.assertEqual(wp_cost(np.array([3.0, -4.0]), "l1"), 7.0)
        for bad in (
            lambda: wp_cost(np.ones(3), "entropy"),
            lambda: wp_cost(np.ones(3), energy=-1.0),
        ):
            with self.assertRaises(ValueError):
                bad()


class TestBestBasis(unittest.TestCase):
    def test_impulse_stays_at_the_root(self) -> None:
        # one non-zero sample: Shannon cost 0 at the root, nothing beats it, ties keep the root
        x = np.zeros(64)
        x[10] = 3.0
        self.assertEqual(best_basis(wp_decompose(x, "db2", level=4)), [""])

    def test_walsh_function(self) -> None:
        # all energy in one full-depth leaf; zero siblings stay as coarse as possible
        for leaf in ("dad", "aaa", "ddd"):
            x = _walsh_signal(leaf, 3)
            want = sorted([leaf] + [leaf[:k] + ("d" if leaf[k] == "a" else "a") for k in range(3)])
            with self.subTest(leaf=leaf):
                self.assertEqual(best_basis(wp_decompose(x, "haar", level=3)), want)

    def test_callable_cost_and_validation(self) -> None:
        x = _walsh_signal("ad", 2)
        nodes = wp_decompose(x, "haar", level=2)

        def count(c: Any) -> float:
            return float(np.count_nonzero(np.abs(c) > 1e-9))

        self.assertEqual(best_basis(nodes, cost=count), ["aa", "ad", "d"])
        with self.assertRaises(ValueError):
            best_basis({"a": nodes["a"], "d": nodes["d"]})  # no root
        with self.assertRaises(ValueError):
            best_basis({"": nodes[""], "a": nodes["a"]})  # missing sibling


class TestReviewFixes(unittest.TestCase):
    def test_numpy_integers_for_level_and_n(self) -> None:
        x = np.random.default_rng(3).standard_normal(32)
        level: Any = np.int64(2)
        n: Any = np.int32(32)
        nodes = wp_decompose(x, "db2", level=level)
        self.assertEqual(max(len(p) for p in nodes), 2)
        got = wp_reconstruct({"a": nodes["a"], "d": nodes["d"]}, "db2", n)
        assert_allclose(got, x, atol=1e-10)
        self.assertEqual(wp_frequency_order(level), ["aa", "ad", "dd", "da"])

    def test_string_fs_rejected(self) -> None:
        bad: Any = "2"
        with self.assertRaises(ValueError):
            wp_frequency_band("a", fs=bad)

    def test_best_basis_on_a_non_uniform_tree(self) -> None:
        # a pruned tree ("d" is a leaf at depth 1) is valid input: leaves are nodes without children
        x = np.random.default_rng(4).standard_normal(32)
        nodes = wp_decompose(x, "haar", level=2)
        pruned = {p: nodes[p] for p in ("", "a", "d", "aa", "ad")}
        basis = best_basis(pruned)
        self.assertTrue(set(basis) <= set(pruned))
        self.assertEqual(sum(2 ** (2 - len(p)) for p in basis), 4)
