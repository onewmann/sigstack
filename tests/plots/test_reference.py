"""Acceptance (never shown to the models).

Real filters from sigstack.metrics, floors instead of -inf, byte-identical saves of every plot type.
"""

import tempfile
import unittest
from pathlib import Path
from typing import cast

import numpy as np
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from numpy.testing import assert_allclose

import sigstack.plots as sp
from sigstack.metrics import freq_response


class Accept(unittest.TestCase):
    def test_response_of_real_filter(self) -> None:
        b = np.ones(4) / 4.0  # zeros on the unit circle at pi/2 and pi
        w, h = freq_response(b, n=256)
        fig = sp.plot_response(w, h)
        mag = np.asarray(fig.axes[0].lines[0].get_ydata(), dtype=float)
        self.assertTrue(np.all(np.isfinite(mag)))
        self.assertGreaterEqual(float(np.min(mag)), -240.0 - 1e-9)
        assert_allclose(np.asarray(fig.axes[0].lines[0].get_xdata(), dtype=float), w)

    def test_spectrogram_floor(self) -> None:
        t = np.arange(4.0)
        f = np.arange(3.0)
        s = np.zeros((3, 4))
        fig = sp.plot_spectrogram(t, f, s)
        assert_allclose(np.asarray(fig.axes[0].collections[0].get_array()), -300.0)

    def test_non_uniform_bands_db(self) -> None:
        edges = np.array([20.0, 250.0, 2000.0, 4000.0, 20000.0])
        e = np.array([4.0, 2.0, 1.0, 0.25])
        ax = sp.plot_band_energies(edges, e, db=True).axes[0]
        bars = cast(BarContainer, ax.containers[0]).patches
        assert_allclose([b.get_width() for b in bars], np.diff(edges))
        assert_allclose([b.get_height() for b in bars], 10 * np.log10(e))

    def test_every_plot_saves_reproducibly(self) -> None:
        w = np.linspace(0.0, np.pi, 128)
        h = np.exp(-1j * 3.0 * w) * np.cos(w / 2.0)
        t = np.linspace(0.0, 2.0, 9)
        f = np.linspace(0.0, 100.0, 6)
        s = np.outer(np.linspace(1.0, 2.0, 6), np.linspace(0.5, 1.5, 9))
        makers = {
            "response": lambda: sp.plot_response(w, h, title="r"),
            "spectrogram": lambda: sp.plot_spectrogram(t, f, s, title="s"),
            "bands": lambda: sp.plot_band_energies([0.0, 1.0, 3.0], [1.0, 2.0], labels=["a", "b"]),
        }
        with tempfile.TemporaryDirectory() as d:
            for name, make in makers.items():
                for suffix in (".png", ".svg", ".pdf"):
                    with self.subTest(plot=name, suffix=suffix):
                        p1, p2 = Path(d) / f"{name}1{suffix}", Path(d) / f"{name}2{suffix}"
                        sp.save_figure(make(), p1)
                        sp.save_figure(make(), p2)
                        self.assertEqual(p1.read_bytes(), p2.read_bytes())

    def test_lists_accepted_and_given_axes_keep_figure(self) -> None:
        fig = Figure()
        ax = fig.add_subplot()
        out = sp.plot_band_energies([0, 1, 2], [3, 4], ax=ax)
        self.assertIs(out, fig)
        self.assertEqual(len(fig.axes), 1)
        resp = sp.plot_response([0.0, 1.0, 2.0], [1.0, 0.5j, -0.25])
        self.assertEqual(len(resp.axes), 2)
