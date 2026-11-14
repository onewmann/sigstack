import unittest
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from sigstack.plots.figures import plot_band_energies, plot_response, plot_spectrogram
from sigstack.plots.style import STYLE


def _x(ax: Axes) -> NDArray[np.float64]:
    return np.asarray(ax.lines[0].get_xdata(), dtype=np.float64)


def _y(ax: Axes) -> NDArray[np.float64]:
    return np.asarray(ax.lines[0].get_ydata(), dtype=np.float64)


def _response() -> tuple[Any, Any]:
    w = np.linspace(0.0, np.pi, 64)
    h = (1.0 + np.exp(-1j * w) + np.exp(-2j * w)) / 3.0  # moving average: zero at 2 pi / 3
    return w, h


def _spectrogram() -> tuple[Any, Any, Any]:
    t = np.linspace(0.0, 1.0, 7)
    f = np.linspace(0.0, 50.0, 5)
    s = np.outer(np.arange(1.0, 6.0), np.arange(1.0, 8.0)) * 1e-3
    s[0, 0] = 0.0
    return t, f, s


class TestPlotResponse(unittest.TestCase):
    def test_data_and_labels(self) -> None:
        # top: 20 log10(max(|h|, 1e-12)); bottom: unwrapped phase; shared x
        w, h = _response()
        fig = plot_response(w, h)
        self.assertIsInstance(fig, Figure)
        top, bottom = fig.axes
        assert_allclose(_x(top), w)
        assert_allclose(
            _y(top),
            20 * np.log10(np.maximum(np.abs(h), 1e-12)),
        )
        assert_allclose(_y(bottom), np.unwrap(np.angle(h)))
        self.assertEqual(top.get_ylabel(), "Magnitude (dB)")
        self.assertEqual(bottom.get_ylabel(), "Phase (rad)")
        self.assertEqual(bottom.get_xlabel(), "Frequency (rad/sample)")

    def test_options(self) -> None:
        w, h = _response()
        fig = plot_response(w * 1000.0 / np.pi, h, hz=True, unwrap=False, title="MA3")
        top, bottom = fig.axes
        self.assertEqual(bottom.get_xlabel(), "Frequency (Hz)")
        assert_allclose(_y(bottom), np.angle(h))
        self.assertEqual(top.get_title(), "MA3")

    def test_into_given_axes(self) -> None:
        w, h = _response()
        fig = Figure()
        a1, a2 = fig.subplots(2, 1)
        self.assertIs(plot_response(w, h, axes=(a1, a2)), fig)
        self.assertEqual((len(a1.lines), len(a2.lines)), (1, 1))

    def test_validation(self) -> None:
        w, h = _response()
        for args in [
            (w[:-1], h),
            (w[:1], h[:1]),
            (np.vstack([w, w]), h),
            (np.r_[w[:-1], np.nan], h),
        ]:
            with self.subTest(shape=np.shape(args[0])), self.assertRaises(ValueError):
                plot_response(*args)


class TestPlotSpectrogram(unittest.TestCase):
    def test_data_and_labels(self) -> None:
        # 10 log10(max(s, 1e-30)) drawn with pcolormesh(t, f, values, shading="nearest") + colorbar
        t, f, s = _spectrogram()
        fig = plot_spectrogram(t, f, s, title="S")
        ax = fig.axes[0]
        mesh = ax.collections[0]
        assert_allclose(np.asarray(mesh.get_array()), 10 * np.log10(np.maximum(s, 1e-30)))
        self.assertEqual(
            (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()), ("Time (s)", "Frequency (Hz)", "S")
        )
        cbar_ax = next(a for a in fig.axes if a is not ax)
        self.assertEqual(cbar_ax.get_ylabel(), "Power (dB)")

    def test_linear_and_given_axes(self) -> None:
        t, f, s = _spectrogram()
        fig = Figure()
        ax = fig.add_subplot()
        self.assertIs(plot_spectrogram(t, f, s, db=False, ax=ax), fig)
        assert_allclose(np.asarray(ax.collections[0].get_array()), s)
        self.assertEqual(next(a for a in fig.axes if a is not ax).get_ylabel(), "Power")

    def test_validation(self) -> None:
        t, f, s = _spectrogram()
        neg = s.copy()
        neg[1, 1] = -1.0
        nan = s.copy()
        nan[2, 2] = np.nan
        cases = [
            (t, f, s.T),
            (t, f, s[0]),
            (t, f, neg),
            (t, f, nan),
            (t[::-1], f, s),
            (t[:1], f, s[:, :1]),
        ]
        for i, args in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(ValueError):
                plot_spectrogram(*args)


class TestPlotBandEnergies(unittest.TestCase):
    def test_bars(self) -> None:
        # bars from edges[:-1] with width diff(edges), align="edge"
        edges = np.array([0.0, 100.0, 300.0, 700.0])
        e = np.array([1.0, 0.5, 0.0])
        fig = plot_band_energies(edges, e)
        ax = fig.axes[0]
        bars = cast(BarContainer, ax.containers[0]).patches
        assert_allclose([b.get_x() for b in bars], edges[:-1])
        assert_allclose([b.get_width() for b in bars], np.diff(edges))
        assert_allclose([b.get_height() for b in bars], e)
        self.assertEqual((ax.get_xlabel(), ax.get_ylabel()), ("Frequency (Hz)", "Energy"))

    def test_db_and_labels(self) -> None:
        edges = np.array([0.0, 100.0, 300.0, 700.0])
        e = np.array([1.0, 0.5, 0.0])
        fig = plot_band_energies(edges, e, db=True, labels=["low", "mid", "high"], title="B")
        ax = fig.axes[0]
        heights = [b.get_height() for b in cast(BarContainer, ax.containers[0]).patches]
        assert_allclose(heights, 10 * np.log10(np.maximum(e, 1e-30)))
        self.assertEqual(ax.get_ylabel(), "Energy (dB)")
        assert_allclose(ax.get_xticks(), [50.0, 200.0, 500.0])
        self.assertEqual([t.get_text() for t in ax.get_xticklabels()], ["low", "mid", "high"])
        self.assertEqual(ax.get_title(), "B")

    def test_validation(self) -> None:
        edges = np.array([0.0, 100.0, 300.0])
        cases: list[tuple[Any, Any, Any]] = [
            (edges, np.array([1.0]), None),
            (np.array([0.0, 300.0, 100.0]), np.array([1.0, 1.0]), None),
            (edges, np.array([1.0, -1.0]), None),
            (edges, np.array([1.0, 1.0]), ["only one"]),
            (np.array([0.0]), np.array([]), None),
        ]
        for i, (ed, en, lab) in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(ValueError):
                plot_band_energies(ed, en, labels=lab)


class TestConsistentStyle(unittest.TestCase):
    def test_same_style_everywhere(self) -> None:
        # every new figure is created inside rc_context(STYLE): same size, label size, grid
        w, h = _response()
        figs = [
            plot_response(w, h),
            plot_spectrogram(*_spectrogram()),
            plot_band_energies(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0])),
        ]
        for fig in figs:
            with self.subTest(fig=len(fig.axes)):
                assert_allclose(fig.get_size_inches(), STYLE["figure.figsize"])
                ax = fig.axes[0]
                self.assertEqual(ax.xaxis.label.get_fontsize(), STYLE["axes.labelsize"])
                self.assertTrue(ax.xaxis.get_gridlines()[0].get_visible())

    def test_style_reaches_the_artists(self) -> None:
        # lines, titles and meshes are created inside rc_context(STYLE) too, also in given axes
        w, h = _response()
        top = plot_response(w, h, title="T").axes[0]
        self.assertEqual(top.lines[0].get_linewidth(), STYLE["lines.linewidth"])
        self.assertEqual(top.title.get_fontsize(), STYLE["axes.titlesize"])
        fig = Figure()
        a1, a2 = fig.subplots(2, 1)
        plot_response(w, h, axes=(a1, a2))
        self.assertEqual(a2.lines[0].get_linewidth(), STYLE["lines.linewidth"])
        mesh = plot_spectrogram(*_spectrogram()).axes[0].collections[0]
        self.assertEqual(mesh.get_cmap().name, STYLE["image.cmap"])

    def test_no_pyplot_figures(self) -> None:
        # object-oriented API only: nothing is registered with pyplot
        before = plt.get_fignums()
        w, h = _response()
        plot_response(w, h)
        plot_spectrogram(*_spectrogram())
        self.assertEqual(plt.get_fignums(), before)
