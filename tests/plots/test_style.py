import tempfile
import unittest
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from sigstack.plots.style import STYLE, save_figure

EXPECTED_STYLE: dict[str, object] = {
    "figure.figsize": (6.4, 4.0),
    "figure.dpi": 100.0,
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.0,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.2,
    "image.cmap": "viridis",
    "svg.hashsalt": "sigstack",
}


def _figure() -> Figure:
    fig = Figure(figsize=(4.0, 3.0))
    ax = fig.add_subplot()
    x = np.linspace(0.0, 1.0, 50)
    ax.plot(x, np.sin(6.0 * x), label="sin")
    ax.set_title("test")
    ax.legend()
    return fig


class TestStyle(unittest.TestCase):
    def test_style_entries(self) -> None:
        self.assertEqual(STYLE, EXPECTED_STYLE)

    def test_save_is_byte_identical(self) -> None:
        # metadata without dates or versions, fixed svg hash salt: same figure twice, same bytes
        with tempfile.TemporaryDirectory() as d:
            for suffix in (".png", ".svg", ".pdf"):
                with self.subTest(suffix=suffix):
                    a, b = Path(d) / f"a{suffix}", Path(d) / f"b{suffix}"
                    save_figure(_figure(), a)
                    save_figure(_figure(), str(b))
                    self.assertGreater(a.stat().st_size, 100)
                    self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_format_from_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.PNG"
            save_figure(_figure(), p)
            self.assertEqual(p.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            q = Path(d) / "x.svg"
            save_figure(_figure(), q, dpi=50)
            self.assertIn(b"<svg", q.read_bytes()[:400])

    def test_unknown_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            for name in ("x.jpg", "x", "x.eps"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    save_figure(_figure(), Path(d) / name)
