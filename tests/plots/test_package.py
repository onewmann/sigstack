import importlib
import sys
import unittest
from unittest import mock

import sigstack.plots as plots
from sigstack.plots.figures import plot_band_energies, plot_response, plot_spectrogram
from sigstack.plots.style import STYLE, save_figure

NAMES = ["STYLE", "plot_band_energies", "plot_response", "plot_spectrogram", "save_figure"]


class TestPackage(unittest.TestCase):
    def test_exports(self) -> None:
        self.assertEqual(sorted(plots.__all__), NAMES)
        objs = [STYLE, plot_band_energies, plot_response, plot_spectrogram, save_figure]
        for name, obj in zip(NAMES, objs, strict=True):
            self.assertIs(getattr(plots, name), obj)

    def test_missing_matplotlib_hint(self) -> None:
        # without matplotlib the import fails with an install hint
        keys = [k for k in sys.modules if k == "sigstack.plots" or k.startswith("sigstack.plots.")]
        saved = {k: sys.modules.pop(k) for k in keys}
        try:
            with mock.patch.dict(sys.modules, {"matplotlib": None}):
                with self.assertRaisesRegex(ImportError, r"pip install 'sigstack\[plot\]'"):
                    importlib.import_module("sigstack.plots")
        finally:
            for k in [
                k for k in sys.modules if k == "sigstack.plots" or k.startswith("sigstack.plots.")
            ]:
                del sys.modules[k]
            sys.modules.update(saved)
