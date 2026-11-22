import unittest

import sigstack.spectral as spectral
from sigstack.spectral import psd as psd_module
from sigstack.spectral import shorttime

NEW = ["band_power", "welch"]
ALL = ["band_power", "check_cola", "check_nola", "istft", "stft", "welch"]


class TestPackage(unittest.TestCase):
    def test_exports(self) -> None:
        self.assertEqual(sorted(psd_module.__all__), NEW)
        self.assertEqual(spectral.__all__, ALL)
        for name in NEW:
            self.assertIs(getattr(spectral, name), getattr(psd_module, name))
        for name in ["check_cola", "check_nola", "istft", "stft"]:
            self.assertIs(getattr(spectral, name), getattr(shorttime, name))

    def test_module_docstrings(self) -> None:
        self.assertTrue(spectral.__doc__)
        self.assertIn("welch", spectral.__doc__ or "")
        self.assertTrue(psd_module.__doc__)
        for name in NEW:
            self.assertTrue(getattr(psd_module, name).__doc__, name)


if __name__ == "__main__":
    unittest.main()
