import unittest

import sigstack.spectral as spectral
from sigstack.spectral import shorttime as stft_module

NEW = ["check_cola", "check_nola", "istft", "stft"]


class TestPackage(unittest.TestCase):
    def test_exports(self) -> None:
        self.assertEqual(sorted(stft_module.__all__), NEW)
        self.assertEqual(sorted(spectral.__all__), NEW)
        for name in NEW:
            self.assertIs(getattr(spectral, name), getattr(stft_module, name))

    def test_module_docstrings(self) -> None:
        self.assertTrue(spectral.__doc__)
        self.assertTrue(stft_module.__doc__)
        for name in NEW:
            self.assertTrue(getattr(stft_module, name).__doc__, name)


if __name__ == "__main__":
    unittest.main()
