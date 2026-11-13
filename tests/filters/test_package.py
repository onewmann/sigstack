import importlib
import unittest

import sigstack.filters as filters
from sigstack.filters.firls import firls
from sigstack.filters.firwin import firwin
from sigstack.filters.remez import remez

FIR_NAMES = ["firls", "firwin", "remez"]  # S3.1
IIR_NAMES = ["butter", "cheby1", "cheby2", "ellip", "sos_response"]  # S3.2
WHERE = {
    "butter": "iir",
    "cheby1": "iir",
    "cheby2": "iir",
    "ellip": "iir",
    "firls": "firls",
    "firwin": "firwin",
    "remez": "remez",
    "sos_response": "zpk",
}


class TestPackage(unittest.TestCase):
    def test_exports(self) -> None:
        self.assertEqual(sorted(filters.__all__), sorted(FIR_NAMES + IIR_NAMES))
        for name in filters.__all__:
            module = importlib.import_module(f"sigstack.filters.{WHERE[name]}")
            self.assertIs(getattr(filters, name), getattr(module, name))

    def test_fir_exports(self) -> None:
        # S3.1: the FIR designs stay exported from the subpackage
        for name, obj in zip(FIR_NAMES, [firls, firwin, remez], strict=True):
            self.assertIn(name, filters.__all__)
            self.assertIs(getattr(filters, name), obj)
