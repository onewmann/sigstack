import unittest

import sigstack.filters as filters
from sigstack.filters.firls import firls
from sigstack.filters.firwin import firwin
from sigstack.filters.remez import remez


class TestPackage(unittest.TestCase):
    def test_exports(self) -> None:
        self.assertEqual(sorted(filters.__all__), ["firls", "firwin", "remez"])
        for name, obj in zip(["firls", "firwin", "remez"], [firls, firwin, remez], strict=True):
            self.assertIs(getattr(filters, name), obj)
