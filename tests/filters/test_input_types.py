import unittest
from typing import Any

import numpy as np
from numpy.testing import assert_array_equal

from sigstack.filters.firls import firls
from sigstack.filters.remez import remez

BANDS = [0.0, 0.3, 0.5, 1.0]
DESIRED = [1.0, 1.0, 0.0, 0.0]


class TestNumpyScalars(unittest.TestCase):
    def test_numpy_integer_numtaps(self) -> None:
        # NumPy integer scalars are integers (operator.index), as in firwin
        for kind in (np.int64, np.int32):
            with self.subTest(kind=kind.__name__):
                n: Any = kind(21)
                assert_array_equal(firls(n, BANDS, DESIRED), firls(21, BANDS, DESIRED))
                assert_array_equal(remez(n, BANDS, [1.0, 0.0]), remez(21, BANDS, [1.0, 0.0]))

    def test_numpy_integer_fs(self) -> None:
        b4 = [0.0, 0.6, 1.0, 2.0]
        fs: Any = np.int64(4)
        assert_array_equal(firls(21, b4, DESIRED, fs=fs), firls(21, b4, DESIRED, fs=4.0))
        assert_array_equal(remez(21, b4, [1.0, 0.0], fs=fs), remez(21, b4, [1.0, 0.0], fs=4.0))

    def test_bools_still_rejected(self) -> None:
        values: tuple[Any, ...] = (True, np.True_)
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    firls(value, BANDS, DESIRED)
                with self.assertRaises(ValueError):
                    remez(value, BANDS, [1.0, 0.0])
                with self.assertRaises(ValueError):
                    firls(21, BANDS, DESIRED, fs=value)


class TestFirlsNonFinite(unittest.TestCase):
    def test_non_finite_inputs_rejected(self) -> None:
        # NaN or inf in bands, desired, weight or fs used to give NaN coefficients
        for bad in (np.nan, np.inf):
            cases = [
                ("bands", lambda b=bad: firls(21, [0.0, 0.3, b, 1.0], DESIRED)),
                ("desired", lambda b=bad: firls(21, BANDS, [1.0, b, 0.0, 0.0])),
                ("weight", lambda b=bad: firls(21, BANDS, DESIRED, weight=[1.0, b])),
                ("fs", lambda b=bad: firls(21, BANDS, DESIRED, fs=b)),
            ]
            for name, call in cases:
                with self.subTest(arg=name, value=bad), self.assertRaises(ValueError):
                    call()
