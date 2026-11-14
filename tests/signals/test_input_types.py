import copy
import dataclasses
import math
import pickle
import unittest
from typing import Any

import numpy as np
from numpy.testing import assert_array_equal

from sigstack.signals import SignalInfo, am, chirp, impulse, multitone


class TestNumpyScalars(unittest.TestCase):
    def test_numpy_integers_for_n(self) -> None:
        # NumPy integer scalars are integers (operator.index), as in firwin and the noise generators
        cases = [
            (lambda n: multitone(n, 16.0, [1.0])),
            (lambda n: chirp(n, 16.0, 1.0, 4.0)),
            (lambda n: impulse(n)),
        ]
        for i, make in enumerate(cases):
            for kind in (np.int64, np.int32, np.uint16):
                with self.subTest(case=i, kind=kind.__name__):
                    x, info = make(kind(64))
                    ref, _ = make(64)
                    assert_array_equal(x, ref)
                    self.assertEqual(type(info.n_samples), int)

    def test_bools_still_rejected(self) -> None:
        for value in (True, np.True_):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    multitone(value, 16.0, [1.0])  # type: ignore[arg-type]
                with self.assertRaises(TypeError):
                    chirp(value, 16.0, 1.0, 4.0)  # type: ignore[arg-type]
                with self.assertRaises(TypeError):
                    am(64, value, 1.0, 0.1)  # type: ignore[arg-type]  # fs

    def test_numpy_integer_fs(self) -> None:
        fs: Any = np.int64(16)
        x, _ = multitone(32, fs, [1.0])
        ref, _ = multitone(32, 16.0, [1.0])
        assert_array_equal(x, ref)


class TestSignalInfo(unittest.TestCase):
    def _info(self) -> SignalInfo:
        return SignalInfo("test", 8.0, 8, (1.0,), (1.0,), 0.5, params={"a": 1.0})

    def test_hashable_and_equal(self) -> None:
        a, b = self._info(), self._info()
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(len({a, b}), 1)

    def test_pickle_copy_asdict(self) -> None:
        # the record stays a plain dataclass: multiprocessing and Monte Carlo runs pickle it
        info = self._info()
        self.assertEqual(pickle.loads(pickle.dumps(info)), info)
        self.assertEqual(copy.deepcopy(info), info)
        self.assertEqual(dataclasses.asdict(info)["params"], {"a": 1.0})

    def test_caller_dict_is_copied(self) -> None:
        p = {"a": 1.0}
        info = SignalInfo("test", 8.0, 8, (), (), 0.0, params=p)
        p["a"] = 5.0
        self.assertEqual(info.params["a"], 1.0)

    def test_generators_still_report_params(self) -> None:
        _, info = am(64, 16.0, 2.0, 0.5, depth=0.5)
        self.assertTrue(math.isclose(info.params["depth"], 0.5))
        hash(info)
