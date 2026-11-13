import dataclasses
import unittest

from sigstack.signals.info import SignalInfo


class TestSignalInfo(unittest.TestCase):
    def test_fields_and_frozen(self) -> None:
        # Frozen dataclass, fields in this order; params maps parameter names to floats
        self.assertEqual(
            [f.name for f in dataclasses.fields(SignalInfo)],
            ["kind", "fs", "n_samples", "freqs", "amplitudes", "power", "params"],
        )
        info = SignalInfo("multitone", 8.0, 8, (1.0,), (2.0,), 2.0, {})
        self.assertEqual(info.freqs, (1.0,))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            info.power = 3.0  # type: ignore[misc]

    def test_params_default_empty(self) -> None:
        info = SignalInfo("impulse", 1.0, 4, (), (), 0.25)
        self.assertEqual(info.params, {})
