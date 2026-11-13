import unittest

import sigstack.signals as signals
from sigstack.signals.info import SignalInfo
from sigstack.signals.modulation import am, fm
from sigstack.signals.noise import pink_noise, white_noise
from sigstack.signals.tones import chirp, impulse, multitone


class TestPackage(unittest.TestCase):
    def test_exports(self) -> None:
        names = [
            "SignalInfo",
            "am",
            "chirp",
            "fm",
            "impulse",
            "multitone",
            "pink_noise",
            "white_noise",
        ]
        self.assertEqual(sorted(signals.__all__), names)
        objs = [SignalInfo, am, chirp, fm, impulse, multitone, pink_noise, white_noise]
        for name, obj in zip(names, objs, strict=True):
            self.assertIs(getattr(signals, name), obj)
