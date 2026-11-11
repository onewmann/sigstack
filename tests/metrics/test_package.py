import unittest

import sigstack.metrics as metrics
from sigstack.metrics.bands import passband_ripple, stopband_attenuation
from sigstack.metrics.response import freq_response, group_delay
from sigstack.metrics.snr import snr_db, snr_db_ref


class TestPackage(unittest.TestCase):
    def test_exports(self) -> None:
        names = [
            "freq_response",
            "group_delay",
            "passband_ripple",
            "snr_db",
            "snr_db_ref",
            "stopband_attenuation",
        ]
        self.assertEqual(sorted(metrics.__all__), names)
        objs = [
            freq_response,
            group_delay,
            passband_ripple,
            snr_db,
            snr_db_ref,
            stopband_attenuation,
        ]
        for name, obj in zip(names, objs, strict=True):
            self.assertIs(getattr(metrics, name), obj)
