import unittest

import sigstack.wavelets as wavelets
from sigstack.wavelets import packets
from sigstack.wavelets.dwt import dwt, dwt_max_level, idwt, wavedec, waverec
from sigstack.wavelets.filters import WAVELETS, WaveletFilters, wavelet_filters

NAMES = [
    "WAVELETS",
    "WaveletFilters",
    "dwt",
    "dwt_max_level",
    "idwt",
    "wavedec",
    "waverec",
    "wavelet_filters",
]
PACKETS = [
    "best_basis",
    "wp_cost",
    "wp_decompose",
    "wp_frequency_band",
    "wp_frequency_order",
    "wp_reconstruct",
]


class TestPackage(unittest.TestCase):
    def test_exports(self) -> None:
        self.assertEqual(sorted(wavelets.__all__), sorted(NAMES + PACKETS))
        self.assertEqual(sorted(packets.__all__), PACKETS)
        for name in PACKETS:
            self.assertIs(getattr(wavelets, name), getattr(packets, name))
        objs = [WAVELETS, WaveletFilters, dwt, dwt_max_level, idwt, wavedec, waverec]
        objs.append(wavelet_filters)
        for name, obj in zip(NAMES, objs, strict=True):
            self.assertIs(getattr(wavelets, name), obj)
