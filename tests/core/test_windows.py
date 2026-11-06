"""Tests for the window functions; scipy.signal serves as the reference."""

from typing import Any

import numpy as np
import pytest
import scipy.signal
from numpy.testing import assert_allclose

import sigstack
import sigstack.core
from sigstack.core import windows
from sigstack.core.windows import coherent_gain, enbw, general_cosine, get_window

LENGTHS = [1, 2, 7, 64, 255]
SCIPY_NAMES = {
    "rectangular": "boxcar",
    "hann": "hann",
    "hamming": "hamming",
    "blackman": "blackman",
    "blackman_harris": "blackmanharris",
    "flattop": "flattop",
}
KAISER_BETAS = [0.0, 0.5, 8.6, 14.0]
# Every window with the parameters it needs, for tests that cover all of them.
ALL_WINDOWS: list[tuple[str, dict[str, Any]]] = [(name, {}) for name in SCIPY_NAMES] + [
    ("kaiser", {"beta": 8.6})
]


# --- agreement with scipy -----------------------------------------------------------


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("name", list(SCIPY_NAMES))
def test_window_matches_scipy(name: str, n: int, sym: bool) -> None:
    ours = get_window(name, n, sym=sym)
    ref = scipy.signal.get_window(SCIPY_NAMES[name], n, fftbins=not sym)
    assert ours.dtype == np.float64
    assert ours.shape == (n,)
    # Same closed-form definition in float64: only rounding differs (observed < 1e-14).
    assert_allclose(ours, ref, rtol=0, atol=1e-12)


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("beta", KAISER_BETAS)
def test_kaiser_matches_scipy(beta: float, n: int, sym: bool) -> None:
    ours = get_window("kaiser", n, sym=sym, beta=beta)
    ref = scipy.signal.get_window(("kaiser", beta), n, fftbins=not sym)
    # numpy.i0 and scipy.special.i0 both use Cephes Chebyshev series; float64 rounding only.
    assert_allclose(ours, ref, rtol=0, atol=1e-12)


# --- structure ----------------------------------------------------------------------


@pytest.mark.parametrize(("name", "params"), ALL_WINDOWS)
@pytest.mark.parametrize("sym", [True, False])
def test_named_function_equals_get_window(name: str, params: dict[str, Any], sym: bool) -> None:
    func = getattr(windows, name)
    # Exact: get_window only dispatches to the same function.
    assert_allclose(
        func(64, sym=sym, **params), get_window(name, 64, sym=sym, **params), rtol=0, atol=0
    )


@pytest.mark.parametrize(("name", "params"), ALL_WINDOWS)
@pytest.mark.parametrize("n", [2, 7, 64, 255])
def test_symmetric_window_is_exactly_symmetric(name: str, params: dict[str, Any], n: int) -> None:
    w = get_window(name, n, sym=True, **params)
    assert_allclose(w, w[::-1], rtol=0, atol=0)  # exact: the second half is mirrored


@pytest.mark.parametrize(("name", "params"), ALL_WINDOWS)
@pytest.mark.parametrize("n", [2, 7, 64, 255])
def test_periodic_window_is_dft_even(name: str, params: dict[str, Any], n: int) -> None:
    w = get_window(name, n, sym=False, **params)
    # Exact: the window is cut from the mirrored symmetric window of length n + 1.
    assert_allclose(w[1:], w[:0:-1], rtol=0, atol=0)  # w[k] == w[n - k]
    assert_allclose(w, get_window(name, n + 1, sym=True, **params)[:n], rtol=0, atol=0)


@pytest.mark.parametrize(("name", "params"), ALL_WINDOWS)
@pytest.mark.parametrize("sym", [True, False])
def test_length_one_window_is_one(name: str, params: dict[str, Any], sym: bool) -> None:
    # Exact: a window of length 1 is defined as [1.0].
    assert_allclose(get_window(name, 1, sym=sym, **params), [1.0], rtol=0, atol=0)


def test_rectangular_ignores_sym() -> None:
    # Exact: both are np.ones(7).
    assert_allclose(
        get_window("rectangular", 7, sym=True), get_window("rectangular", 7), rtol=0, atol=0
    )


@pytest.mark.parametrize("sym", [True, False])
def test_kaiser_with_beta_zero_is_rectangular(sym: bool) -> None:
    # Exact: I0(0) / I0(0) = 1 for every sample.
    assert_allclose(get_window("kaiser", 9, sym=sym, beta=0.0), np.ones(9), rtol=0, atol=0)


def test_numpy_integer_length_is_accepted() -> None:
    assert get_window("hann", np.int64(16)).shape == (16,)


# --- generalised cosine -------------------------------------------------------------


def test_general_cosine_with_one_coefficient_is_constant() -> None:
    assert_allclose(general_cosine(8, [1.0]), np.ones(8), rtol=0, atol=0)  # exact: a0 * cos(0)


@pytest.mark.parametrize("sym", [True, False])
def test_general_cosine_reproduces_hann(sym: bool) -> None:
    # Exact: hann calls general_cosine with the same coefficients.
    assert_allclose(
        general_cosine(33, [0.5, 0.5], sym=sym), get_window("hann", 33, sym=sym), rtol=0, atol=0
    )


@pytest.mark.parametrize("a", [[], [[0.5, 0.5]], [0.5, float("nan")], [float("inf")]])
def test_general_cosine_rejects_bad_coefficients(a: Any) -> None:
    with pytest.raises(ValueError, match="a must"):
        general_cosine(8, a)


@pytest.mark.parametrize("a", [[0.5 + 1j, 0.5], [True, False], ["0.5", "0.5"]])
def test_general_cosine_rejects_non_real_coefficients(a: Any) -> None:
    with pytest.raises(TypeError, match="a must contain integers or floats"):
        general_cosine(8, a)


# --- coherent gain and ENBW ---------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected_enbw", "expected_cg"),
    [
        ("rectangular", 1.0, 1.0),
        ("hann", 1.5, 0.5),
        ("hamming", 1.3628, 0.54),
        ("blackman", 1.7268, 0.42),
        ("blackman_harris", 2.0044, 0.35875),
        ("flattop", 3.7702, 0.21557895),
    ],
)
def test_enbw_and_coherent_gain(name: str, expected_enbw: float, expected_cg: float) -> None:
    w = get_window(name, 1024)
    # Periodic cosine window: ENBW = (a0^2 + sum(a_m^2) / 2) / a0^2, given here to four
    # decimals; Harris (1978) Table 1 lists 1.00, 1.50, 1.36, 1.73, 2.00.
    assert_allclose(enbw(w), expected_enbw, rtol=0, atol=1e-3)
    # The cosine terms sum to zero over a full period, so CG = a0 up to rounding.
    assert_allclose(coherent_gain(w), expected_cg, rtol=0, atol=1e-12)


def test_hann_enbw_is_exactly_one_and_a_half() -> None:
    # Orthogonality of the DFT basis makes the periodic Hann ENBW exactly 1.5 for n > 2.
    assert_allclose(enbw(get_window("hann", 64)), 1.5, rtol=0, atol=1e-12)


@pytest.mark.parametrize(("alpha", "expected"), [(2.0, 1.50), (3.0, 1.80)])
def test_kaiser_enbw_matches_harris_table(alpha: float, expected: float) -> None:
    w = get_window("kaiser", 4096, beta=np.pi * alpha)
    # Harris (1978) Table 1, Kaiser-Bessel with alpha = beta / pi, rounded to 0.01.
    assert_allclose(enbw(w), expected, rtol=0, atol=5e-3)


def test_helpers_accept_lists() -> None:
    assert_allclose(coherent_gain([1, 0]), 0.5, rtol=0, atol=0)  # exact: one division
    assert_allclose(enbw([1, 1, 1]), 1.0, rtol=0, atol=0)  # exact: 3 * 3 / 9


@pytest.mark.parametrize("helper", [coherent_gain, enbw])
@pytest.mark.parametrize("w", [[], [[1.0, 2.0]], np.float64(1.0)])
def test_helpers_reject_non_1d_or_empty(helper: Any, w: Any) -> None:
    with pytest.raises(ValueError, match="non-empty 1-D"):
        helper(w)


@pytest.mark.parametrize("helper", [coherent_gain, enbw])
@pytest.mark.parametrize("w", [[1 + 1j, 1 - 1j], [True, False], ["1", "2"]])
def test_helpers_reject_non_real_samples(helper: Any, w: Any) -> None:
    with pytest.raises(TypeError, match="w must contain integers or floats"):
        helper(w)


@pytest.mark.parametrize("helper", [coherent_gain, enbw])
@pytest.mark.parametrize("w", [[float("nan"), 1.0], [float("inf"), 1.0]])
def test_helpers_reject_non_finite_samples(helper: Any, w: Any) -> None:
    with pytest.raises(ValueError, match="w must contain only finite values"):
        helper(w)


def test_enbw_rejects_zero_sum() -> None:
    with pytest.raises(ValueError, match="sum to zero"):
        enbw([1.0, -1.0])


# --- invalid arguments --------------------------------------------------------------


@pytest.mark.parametrize(("name", "params"), ALL_WINDOWS)
@pytest.mark.parametrize("n", [0, -1])
def test_length_below_one_raises_value_error(name: str, params: dict[str, Any], n: int) -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        get_window(name, n, **params)


@pytest.mark.parametrize("n", [2.5, 8.0, "8", None, True])
def test_non_integer_length_raises_type_error(n: Any) -> None:
    with pytest.raises(TypeError, match="n must be an integer"):
        get_window("hann", n)


@pytest.mark.parametrize("name", ["hanning", "Hann", "boxcar", ""])
def test_unknown_name_raises_value_error(name: str) -> None:
    with pytest.raises(ValueError, match="unknown window"):
        get_window(name, 8)


def test_non_string_name_raises_type_error() -> None:
    with pytest.raises(TypeError, match="name must be a string"):
        get_window(("kaiser", 8.6), 8)  # type: ignore[arg-type]


@pytest.mark.parametrize("sym", [1, "False", None])
def test_non_bool_sym_raises_type_error(sym: Any) -> None:
    with pytest.raises(TypeError, match="sym must be a bool"):
        get_window("hann", 8, sym=sym)


def test_kaiser_without_beta_raises_type_error() -> None:
    with pytest.raises(TypeError, match="beta"):
        get_window("kaiser", 8)


def test_unknown_parameter_raises_type_error() -> None:
    with pytest.raises(TypeError, match="beta"):
        get_window("hann", 8, beta=1.0)


@pytest.mark.parametrize("beta", [-1.0, float("nan"), float("inf")])
def test_invalid_beta_value_raises_value_error(beta: float) -> None:
    with pytest.raises(ValueError, match="beta must be finite and >= 0"):
        get_window("kaiser", 8, beta=beta)


@pytest.mark.parametrize("beta", [709.79, 710.0, 1000.0, 1e6])
def test_beta_that_overflows_i0_raises_value_error(beta: float) -> None:
    # Above about 709.78, I0(beta) is inf in float64 and the window would be NaN and 0.
    with pytest.raises(ValueError, match="I0"):
        get_window("kaiser", 8, beta=beta)


# Both cases contain the centre sample: symmetric 65, and periodic 64 (cut from symmetric 65).
@pytest.mark.parametrize(("n", "sym"), [(65, True), (64, False)])
def test_large_valid_beta_gives_finite_window(n: int, sym: bool) -> None:
    w = get_window("kaiser", n, sym=sym, beta=709.0)
    assert np.all(np.isfinite(w))
    assert np.all(w > 0.0)
    assert_allclose(np.max(w), 1.0, rtol=0, atol=0)  # exact at the centre: I0(beta) / I0(beta)


@pytest.mark.parametrize("beta", ["8.6", True, None])
def test_invalid_beta_type_raises_type_error(beta: Any) -> None:
    with pytest.raises(TypeError, match="beta must be a real number"):
        get_window("kaiser", 8, beta=beta)


# --- exports ------------------------------------------------------------------------


@pytest.mark.parametrize("name", windows.__all__)
def test_core_exports_every_window_function(name: str) -> None:
    assert name in sigstack.core.__all__
    assert getattr(sigstack.core, name) is getattr(windows, name)


@pytest.mark.parametrize("name", ["get_window", "coherent_gain", "enbw"])
def test_top_level_exports(name: str) -> None:
    assert name in sigstack.__all__
    assert getattr(sigstack, name) is getattr(windows, name)
