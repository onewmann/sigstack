"""Tests for the block contract: dtype and shape rules, complex opt-out, output shape.

The rules are documented in docs/conventions.md (D-016, D-017).
"""

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from sigstack.core import BaseBlock, Gain

FS = 1000.0

DTYPES = [np.bool_, np.int16, np.float32, np.float64, np.complex64, np.complex128]
SHAPES = [(0,), (1,), (17,), (1, 17), (3, 17), (2, 3, 17), (3, 0)]
# 0.1 and 1/3 + 0.2j are not exact in float32, so a gain rounded the wrong way fails.
GAINS = [0.1, 1 / 3 + 0.2j]


def _signal(dtype: type, shape: tuple[int, ...]) -> NDArray[Any]:
    """Seeded test signal of the given dtype and shape."""
    rng = np.random.default_rng(3)
    if dtype in (np.bool_, np.int16):
        return rng.integers(-5, 6, size=shape).astype(dtype)
    x = rng.standard_normal(shape)
    if np.issubdtype(dtype, np.complexfloating):
        x = x + 1j * rng.standard_normal(shape)
    return x.astype(dtype)


def _expected_dtype(in_dtype: type, g: complex) -> np.dtype[Any]:
    """Output dtype from the table in docs/conventions.md."""
    base = np.dtype(np.float64) if np.dtype(in_dtype).kind in "biu" else np.dtype(in_dtype)
    if isinstance(g, complex) and base.kind == "f":
        return np.dtype(np.complex64) if base == np.float32 else np.dtype(np.complex128)
    return base


class _RealOnly(BaseBlock):
    """Test block that opts out of complex input."""

    SUPPORTS_COMPLEX = False

    @property
    def latency(self) -> int:
        return 0

    def reset(self) -> None:
        pass

    def _process(self, x: NDArray[Any]) -> NDArray[Any]:
        out: NDArray[Any] = 2.0 * x
        return out


# --- dtype x shape matrix -----------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("g", GAINS)
def test_gain_contract_matrix(dtype: type, shape: tuple[int, ...], g: complex) -> None:
    x = _signal(dtype, shape)
    gain = Gain(g, fs=FS)
    y = gain.process(x)
    assert y.shape == x.shape == gain.output_shape(x.shape)
    assert y.dtype == _expected_dtype(dtype, g)
    # Reference: bool/int converted to float64, then the same single multiply in the
    # output precision and the same operand order, so the result is exact.
    xf = x.astype(np.float64) if x.dtype.kind in "biu" else x
    assert_allclose(y, xf * g, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(16,), (3, 16), (2, 3, 16)])
@pytest.mark.parametrize("g", [2.5, 0.1, -1.0 + 0.5j, 1 / 3 + 0.2j])
@pytest.mark.parametrize("complex_input", [False, True])
def test_gain_values_and_shapes(shape: tuple[int, ...], g: complex, complex_input: bool) -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(shape)
    if complex_input:
        x = x + 1j * rng.standard_normal(shape)
    y = Gain(g, fs=FS).process(x)
    assert y.shape == x.shape
    # Same single multiply in float64/complex128 and the same operand order as the block
    # (complex multiplication is not bitwise commutative), so the result is exact.
    assert_allclose(y, x * g, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("in_dtype", "g", "out_dtype"),
    [
        (np.int16, 2.0, np.float64),
        (np.int64, 2.0, np.float64),
        (np.uint8, 2.0, np.float64),
        (np.float32, 2.0, np.float32),
        (np.float64, 2.0, np.float64),
        (np.complex64, 2.0, np.complex64),
        (np.complex128, 2.0, np.complex128),
        (np.int16, 1j, np.complex128),
        (np.float32, 1j, np.complex64),
        (np.float64, 1j, np.complex128),
        (np.complex64, 1j, np.complex64),
        (np.complex128, 1j, np.complex128),
    ],
)
def test_gain_dtype_rules(in_dtype: type, g: complex, out_dtype: type) -> None:
    x: NDArray[Any] = np.arange(8).astype(in_dtype)
    y = Gain(g, fs=FS).process(x)
    assert y.dtype == out_dtype
    # Small integers times 2 or 1j are exact in every float format.
    assert_allclose(y, g * np.arange(8), rtol=0, atol=0)


# --- complex opt-out ----------------------------------------------------------------


def test_base_block_supports_complex_by_default() -> None:
    assert BaseBlock.SUPPORTS_COMPLEX is True
    assert Gain.SUPPORTS_COMPLEX is True


@pytest.mark.parametrize(
    "x", [np.ones(4, dtype=np.complex128), np.ones((2, 3), dtype=np.complex64), [1 + 0j, 2 + 0j]]
)
def test_real_only_block_rejects_complex_input(x: Any) -> None:
    # Also rejected when the imaginary part is zero: the dtype decides, not the values.
    with pytest.raises(TypeError, match="_RealOnly does not support complex input"):
        _RealOnly(fs=FS).process(x)


@pytest.mark.parametrize("dtype", [np.bool_, np.int16, np.float32, np.float64])
def test_real_only_block_accepts_real_input(dtype: type) -> None:
    y = _RealOnly(fs=FS).process(np.ones(4, dtype=dtype))
    assert_allclose(y, np.full(4, 2.0), rtol=0, atol=0)  # exact: 2 * 1


# --- output shape -------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_default_output_shape_is_the_input_shape(shape: tuple[int, ...]) -> None:
    assert Gain(1.0, fs=FS).output_shape(shape) == shape
    assert _RealOnly(fs=FS).output_shape(shape) == shape


def test_output_shape_accepts_numpy_integers() -> None:
    shape = Gain(1.0, fs=FS).output_shape((np.int64(3), np.int32(17)))  # type: ignore[arg-type]
    assert shape == (3, 17)
    assert all(type(n) is int for n in shape)


@pytest.mark.parametrize("shape", [(), (3, -1), (-2,)])
def test_invalid_output_shape_input_raises_value_error(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="input_shape must have at least one axis"):
        Gain(1.0, fs=FS).output_shape(shape)
