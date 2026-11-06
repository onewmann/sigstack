"""Tests for the block interface, BaseBlock and the Gain block."""

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray

import sigstack
import sigstack.core
from sigstack.core import BaseBlock, Block, Gain

FS = 1000.0


class _Recorder(BaseBlock):
    """Minimal BaseBlock subclass that records the dtype `_process` receives."""

    def __init__(self, *, fs: float, fs_out: float | None = None) -> None:
        super().__init__(fs=fs, fs_out=fs_out)
        self.seen: list[np.dtype[Any]] = []

    @property
    def latency(self) -> int:
        return 0

    def reset(self) -> None:
        self.seen.clear()

    def _process(self, x: NDArray[Any]) -> NDArray[Any]:
        self.seen.append(x.dtype)
        return x


# --- protocol and exports ---------------------------------------------------------


def test_gain_satisfies_block_protocol() -> None:
    gain = Gain(2.0, fs=FS)
    assert isinstance(gain, Block)
    block: Block = gain  # checked statically by mypy
    assert block.latency == 0


def test_base_block_subclass_satisfies_block_protocol() -> None:
    assert isinstance(_Recorder(fs=FS), Block)


@pytest.mark.parametrize("obj", [object(), np.ones(3), 2.0])
def test_other_objects_are_not_blocks(obj: object) -> None:
    assert not isinstance(obj, Block)


@pytest.mark.parametrize("name", ["Block", "BaseBlock", "Gain"])
def test_public_names_are_exported(name: str) -> None:
    assert name in sigstack.__all__
    assert name in sigstack.core.__all__
    assert getattr(sigstack, name) is getattr(sigstack.core, name)


def test_base_block_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BaseBlock(fs=FS)  # type: ignore[abstract]


# --- BaseBlock ----------------------------------------------------------------------


def test_base_block_rates() -> None:
    same = _Recorder(fs=48_000)
    assert same.fs_in == same.fs_out == 48_000.0
    assert isinstance(same.fs_in, float)
    changed = _Recorder(fs=48_000, fs_out=16_000)
    assert (changed.fs_in, changed.fs_out) == (48_000.0, 16_000.0)


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (np.arange(4, dtype=np.int16), np.float64),
        (np.arange(4, dtype=np.uint8), np.float64),
        ([1, 2, 3], np.float64),
        (np.ones(4, dtype=np.float32), np.float32),
        (np.ones(4, dtype=np.complex64), np.complex64),
    ],
)
def test_base_block_passes_validated_dtype_to_process(x: Any, expected: type) -> None:
    block = _Recorder(fs=FS)
    block.process(x)
    assert block.seen == [np.dtype(expected)]


def test_base_block_does_not_copy_float_input() -> None:
    x = np.ones((2, 8))
    assert _Recorder(fs=FS).process(x) is x


@pytest.mark.parametrize("fs", [0.0, -1.0, float("nan"), float("inf"), -np.inf])
def test_invalid_rate_value_raises_value_error(fs: Any) -> None:
    with pytest.raises(ValueError, match="fs must be finite and > 0"):
        _Recorder(fs=fs)
    with pytest.raises(ValueError, match="fs_out must be finite and > 0"):
        _Recorder(fs=FS, fs_out=fs)


@pytest.mark.parametrize("fs", ["1000", None, True, np.bool_(True), 1000j])
def test_invalid_rate_type_raises_type_error(fs: Any) -> None:
    with pytest.raises(TypeError, match="fs must be a real number"):
        _Recorder(fs=fs)


# --- Gain: values, shapes, dtypes ---------------------------------------------------


def test_gain_properties() -> None:
    gain = Gain(-0.5, fs=48_000)
    assert gain.gain == -0.5
    assert Gain(0.1, fs=FS).gain == 0.1  # stored in double precision, not rounded
    assert gain.fs_in == gain.fs_out == 48_000.0
    assert gain.latency == 0
    assert repr(gain) == "Gain(gain=-0.5, fs=48000.0)"


@pytest.mark.parametrize("shape", [(16,), (3, 16), (2, 3, 16)])
# 0.1 and 1/3 + 0.2j are not exact in float32, so a gain rounded to single precision fails.
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


@pytest.mark.parametrize(
    ("g", "out_dtype"),
    [(np.float64(2.0), np.float32), (np.complex128(1j), np.complex64), (3, np.float32)],
)
def test_numpy_and_int_gains_keep_input_precision(g: Any, out_dtype: type) -> None:
    y = Gain(g, fs=FS).process(np.ones(4, dtype=np.float32))
    assert y.dtype == out_dtype


def test_gain_accepts_lists() -> None:
    y = Gain(2.0, fs=FS).process([1, 2, 3])
    assert y.dtype == np.float64
    assert_allclose(y, [2.0, 4.0, 6.0], rtol=0, atol=0)  # exact: small integers


def test_gain_on_empty_chunk() -> None:
    y = Gain(1j, fs=FS).process(np.empty((3, 0)))
    assert y.shape == (3, 0)
    assert y.dtype == np.complex128


def test_gain_does_not_modify_or_alias_input() -> None:
    x = np.arange(5.0)
    y = Gain(1.0, fs=FS).process(x)
    assert not np.shares_memory(x, y)
    assert_array_equal(x, np.arange(5.0))


# --- Gain: streaming ----------------------------------------------------------------


@pytest.mark.parametrize("shape", [(64,), (3, 64)])
def test_gain_chunked_equals_one_shot(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    gain = Gain(0.3 - 2j, fs=FS)
    one_shot = gain.process(x)
    n = shape[-1]
    for _ in range(50):
        # Repeated cut points give empty chunks, which the contract allows.
        cuts = np.sort(rng.integers(0, n + 1, size=rng.integers(0, 6)))
        gain.reset()
        chunks = np.split(x, cuts, axis=-1)
        y = np.concatenate([gain.process(c) for c in chunks], axis=-1)
        # Identical elementwise arithmetic in both paths.
        assert_allclose(y, one_shot, rtol=0, atol=0)


def test_gain_reset_keeps_behaviour() -> None:
    gain = Gain(3.0, fs=FS)
    x = np.linspace(-1.0, 1.0, 9)
    before = gain.process(x)
    gain.reset()
    assert_allclose(gain.process(x), before, rtol=0, atol=0)  # stateless: same arithmetic


# --- Gain: invalid arguments --------------------------------------------------------


@pytest.mark.parametrize("x", [np.float64(1.0), 3.0, np.array(2.0)])
def test_scalar_input_raises_value_error(x: Any) -> None:
    with pytest.raises(ValueError, match="at least one dimension"):
        Gain(1.0, fs=FS).process(x)


@pytest.mark.parametrize(
    "x",
    [
        np.array([True, False]),
        np.array(["a", "b"]),
        np.array([object(), object()]),
        np.array(["2026-11-06"], dtype="datetime64[D]"),
    ],
)
def test_non_numeric_input_raises_type_error(x: Any) -> None:
    with pytest.raises(TypeError, match="integer, floating or complex array"):
        Gain(1.0, fs=FS).process(x)


@pytest.mark.parametrize("g", [float("nan"), float("inf"), complex(0.0, float("inf"))])
def test_non_finite_gain_raises_value_error(g: Any) -> None:
    with pytest.raises(ValueError, match="gain must be finite"):
        Gain(g, fs=FS)


@pytest.mark.parametrize("g", ["2", None, True, [1.0]])
def test_non_numeric_gain_raises_type_error(g: Any) -> None:
    with pytest.raises(TypeError, match="gain must be a real or complex number"):
        Gain(g, fs=FS)


def test_gain_needs_a_valid_rate() -> None:
    with pytest.raises(ValueError, match="fs must be finite and > 0"):
        Gain(1.0, fs=0.0)
