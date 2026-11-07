"""Block interface: the contract that every processing step follows.

A block turns an input signal into an output signal, either in one call or as a
stream of chunks. :class:`Block` defines the contract, :class:`BaseBlock` is a
small helper that validates sampling rates and input arrays, and :class:`Gain`
is the simplest block that implements it.

The block model is described in ``docs/ROADMAP.md``, section 2.2; the full
rules are in ``docs/conventions.md``.

Dtype and shape rules
---------------------
Unless a block documents otherwise:

- Boolean and integer input is converted to ``float64`` before processing
  (``True`` becomes 1.0). String, object and date/time arrays are rejected.
  Scalar parameters such as ``fs`` stay strict: ``True`` is not a number.
- The output has the precision of the input: ``float32`` stays ``float32`` (or
  becomes ``complex64`` if the block produces complex output), ``float64``
  stays ``float64`` (or ``complex128``). Parameters are rounded to that
  precision. A block may compute internally at a higher precision, but it
  returns the input precision (D-017).
- Complex input gives complex output. A block that cannot handle it sets
  ``SUPPORTS_COMPLEX = False`` and rejects complex input with ``TypeError``.
- The last axis is time. Leading axes are channels or batch dimensions and are
  unchanged by default; :meth:`BaseBlock.output_shape` reports the shape a
  block returns.
"""

from __future__ import annotations

import cmath
import math
import numbers
import operator
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["BaseBlock", "Block", "Gain"]


@runtime_checkable
class Block(Protocol):
    r"""Interface of every processing block.

    A block maps an input signal sampled at :attr:`fs_in` to an output signal
    sampled at :attr:`fs_out`. It processes a whole recording in one call or a
    stream of consecutive chunks.

    Notes
    -----
    Every block follows this contract:

    - **Array layout.** Samples are on the last axis: the input has shape
      ``(..., n_samples)``. Leading axes are channels or batch dimensions; by
      default they are processed independently and appear unchanged in the
      output. A multichannel signal has shape ``(n_channels, n_samples)``. A
      block that adds, removes or mixes axes before the last one documents its
      output shape.
    - **State.** :meth:`process` keeps its internal state between calls, so
      consecutive calls receive consecutive pieces of one signal.
    - **Chunk invariance.** For any split of a signal :math:`x` along the last
      axis into :math:`x_1` and :math:`x_2` (either may be empty), a freshly
      constructed or reset block satisfies

      .. math::

         \operatorname{process}(x_1) \,\|\, \operatorname{process}(x_2)
         = \operatorname{process}(x_1 \,\|\, x_2),

      where :math:`\|` is concatenation along the last axis.
    - **Reset.** :meth:`reset` restores the state the block had right after
      construction.
    - **Dtypes.** Real and complex input are supported unless the block
      documents otherwise; see the dtype rules in :mod:`sigstack.core.block`
      and ``docs/conventions.md``.

    ``isinstance(obj, Block)`` only checks that the members exist, not their
    signatures or the contract.

    References
    ----------
    ``docs/ROADMAP.md``, section 2.2 (block model).
    """

    @property
    def fs_in(self) -> float:
        """Sampling rate of the input in Hz."""

    @property
    def fs_out(self) -> float:
        """Sampling rate of the output in Hz; differs from ``fs_in`` for rate changers."""

    @property
    def latency(self) -> int:
        """Delay the block adds, in output samples (0 for memoryless blocks)."""

    def reset(self) -> None:
        """Restore the state the block had right after construction."""

    def process(self, x: NDArray[Any]) -> NDArray[Any]:
        """Process the next chunk ``x`` of shape ``(..., n_samples)`` and return the output."""


def _check_rate(value: object, name: str) -> float:
    """Return ``value`` as a float after checking that it is a finite rate > 0."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return rate


def _as_signal(x: ArrayLike) -> NDArray[Any]:
    """Return ``x`` as an array with ``ndim >= 1`` and a floating or complex dtype.

    Boolean and integer arrays are converted to ``float64``; floating and complex
    arrays are returned without a copy.
    """
    arr = np.asarray(x)
    if arr.ndim < 1:
        raise ValueError(
            "input must have at least one dimension (samples on the last axis), "
            f"got a {arr.ndim}-d array"
        )
    kind = arr.dtype.kind
    if kind in "biu":
        return arr.astype(np.float64)
    if kind in "fc":
        return arr
    raise TypeError(
        f"input must be a boolean, integer, floating or complex array, got dtype {arr.dtype}"
    )


class BaseBlock(ABC):
    """Helper base class for blocks: validated sampling rates and input arrays.

    Subclasses implement :attr:`latency`, :meth:`reset` and :meth:`_process`.
    :meth:`process` validates the input and then calls :meth:`_process`, so
    ``_process`` always receives an array with ``ndim >= 1`` and a floating or
    complex dtype (boolean and integer input already converted to ``float64``,
    complex input already rejected if the class sets ``SUPPORTS_COMPLEX =
    False``). It must not modify that array in place, because it may be the
    caller's array.

    A subclass that changes the shape (adds, removes or mixes leading axes, or
    changes the number of samples) overrides :meth:`output_shape`.

    Attributes
    ----------
    SUPPORTS_COMPLEX : bool
        Class attribute, ``True`` by default. Set it to ``False`` in a subclass
        that cannot process complex input.

    Parameters
    ----------
    fs : float
        Input sampling rate in Hz. Must be finite and > 0.
    fs_out : float, optional
        Output sampling rate in Hz. Must be finite and > 0. Defaults to ``fs``.

    Raises
    ------
    TypeError
        If a rate is not a real number.
    ValueError
        If a rate is not finite or not > 0.
    """

    SUPPORTS_COMPLEX: ClassVar[bool] = True

    def __init__(self, *, fs: float, fs_out: float | None = None) -> None:
        self._fs_in = _check_rate(fs, "fs")
        self._fs_out = self._fs_in if fs_out is None else _check_rate(fs_out, "fs_out")

    @property
    def fs_in(self) -> float:
        """Sampling rate of the input in Hz."""
        return self._fs_in

    @property
    def fs_out(self) -> float:
        """Sampling rate of the output in Hz."""
        return self._fs_out

    @property
    @abstractmethod
    def latency(self) -> int:
        """Delay the block adds, in output samples."""

    @abstractmethod
    def reset(self) -> None:
        """Restore the state the block had right after construction."""

    def process(self, x: ArrayLike) -> NDArray[Any]:
        """Validate the next chunk and process it.

        Parameters
        ----------
        x : array_like
            Input of shape ``(..., n_samples)`` with a boolean, integer, floating
            or complex dtype. ``n_samples`` may be 0.

        Returns
        -------
        numpy.ndarray
            Output of shape ``(..., n_out)``.

        Raises
        ------
        ValueError
            If ``x`` is a scalar (0-d).
        TypeError
            If ``x`` does not have a boolean, integer, floating or complex dtype
            (for example string, object or date/time), or if ``x`` is complex
            and the class sets ``SUPPORTS_COMPLEX = False`` (also when the
            imaginary part is zero).
        """
        arr = _as_signal(x)
        if arr.dtype.kind == "c" and not self.SUPPORTS_COMPLEX:
            raise TypeError(
                f"{type(self).__name__} does not support complex input, got dtype {arr.dtype}"
            )
        return self._process(arr)

    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Return the shape that :meth:`process` returns for input of ``input_shape``.

        The default is the input shape. A block that adds, removes or mixes
        leading axes, or changes the number of samples, overrides it, so that a
        pipeline can check a chain of blocks before it runs.

        Parameters
        ----------
        input_shape : tuple of int
            Shape ``(..., n_samples)`` of the input: at least one axis, no
            negative lengths.

        Returns
        -------
        tuple of int
            Shape of the output.

        Raises
        ------
        ValueError
            If ``input_shape`` is empty or has a negative length.
        """
        shape = tuple(operator.index(n) for n in input_shape)
        if not shape or any(n < 0 for n in shape):
            raise ValueError(
                f"input_shape must have at least one axis and no negative length, got {shape}"
            )
        return shape

    @abstractmethod
    def _process(self, x: NDArray[Any]) -> NDArray[Any]:
        """Process a validated chunk; see the class docstring for the guarantees."""


class Gain(BaseBlock):
    r"""Multiply the signal by a constant factor.

    .. math::

       y[n] = g \, x[n]

    The block is memoryless: no state, latency 0, ``fs_out == fs_in``.

    Parameters
    ----------
    gain : float or complex
        Factor :math:`g`. Must be finite. A complex gain makes the output complex
        even for real input.
    fs : float
        Sampling rate in Hz. Must be finite and > 0.

    Raises
    ------
    TypeError
        If ``gain`` is not a number or ``fs`` is not a real number.
    ValueError
        If ``gain`` is not finite, or ``fs`` is not finite or not > 0.

    Notes
    -----
    The product is computed in the precision of the input:

    ==================  ==============  ==============
    input dtype         real gain       complex gain
    ==================  ==============  ==============
    boolean, integer    ``float64``     ``complex128``
    ``float32``         ``float32``     ``complex64``
    ``float64``         ``float64``     ``complex128``
    ``complex64``       ``complex64``   ``complex64``
    ``complex128``      ``complex128``  ``complex128``
    ==================  ==============  ==============

    The gain is rounded to the input precision as well, so a gain outside the
    range of that dtype (e.g. above about ``3.4e38`` for ``float32``) overflows
    to inf with a ``RuntimeWarning``.

    A gain is the simplest memoryless, linear, time-invariant system
    (Oppenheim and Schafer, *Discrete-Time Signal Processing*, 3rd ed., sec. 2.2).

    Examples
    --------
    >>> import numpy as np
    >>> from sigstack import Gain
    >>> Gain(0.5, fs=1000.0).process(np.array([2.0, -4.0]))
    array([ 1., -2.])
    """

    def __init__(self, gain: float | complex, *, fs: float) -> None:
        super().__init__(fs=fs)
        self._gain = _check_gain(gain)

    @property
    def gain(self) -> float | complex:
        """The factor :math:`g`."""
        return self._gain

    @property
    def latency(self) -> int:
        """Always 0: the output does not lag the input."""
        return 0

    def reset(self) -> None:
        """Do nothing: the block has no state."""

    def _process(self, x: NDArray[Any]) -> NDArray[Any]:
        # The gain is a Python float or complex, so NumPy keeps the precision of x
        # (NEP 50) and the multiplication returns a new array.
        out: NDArray[Any] = x * self._gain
        return out

    def __repr__(self) -> str:
        """Return ``Gain(gain=..., fs=...)``."""
        return f"Gain(gain={self._gain!r}, fs={self._fs_in!r})"


def _check_gain(gain: object) -> float | complex:
    """Return ``gain`` as a Python float or complex after checking that it is finite."""
    value: float | complex
    if isinstance(gain, bool) or not isinstance(gain, numbers.Complex):
        raise TypeError(f"gain must be a real or complex number, got {type(gain).__name__}")
    value = float(gain) if isinstance(gain, numbers.Real) else complex(gain)
    if not cmath.isfinite(value):
        raise ValueError(f"gain must be finite, got {gain!r}")
    return value
