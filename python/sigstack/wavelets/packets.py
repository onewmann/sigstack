"""Wavelet packet decomposition, reconstruction, cost functions and best basis.

This module implements the full wavelet packet tree (Coifman-Wickerhauser)
for orthogonal wavelets using only NumPy and the existing
:meth:`sigstack.wavelets.dwt` / :meth:`sigstack.wavelets.idwt` primitives.

Public API
----------
* :func:`wp_decompose` - full packet tree decomposition
* :func:`wp_reconstruct` - reconstruction from any valid basis
* :func:`wp_frequency_order` - frequency-ordered node paths at a level
* :func:`wp_frequency_band` - normalized or absolute frequency band of a path
* :func:`wp_cost` - additive cost functions (Shannon, log-energy, L1)
* :func:`best_basis` - Coifman-Wickerhauser best-basis search

References
----------
.. [Mertins2023] A. Mertins, *Signaltheorie: Grundlagen der Signalbeschreibung,
   Filterbänke, Wavelets, Zeit-Frequenz-Analyse, Parameter- und Signalschätzung*, 5th ed.,
   Springer Vieweg, 2023, sec. 7.3 (filter banks in tree structure, wavelet packets and the
   best-basis algorithm, p. 270) and sec. 9.11 (best basis, p. 377).
.. [Coifman1992] R. R. Coifman and M. V. Wickerhauser, "Entropy-based algorithms for best
   basis selection", *IEEE Trans. Inform. Theory*, vol. 38, no. 2, pp. 713-718, 1992.
"""

from __future__ import annotations

import numbers
import operator
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sigstack.wavelets.dwt import dwt, dwt_max_level, idwt
from sigstack.wavelets.filters import wavelet_filters

__all__ = [
    "best_basis",
    "wp_cost",
    "wp_decompose",
    "wp_frequency_band",
    "wp_frequency_order",
    "wp_reconstruct",
]

_VALID_MODES: tuple[str, ...] = ("symmetric", "zero", "periodization")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_float64_1d(x: ArrayLike) -> NDArray[np.float64]:
    """Convert *x* to a 1-D ``float64`` array, validating shape and values."""
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"Input must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("Input must be non-empty")
    if np.iscomplexobj(arr):
        raise ValueError("Input must be real-valued")
    arr = arr.astype(np.float64, copy=True)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input must contain only finite values")
    return arr


def _validate_mode(mode: str) -> None:
    """Raise ``ValueError`` if *mode* is not a recognised boundary mode."""
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {_VALID_MODES}")


def _validate_level(level: Any) -> int:
    """Validate that *level* is a non-negative integer (bool rejected)."""
    return _as_int("level", level, minimum=0)


def _as_int(name: str, value: object, *, minimum: int) -> int:
    """Return *value* as an ``int`` >= *minimum* (NumPy integers included, bool rejected)."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, got bool {value!r}")
    try:
        result = operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}") from None
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _node_length(
    n: int,
    depth: int,
    mode: str,
    filt_len: int,
) -> int:
    """Compute the expected coefficient length at *depth* for signal length *n*.

    Parameters
    ----------
    n : int
        Original signal length.
    depth : int
        Depth in the packet tree (0 = root).
    mode : str
        Boundary mode.
    filt_len : int
        Filter length.

    Returns
    -------
    int
        Expected coefficient array length at *depth*.
    """
    m = n
    for _ in range(depth):
        if mode == "periodization":
            m = (m + 1) // 2
        else:
            m = (m + filt_len - 1) // 2
    return m


def _sibling(path: str) -> str:
    """Return the sibling path of *path* (swap last 'a'/'d')."""
    if len(path) == 0:
        raise ValueError("Root has no sibling")
    last = "d" if path[-1] == "a" else "a"
    return path[:-1] + last


def _validate_path_chars(path: str) -> None:
    """Raise ``ValueError`` if *path* contains characters other than 'a'/'d'."""
    for ch in path:
        if ch not in ("a", "d"):
            raise ValueError(f"Invalid character {ch!r} in path {path!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def wp_decompose(
    x: ArrayLike,
    wavelet: str,
    *,
    level: int | None = None,
    mode: str = "periodization",
) -> dict[str, NDArray[np.float64]]:
    """Full wavelet packet tree decomposition.

    Parameters
    ----------
    x : array_like
        1-D input signal.
    wavelet : str
        Wavelet name.
    level : int or None, optional
        Decomposition depth.  If ``None``, the maximum admissible level is
        used (see :func:`dwt_max_level`).
    mode : str, optional
        Boundary handling: ``"periodization"`` (default), ``"symmetric"``,
        or ``"zero"``.

    Returns
    -------
    dict[str, NDArray[np.float64]]
        Mapping from node path to coefficient array.  The root is ``""``,
        children of path *p* are ``p + "a"`` and ``p + "d"``.  Keys are
        ordered breadth-first by level, then lexicographically within each
        level.

    Raises
    ------
    ValueError
        If *x* is invalid, *wavelet* unknown, *mode* invalid, or *level*
        negative or non-integer (bool rejected).
    """
    x_arr = _to_float64_1d(x)
    _validate_mode(mode)
    wf = wavelet_filters(wavelet)

    if level is None:
        resolved_level = dwt_max_level(len(x_arr), len(wf.dec_lo))
    else:
        resolved_level = _validate_level(level)

    nodes: dict[str, NDArray[np.float64]] = {"": x_arr}
    for _ in range(resolved_level):
        next_nodes: list[tuple[str, NDArray[np.float64]]] = []
        for p in sorted(nodes.keys(), key=lambda s: (len(s), s)):
            if len(p) == resolved_level:
                continue
            ca, cd = dwt(nodes[p], wavelet, mode=mode)
            next_nodes.append((p + "a", ca))
            next_nodes.append((p + "d", cd))
        for pth, arr in next_nodes:
            nodes[pth] = arr

    # Return ordered by level then lexicographic within level
    result: dict[str, NDArray[np.float64]] = {}
    for p in sorted(nodes.keys(), key=lambda s: (len(s), s)):
        result[p] = nodes[p]
    return result


def wp_frequency_order(level: int) -> list[str]:
    """Return node paths at *level* in frequency order.

    Parameters
    ----------
    level : int
        Depth in the packet tree (0 = root).  Must be a non-negative integer;
        bool is rejected.

    Returns
    -------
    list[str]
        Paths at *level* ordered by increasing centre frequency.

    Raises
    ------
    ValueError
        If *level* is negative or not an integer (bool rejected).
    """
    _validate_level(level)
    if level == 0:
        return [""]
    paths = [""]
    for _ in range(level):
        next_paths: list[str] = []
        for i, p in enumerate(paths):
            if i % 2 == 0:
                next_paths.append(p + "a")
                next_paths.append(p + "d")
            else:
                next_paths.append(p + "d")
                next_paths.append(p + "a")
        paths = next_paths
    return paths


def wp_frequency_band(
    path: str,
    fs: float = 2.0,
) -> tuple[float, float]:
    """Return the frequency band of a packet node.

    Parameters
    ----------
    path : str
        Node path consisting only of ``"a"`` and ``"d"`` characters.
    fs : float, optional
        Sampling frequency (default 2.0 for normalized frequency).

    Returns
    -------
    tuple[float, float]
        ``(f_low, f_high)`` in the same units as *fs*.

    Raises
    ------
    ValueError
        If *path* is invalid or *fs* is not a finite positive real number.
    """
    _validate_path_chars(path)
    fs_obj: object = fs  # checked as object: NumPy scalars are numbers.Real, str and bool are not
    if isinstance(fs_obj, (bool, np.bool_)) or not isinstance(fs_obj, numbers.Real):
        raise ValueError(f"fs must be a real number, got {type(fs_obj).__name__}")
    fs_val = float(fs_obj)
    if not np.isfinite(fs_val) or fs_val <= 0.0:
        raise ValueError(f"fs must be a finite positive number, got {fs}")

    order = wp_frequency_order(len(path))
    if path not in order:
        raise ValueError(f"Path {path!r} is not in frequency order at level {len(path)}")
    i = order.index(path)
    width = (fs_val / 2.0) / (2 ** len(path))
    return (i * width, (i + 1) * width)


def wp_reconstruct(
    nodes: Mapping[str, ArrayLike],
    wavelet: str,
    n: int,
    *,
    mode: str = "periodization",
) -> NDArray[np.float64]:
    """Reconstruct a signal from a set of packet-tree coefficients.

    Parameters
    ----------
    nodes : Mapping[str, array_like]
        Coefficient arrays for a valid basis of the packet tree.  Keys are
        node paths; the mapping must form a complete partition (no overlap,
        no gaps).
    wavelet : str
        Wavelet name.
    n : int
        Expected length of the reconstructed signal.
    mode : str, optional
        Boundary handling (default ``"periodization"``).

    Returns
    -------
    NDArray[np.float64]
        Reconstructed signal of shape ``(n,)``.

    Raises
    ------
    ValueError
        If *nodes* is empty, keys are invalid, the basis is not a complete
        partition, *n* is invalid, or coefficient lengths mismatch.
    """
    n = _as_int("n", n, minimum=1)

    if not nodes:
        raise ValueError("nodes must be a non-empty mapping")

    _validate_mode(mode)
    wf = wavelet_filters(wavelet)
    filt_len = len(wf.dec_lo)

    keys = list(nodes.keys())
    for k in keys:
        _validate_path_chars(k)

    # Validate complete partition
    if "" in keys and len(keys) > 1:
        raise ValueError("Root path '' cannot be combined with other paths")

    max_depth = max(len(k) for k in keys)
    total = sum(2 ** (max_depth - len(k)) for k in keys)
    if total != 2**max_depth:
        raise ValueError(
            f"Basis does not form a complete partition: sum of spans {total} != 2^{max_depth}"
        )

    # Check no prefix overlap: if path p is in keys, no child of p should be
    for k in keys:
        for j in range(1, len(k)):
            if k[:j] in keys:
                raise ValueError(f"Prefix overlap: {k[:j]!r} and {k!r}")

    # Recursive reconstruction
    def rec(p: str) -> NDArray[np.float64]:
        if p in nodes:
            arr = _to_float64_1d(nodes[p])
            expected_len = _node_length(n, len(p), mode, filt_len)
            if len(arr) != expected_len:
                raise ValueError(
                    f"Coefficient length mismatch for path {p!r}: "
                    f"got {len(arr)}, expected {expected_len}"
                )
            return arr
        # Internal node: reconstruct children and merge
        ca = rec(p + "a")
        cd = rec(p + "d")
        merged = idwt(ca, cd, wavelet, mode=mode)
        expected_len = _node_length(n, len(p), mode, filt_len)
        if len(merged) < expected_len:
            raise ValueError(
                f"Reconstructed length {len(merged)} for path {p!r} "
                f"is less than expected {expected_len}"
            )
        return merged[:expected_len]

    return rec("")  # trimmed to the node length of depth 0, which is n


def wp_cost(
    c: ArrayLike,
    cost: str | Callable[[NDArray[np.float64]], float] = "shannon",
    *,
    energy: float | None = None,
) -> float:
    """Compute a cost function for a coefficient array.

    Parameters
    ----------
    c : array_like
        1-D real finite coefficient array.
    cost : str or callable, optional
        Cost name (``"shannon"``, ``"log_energy"``, ``"l1"``) or a callable
        that takes an array and returns a float (default ``"shannon"``).
    energy : float, optional
        Total energy for Shannon cost.  If ``None``, computed as
        ``sum(c**2)``.  An explicit ``energy=0.0`` is honoured.

    Returns
    -------
    float
        The cost value.

    Raises
    ------
    ValueError
        If *c* is invalid, *cost* is an unknown string, or *energy* is not
        a finite non-negative number.
    """
    arr = _to_float64_1d(c)

    if callable(cost):
        return float(cost(arr))

    if cost == "shannon":
        if energy is None:
            total_energy = float(np.sum(arr * arr))
        else:
            given = float(energy)
            if not np.isfinite(given) or given < 0.0:
                raise ValueError(f"energy must be a finite non-negative number, got {energy}")
            total_energy = given
        if total_energy == 0.0:
            return 0.0
        q = arr * arr / total_energy
        # -sum(q*log(q)) for q > 0
        mask = q > 0.0
        return float(-np.sum(q[mask] * np.log(q[mask])))
    elif cost == "log_energy":
        mask = arr != 0.0
        if not np.any(mask):
            return 0.0
        return float(np.sum(np.log(arr[mask] * arr[mask])))
    elif cost == "l1":
        return float(np.sum(np.abs(arr)))
    else:
        raise ValueError(f"Unknown cost {cost!r}; expected 'shannon', 'log_energy', or 'l1'")


def best_basis(
    nodes: Mapping[str, ArrayLike],
    cost: str | Callable[[NDArray[np.float64]], float] = "shannon",
) -> list[str]:
    """Find the best basis in a wavelet packet tree (Coifman-Wickerhauser).

    Parameters
    ----------
    nodes : Mapping[str, array_like]
        Full packet-tree decomposition as returned by :func:`wp_decompose`.
        Must contain the root ``""`` and for every non-leaf node both children.
    cost : str or callable, optional
        Cost function (default ``"shannon"``).

    Returns
    -------
    list[str]
        Sorted list of paths forming the best basis.

    Raises
    ------
    ValueError
        If *nodes* is missing the root, or any non-root path's sibling is
        missing.
    """
    if "" not in nodes:
        raise ValueError("nodes must contain the root path ''")

    # Validate that for every non-leaf node, both children exist
    all_paths = list(nodes.keys())
    for p in all_paths:
        if len(p) == 0:
            continue
        s = _sibling(p)
        if s not in nodes:
            raise ValueError(f"Sibling {s!r} of path {p!r} is missing")

    # Compute total energy at root
    root_arr = _to_float64_1d(nodes[""])
    root_energy = float(np.sum(root_arr * root_arr))

    def compute_cost(p: str) -> float:
        arr = _to_float64_1d(nodes[p])
        if callable(cost):
            return float(cost(arr))
        elif cost == "shannon":
            return wp_cost(arr, "shannon", energy=root_energy)
        else:
            return wp_cost(arr, cost)

    # Bottom-up best basis search
    def solve(p: str) -> tuple[list[str], float]:
        if p + "a" not in nodes:  # no children: a leaf (the sibling check guarantees both)
            # Leaf
            return ([p], compute_cost(p))

        ba, ca = solve(p + "a")
        bd, cd = solve(p + "d")
        cp = compute_cost(p)

        if cp <= ca + cd:
            return ([p], cp)
        else:
            return (ba + bd, ca + cd)

    basis, _ = solve("")
    return sorted(basis)
