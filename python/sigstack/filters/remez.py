"""Parks-McClellan (Remez exchange) FIR filter design.

References
----------
.. [OS2010] A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed.,
   Pearson, 2010, sec. 7.7 (optimum approximations of FIR filters), 7.7.3 (Parks-McClellan
   algorithm).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["remez"]


def remez(
    numtaps: int,
    bands: ArrayLike,
    desired: ArrayLike,
    weight: ArrayLike | None = None,
    fs: float = 2.0,
) -> NDArray[np.float64]:
    r"""Design a linear-phase FIR filter by the Parks-McClellan method.

    The Remez exchange algorithm minimises the weighted maximum deviation
    between the actual frequency response and the desired response over the
    specified bands:

    .. math::

        \\min_{\\mathbf{h}} \\max_{f \\in \\bigcup_i [f_{2i}, f_{2i+1}]}
            w_i \\, \\left| |H(f)| - d_i \\right|

    where $w_i$ is the weight and $d_i$ the desired magnitude in band $i$.
    For an odd-length filter (type I) the amplitude function is

    .. math::

        A(f) = \\sum_{j=0}^{M} a_j \\cos(\\pi j f),

    and for an even-length filter (type II) with $N = 2M$ it is
    $A(f) = Q(f) P(f)$ with $Q(f) = \\cos(\\pi f / 2)$ and

    .. math::

        P(f) = \\sum_{j=0}^{M-1} p_j \\cos(\\pi j f).

    Parameters
    ----------
    numtaps : int
        Number of filter taps (filter order + 1). Must be >= 3.
    bands : float or array_like
        Monotonically increasing sequence of band edges in Hz, with an even
        number of entries. Each pair defines a band.
    desired : float or array_like
        Desired magnitude for each band. Length must equal ``len(bands) // 2``.
    weight : float or array_like or None, optional
        Relative weight for each band. Defaults to ones. All values must be > 0.
    fs : float, optional
        Sampling frequency in Hz. Default is 2.0 (normalised).

    Returns
    -------
    numpy.ndarray
        Filter coefficients of shape ``(numtaps,)`` with dtype ``float64``.

    Raises
    ------
    ValueError
        If any input fails validation.
    """
    # --- numtaps ---
    if isinstance(numtaps, bool) or not isinstance(numtaps, int):
        raise ValueError("numtaps must be an integer")
    if numtaps < 3:
        raise ValueError("numtaps must be >= 3")

    # --- fs ---
    if not isinstance(fs, (int, float)) or isinstance(fs, bool):
        raise ValueError("fs must be a positive finite number")
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be a positive finite number")

    # --- bands ---
    try:
        bands_arr = np.asarray(bands, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        raise ValueError("bands cannot be made into a 1D float array") from None
    n_edges = len(bands_arr)
    if n_edges == 0 or n_edges % 2 != 0:
        raise ValueError("bands must have an even positive number of entries")

    # Normalise so that the Nyquist frequency corresponds to 1.
    edges = bands_arr / (fs / 2.0)
    if np.any(edges < 0) or np.any(edges > 1):
        raise ValueError("band edges must be within [0, fs/2]")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("band edges must be strictly increasing and non-overlapping")

    # --- desired ---
    n_bands = n_edges // 2
    try:
        desired_arr = np.asarray(desired, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        raise ValueError("desired cannot be made into a 1D float array") from None
    if desired_arr.ndim != 1 or len(desired_arr) != n_bands:
        raise ValueError(
            f"desired must have length {n_bands} (one per band), got {len(desired_arr)}"
        )

    # --- weight ---
    if weight is None:
        weight_arr = np.ones(n_bands, dtype=np.float64)
    else:
        try:
            weight_arr = np.asarray(weight, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            raise ValueError("weight cannot be made into a 1D float array") from None
        if weight_arr.ndim != 1 or len(weight_arr) != n_bands:
            raise ValueError(
                f"weight must have length {n_bands} (one per band), got {len(weight_arr)}"
            )
        if np.any(weight_arr <= 0):
            raise ValueError("all weights must be > 0")

    # --- even-length Nyquist constraint ---
    if numtaps % 2 == 0:
        if abs(edges[-1] - 1.0) < 1e-9 and desired_arr[-1] != 0.0:
            raise ValueError(
                "even-length filter cannot have a non-zero desired response at Nyquist"
            )

    return _remez_exchange(numtaps, edges, desired_arr, weight_arr)


def _build_grid(edges: NDArray[np.float64], r: int, exclude_unity: bool) -> NDArray[np.float64]:
    """Dense grid: ~16 points per coefficient per band, band edges included."""
    grid_density = 64
    per_band = grid_density * r
    parts = []
    for i in range(len(edges) // 2):
        lo, hi = edges[2 * i], edges[2 * i + 1]
        if hi - lo < 1e-12:
            parts.append(np.array([hi]))
            continue
        n = max(2, math.ceil(per_band * (hi - lo)))
        parts.append(np.linspace(lo, hi, n))
    g = np.concatenate(parts)
    # De-duplicate points shared by adjacent bands.
    g = g[np.concatenate(([True], np.diff(g) > 0))]
    if exclude_unity and g[-1] == 1.0:
        g = g[:-1]
    return g


def _barycentric_weights(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Barycentric weights b_k = 1 / prod_{i != k} (x_k - x_i)."""
    n = len(x)
    b = np.ones(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if j != i:
                b[i] *= x[i] - x[j]
    return 1.0 / b


def _barycentric_eval(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    b: NDArray[np.float64],
    fq: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate the barycentric Lagrange interpolant through (x, y) at fq.

    The interpolant is exact at the nodes.
    """
    xq = np.cos(np.pi * fq)
    d = xq[:, None] - x[None, :]
    out = np.empty_like(xq)
    # Node match tolerance: 1e-12 relative to the ~O(1) spacing between
    # consecutive grid points (the grid has ~16 * r points per band).
    tol = 1e-12 * max(1.0, len(x) / 16.0)
    mask = np.abs(d).min(axis=1) < tol
    if not np.all(mask):
        w = b / d[~mask]
        out[~mask] = np.sum(w * y, axis=1) / np.sum(w, axis=1)
    for i in np.flatnonzero(mask):
        j = int(np.flatnonzero(np.abs(d[i]) < tol)[0])
        out[i] = y[j]
    return out


def _find_extrema(err: NDArray[np.float64], edge_idx: NDArray[np.intp]) -> NDArray[np.intp]:
    """All local extrema of E on the grid, including the band edges."""
    n = len(err)
    idx = set(edge_idx.tolist())
    for i in range(1, n - 1):
        if (err[i] - err[i - 1]) * (err[i + 1] - err[i]) < 0:
            idx.add(i)
    return np.array(sorted(idx), dtype=np.intp)


def _select_extrema(err: NDArray[np.float64], idx: NDArray[np.intp], r: int) -> NDArray[np.intp]:
    """Keep r+1 extremal indices with alternating signs.

    The global maximum of |E| is kept. If two neighbours with the same sign
    are adjacent the smaller one is dropped; while too many remain the
    smallest end is dropped.
    """
    cands = [int(i) for i in idx.tolist()]
    vals = [float(err[i]) for i in cands]
    n = r + 1
    changed = True
    while changed and len(cands) > n:
        changed = False
        for i in range(len(cands) - 1):
            if vals[i] * vals[i + 1] > 0:
                drop = i if abs(vals[i]) <= abs(vals[i + 1]) else i + 1
                del cands[drop]
                del vals[drop]
                changed = True
                break
    while len(cands) > n:
        if abs(vals[0]) <= abs(vals[-1]):
            cands.pop(0)
            vals.pop(0)
        else:
            cands.pop()
            vals.pop()
    return np.array(sorted(cands), dtype=np.intp)


def _solve_extrema(
    xk: NDArray[np.float64],
    d_k: NDArray[np.float64],
    w_k: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve the Chebyshev alternation system for C_k and delta."""
    r = len(xk) - 1
    b = _barycentric_weights(xk)
    sign = (-1.0) ** np.arange(r + 1, dtype=np.float64)
    delta = np.sum(b * d_k) / np.sum(b * sign / w_k)
    return d_k - sign * delta / w_k


def _remez_exchange(
    numtaps: int,
    edges: NDArray[np.float64],
    desired: NDArray[np.float64],
    weight: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Run the Parks-McClellan Remez exchange and return the taps."""
    n_bands = len(desired)
    even = numtaps % 2 == 0
    m = numtaps // 2
    r = m if even else m + 1

    grid = _build_grid(edges, r, exclude_unity=even)
    n_grid = len(grid)
    f = grid
    x = np.cos(np.pi * f)

    # Desired response D and weights W sampled on the grid.
    des = np.empty(n_grid, dtype=np.float64)
    wgt = np.empty(n_grid, dtype=np.float64)
    for i in range(n_bands):
        lo, hi = edges[2 * i], edges[2 * i + 1]
        sel = (f >= lo - 1e-12) & (f <= hi + 1e-12)
        des[sel] = desired[i]
        wgt[sel] = weight[i]
    if even:
        # Type II: approximate P with D / Q and W * Q, excluding f = 1.
        q = np.cos(np.pi * f / 2.0)
        des = des / q
        wgt = wgt * q

    # Band edge positions on the grid (f = 1 is excluded for type II).
    edge_list: list[int] = []
    for e in edges:
        j = int(np.searchsorted(f, e, side="left"))
        if j < n_grid and abs(f[j] - e) < 1e-9:
            edge_list.append(j)
    edge_idx = np.array(sorted(set(edge_list)), dtype=np.intp)

    # Initial r+1 extremal frequencies equally spaced over the grid points.
    ext = np.unique(np.linspace(0, n_grid - 1, r + 1).round().astype(np.intp))

    prev_ext: NDArray[np.intp] | None = None
    delta = 0.0
    for _ in range(100):
        xk = x[ext]
        c_k = _solve_extrema(xk, des[ext], wgt[ext])
        delta = float(np.max(np.abs(c_k - des[ext]) * wgt[ext]))
        amp = _barycentric_eval(xk, c_k, _barycentric_weights(xk), f)
        err = wgt * (amp - des)
        new_ext = _select_extrema(err, _find_extrema(err, edge_idx), r)

        converged = np.max(np.abs(err)) <= abs(delta) * (1.0 + 1e-12) + 1e-15
        unchanged = prev_ext is not None and np.array_equal(prev_ext, new_ext)
        ext = new_ext
        if converged or unchanged:
            break
        prev_ext = ext

    # Final amplitude on the grid, then solve the cosine system.
    xk = x[ext]
    c_k = _solve_extrema(xk, des[ext], wgt[ext])
    amp = _barycentric_eval(xk, c_k, _barycentric_weights(xk), f)
    fq = np.arange(r, dtype=np.float64) / r
    yq = _barycentric_eval(xk, c_k, _barycentric_weights(xk), fq)
    jcol = np.arange(r, dtype=np.float64)[:, None]
    coeffs = np.linalg.solve(np.cos(np.pi * jcol * fq[None, :]), yq)

    h = np.zeros(numtaps, dtype=np.float64)
    if not even:
        # Type I: A(f) = sum a_j cos(pi j f); h[M] = a_0, h[M +- j] = a_j / 2.
        h[m] = coeffs[0]
        for k in range(1, r):
            h[m - k] = coeffs[k] / 2.0
            h[m + k] = coeffs[k] / 2.0
    else:
        # Type II: A(f) = sum_{j=1..M} b_j cos(pi (j - 1/2) f) with
        # b_1 = (p_0 + p_1 / 2) / 2 etc., and h[M - j] = h[M - 1 + j] = b_j / 2.
        p = coeffs
        for jj in range(1, m + 1):
            if jj == 1:
                bj = p[0] + p[1] / 2.0
            elif jj == m:
                bj = p[m - 1] / 2.0
            else:
                bj = (p[jj - 1] + p[jj]) / 2.0
            h[m - jj] = bj / 2.0
            h[m - 1 + jj] = bj / 2.0
    return h
