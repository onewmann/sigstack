r"""Orthogonal wavelet filters (Haar, Daubechies, Symlets) by spectral factorisation.

The reconstruction low-pass filter of order :math:`N` has the zero :math:`z = -1` with
multiplicity :math:`N` plus one zero of each pair :math:`z, 1/z` that solves
:math:`z^2 - (2 - 4y) z + 1 = 0` for a root :math:`y` of

.. math::

    P(y) = \sum_{k=0}^{N-1} \binom{N-1+k}{k}\, y^k,

scaled so that :math:`\sum_n h[n] = \sqrt{2}`. Daubechies filters take every zero inside
the unit circle (minimum phase) [Mertins2023]_; Symlets take the zero selection listed in
``_SYM_BITS``, which reproduces the least-asymmetric coefficient tables of PyWavelets.
The high-pass and decomposition filters follow from the paraunitary two-channel filter
bank relations [Mertins2023]_.

References
----------
.. [Mertins2023] A. Mertins, *Signaltheorie: Grundlagen der Signalbeschreibung,
   Filterbänke, Wavelets, Zeit-Frequenz-Analyse, Parameter- und Signalschätzung*, 5th ed.,
   Springer Vieweg, 2023,
   sec. 9.7.2, eqs. (9.164) and (9.171) (Daubechies wavelets by spectral factorisation),
   sec. 7.2.5, eq. (7.70) (paraunitary two-channel filter bank), sec. 9.6.2 (moments).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

__all__ = ["WAVELETS", "WaveletFilters", "wavelet_filters"]

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

WAVELETS: tuple[str, ...] = (
    "haar",
    *(f"db{n}" for n in range(1, 11)),
    *(f"sym{n}" for n in range(2, 11)),
)

# Selection bits for symlet wavelets.
# Index i corresponds to the i-th chosen zero (sorted by ascending real part).
# 1 -> take z_in (|z| < 1), 0 -> take 1/z_in.
_SYM_BITS: dict[int, tuple[int, ...]] = {
    2: (1,),
    3: (1,),
    4: (1, 0),
    5: (0, 1),
    6: (0, 1, 0),
    7: (0, 1, 1),
    8: (1, 0, 1, 0),
    9: (1, 0, 0, 1),
    10: (0, 1, 0, 1, 0),
}


# ---------------------------------------------------------------------------
# Named tuple for the four filter arrays
# ---------------------------------------------------------------------------


class WaveletFilters(NamedTuple):
    """Four filter coefficient arrays for an orthogonal wavelet.

    Attributes
    ----------
    dec_lo : NDArray[np.float64]
        Decomposition low-pass filter (reversed reconstruction low-pass).
    dec_hi : NDArray[np.float64]
        Decomposition high-pass filter (reversed reconstruction high-pass).
    rec_lo : NDArray[np.float64]
        Reconstruction low-pass filter.
    rec_hi : NDArray[np.float64]
        Reconstruction high-pass filter.
    """

    dec_lo: NDArray[np.float64]
    dec_hi: NDArray[np.float64]
    rec_lo: NDArray[np.float64]
    rec_hi: NDArray[np.float64]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _binom(n: int, k: int) -> float:
    """Compute the binomial coefficient C(n, k) as a float."""
    if k < 0 or k > n:
        return 0.0
    # Use math.comb for exact integer arithmetic, then convert.
    return float(math.comb(n, k))


def _poly_p(order: int) -> NDArray[np.float64]:
    r"""Return coefficients of $P(y)$ highest-power first.

    :math:`P(y) = \\sum_{k=0}^{N-1} \\binom{N-1+k}{k}\\, y^k`

    Parameters
    ----------
    order : int
        Wavelet order (length of filter is 2*order).

    Returns
    -------
    NDArray[np.float64]
        Coefficients from highest power to constant term.
    """
    # P(y) = sum_{k=0}^{N-1} C(N-1+k, k) y^k
    # Highest power is y^(N-1), so coefficients array has length N.
    coeffs = np.empty(order, dtype=np.float64)
    for k in range(order):
        coeffs[order - 1 - k] = _binom(order - 1 + k, k)
    return coeffs


def _choose_zeros(
    y_roots: NDArray[np.complex128],
    bits: tuple[int, ...],
) -> list[complex]:
    r"""Choose the zeros inside/outside the unit circle for each root of P.

    For each representative root $y$ (real roots and one from each
    conjugate pair), solve $z^2 - (2 - 4y)\, z + 1 = 0$ and select
    either $z_{\text{in}}$ (|z| < 1) or $1/z_{\text{in}}$ according
    to the selection bit.

    Parameters
    ----------
    y_roots : NDArray[np.complex128]
        Representative roots of P(y), sorted by ascending real part.
    bits : tuple[int, ...]
        Selection bits; 1 -> z_in, 0 -> 1/z_in.

    Returns
    -------
    list[complex]
        Chosen zeros (including conjugates for complex y).
    """
    chosen: list[complex] = []
    for i, y in enumerate(y_roots):
        # Solve z^2 - (2 - 4y) z + 1 = 0
        a = 1.0 + 0j
        b = -(2.0 - 4.0 * y)
        c = 1.0 + 0j
        disc = b * b - 4.0 * a * c
        sqrt_disc = np.sqrt(disc + 0j)
        z1 = (-b + sqrt_disc) / (2.0 * a)
        z2 = (-b - sqrt_disc) / (2.0 * a)

        # Determine which is inside the unit circle
        if abs(z1) < abs(z2):
            z_in, z_out = z1, z2
        else:
            z_in, z_out = z2, z1

        bit = bits[i]
        if bit == 1:
            chosen_zero = z_in
        else:
            chosen_zero = z_out  # which is 1/z_in

        # If y is complex (non-zero imaginary part), add conjugate
        if abs(y.imag) > 1e-14:
            chosen.append(chosen_zero)
            chosen.append(np.conj(chosen_zero))
        else:
            chosen.append(complex(chosen_zero.real, 0.0))

    return chosen


def _compute_rec_lo(order: int, bits: tuple[int, ...]) -> NDArray[np.float64]:
    r"""Compute the reconstruction low-pass filter for order N.

    Parameters
    ----------
    order : int
        Wavelet order; filter length is 2*order.
    bits : tuple[int, ...]
        Selection bits for zero choice.

    Returns
    -------
    NDArray[np.float64]
        Low-pass filter coefficients scaled so sum = sqrt(2).
    """
    # Zeros: -1 with multiplicity N, plus chosen zeros from P(y) roots
    zeros: list[complex] = [-1.0 + 0j] * order

    if order > 1:
        p_coeffs = _poly_p(order)
        y_roots_all = np.roots(p_coeffs)

        # Choose representatives: all real roots, and for each complex
        # conjugate pair the one with positive imaginary part.
        reps: list[complex] = []
        for y in y_roots_all:
            if abs(y.imag) <= 1e-12:
                reps.append(complex(y.real, 0.0))
            elif y.imag > 1e-12:
                reps.append(complex(y))

        # Sort representatives by ascending real part
        reps.sort(key=lambda z: (z.real, z.imag))
        y_roots_arr = np.array(reps, dtype=np.complex128)

        chosen = _choose_zeros(y_roots_arr, bits)
        zeros.extend(chosen)

    # Build polynomial from zeros
    poly_coeffs = np.poly(zeros)  # highest power first
    h = np.real(poly_coeffs).astype(np.float64)

    # Scale so sum(h) = sqrt(2)
    s = float(h.sum())
    if abs(s) < 1e-15:
        raise RuntimeError("Filter sum is zero; cannot normalise.")
    h = h / s * math.sqrt(2.0)

    return h


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def wavelet_filters(name: str) -> WaveletFilters:
    r"""Return the four filter arrays for the named orthogonal wavelet.

    Parameters
    ----------
    name : str
        Wavelet name.  Must be one of ``WAVELETS``: ``"haar"``,
        ``"db1"``-``"db10"``, or ``"sym2"``-``"sym10"``.

    Returns
    -------
    WaveletFilters
        Named tuple with fields ``dec_lo``, ``dec_hi``, ``rec_lo``,
        ``rec_hi``, each a fresh 1-D array of dtype ``float64``.

    Raises
    ------
    ValueError
        If *name* is not in :data:`WAVELETS`.

    Notes
    -----
    The reconstruction low-pass filter $h$ satisfies the orthonormality
    conditions:

    .. math::

        \sum_n h[n] = \sqrt{2}, \qquad
        \sum_n h[n]\, h[n + 2k] = \delta_{k0}

    The high-pass filter is $g[n] = (-1)^n\, \tilde{h}[n]$ where
    $\tilde{h}$ is the reversed low-pass.  Decomposition filters are
    reversals of the corresponding reconstruction filters.
    """
    if name not in WAVELETS:
        raise ValueError(f"Unknown wavelet name: {name!r}")

    # Determine N and selection bits
    if name == "haar":
        order = 1
        bits: tuple[int, ...] = ()
    elif name.startswith("db"):
        order = int(name[2:])
        bits = (1,) * (order - 1)  # all ones for Daubechies
    else:  # sym
        order = int(name[3:])
        bits = _SYM_BITS[order]

    rec_lo = _compute_rec_lo(order, bits)

    # dec_lo = reversed rec_lo
    dec_lo = rec_lo[::-1].copy()

    # rec_hi[n] = (-1)^n * dec_lo[n]
    signs = np.where(np.arange(len(dec_lo)) % 2 == 0, 1.0, -1.0)
    rec_hi = (signs * dec_lo).astype(np.float64)

    # dec_hi = reversed rec_hi
    dec_hi = rec_hi[::-1].copy()

    return WaveletFilters(
        dec_lo=dec_lo.copy(),
        dec_hi=dec_hi.copy(),
        rec_lo=rec_lo.copy(),
        rec_hi=rec_hi.copy(),
    )
