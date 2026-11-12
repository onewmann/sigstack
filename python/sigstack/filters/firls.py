r"""Least-squares FIR filter design (firls).

This module implements least-squares FIR filter design.
The method minimises the weighted integral of the squared error between the actual
amplitude response A(f) and a piecewise-linear desired amplitude D(f) over specified
frequency bands.

For an odd-length linear-phase FIR filter with M = (numtaps - 1) // 2, the amplitude
response is:

    :math:`A(f) = \\sum_{j=0}^{M} a_j \\cos(\\pi j f)`

where f is normalised so that Nyquist frequency equals 1. The filter coefficients are
recovered from the amplitude coefficients via:

    :math:`h[M] = a_0, \\quad h[M-j] = h[M+j] = a_j / 2`

The design problem minimises:

    :math:`\\sum_{b} w_b \\int_{f_{1,b}}^{f_{2,b}} (A(f) - D_b(f))^2 df`

where D_b(f) is linear between the desired values at the band edges.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["firls"]


def firls(
    numtaps: int,
    bands: float | list[float] | NDArray[np.float64],
    desired: float | list[float] | NDArray[np.float64],
    weight: float | list[float] | NDArray[np.float64] | None = None,
    fs: float = 2.0,
) -> NDArray[np.float64]:
    """Design a linear-phase FIR filter by least squares.

    Minimises the weighted integral of the squared error between the actual amplitude
    response and a piecewise-linear desired amplitude over specified frequency bands.

    Parameters
    ----------
    numtaps : int
        Number of filter taps (must be odd).
    bands : float or array_like
        Even-length sequence of band edges in [0, fs/2]. Edges must be non-decreasing.
    desired : float or array_like
        Desired amplitude at each band edge. Must have the same length as ``bands``.
    weight : float or array_like or None, optional
        Weight for each band. If None, all weights are 1. Must be positive.
    fs : float, optional
        Sampling frequency. Default is 2.0 (normalised).

    Returns
    -------
    numpy.ndarray
        FIR filter coefficients of shape (numtaps,), dtype float64.

    Raises
    ------
    ValueError
        If any input parameter is invalid.

    """
    # Validate numtaps
    if isinstance(numtaps, bool) or not isinstance(numtaps, int):
        raise ValueError("numtaps must be an integer")
    if numtaps <= 0:
        raise ValueError("numtaps must be positive")
    if numtaps % 2 == 0:
        raise ValueError("numtaps must be odd")

    # Validate fs
    if not isinstance(fs, (int, float)) or isinstance(fs, bool):
        raise ValueError("fs must be a number")
    if fs <= 0:
        raise ValueError("fs must be positive")

    # Convert bands to array and validate
    bands_arr = np.asarray(bands, dtype=np.float64)
    if bands_arr.ndim != 1:
        raise ValueError("bands must be a 1-D sequence")
    if len(bands_arr) % 2 != 0:
        raise ValueError("bands must have an even number of edges")
    if len(bands_arr) < 2:
        raise ValueError("bands must have at least 2 edges")

    # Normalize band edges by fs/2
    nyquist = fs / 2.0
    bands_norm = bands_arr / nyquist

    # Validate band edges are within [0, 1] and non-decreasing
    if np.any(bands_norm < 0) or np.any(bands_norm > 1):
        raise ValueError("band edges must be within [0, fs/2]")
    if np.any(np.diff(bands_norm) < 0):
        raise ValueError("band edges must be non-decreasing")

    # Convert desired to array and validate
    desired_arr = np.asarray(desired, dtype=np.float64)
    if desired_arr.ndim != 1:
        raise ValueError("desired must be a 1-D sequence")
    if len(desired_arr) != len(bands_arr):
        raise ValueError(
            f"desired must have the same length as bands ({len(bands_arr)}), got {len(desired_arr)}"
        )

    # Convert weight to array and validate
    n_bands = len(bands_arr) // 2
    if weight is None:
        weights = np.ones(n_bands, dtype=np.float64)
    else:
        weights = np.asarray(weight, dtype=np.float64)
        if weights.ndim != 1:
            raise ValueError("weight must be a 1-D sequence")
        if len(weights) != n_bands:
            raise ValueError(
                f"weight must have {n_bands} values (one per band), got {len(weights)}"
            )
        if np.any(weights <= 0):
            raise ValueError("weights must be positive")

    # M = (numtaps - 1) // 2
    m_half = (numtaps - 1) // 2

    # Build the normal equations Q a = c
    # Q[i, j] = sum_b w_b * int_{f1}^{f2} cos(pi*i*f) * cos(pi*j*f) df
    # c[i] = sum_b w_b * int_{f1}^{f2} D(f) * cos(pi*i*f) df

    q_mat = np.zeros((m_half + 1, m_half + 1), dtype=np.float64)
    c = np.zeros(m_half + 1, dtype=np.float64)

    for b in range(n_bands):
        f1 = bands_norm[2 * b]
        f2 = bands_norm[2 * b + 1]
        w = weights[b]

        # Desired amplitude is linear: D(f) = s*f + d
        d0 = desired_arr[2 * b]
        d1 = desired_arr[2 * b + 1]
        if f2 != f1:
            s = (d1 - d0) / (f2 - f1)
            d_const = d0 - s * f1
        else:
            s = 0.0
            d_const = d0

        for i in range(m_half + 1):
            # c[i] = w * int_{f1}^{f2} (s*f + d_const) * cos(pi*i*f) df
            # = w * [ s * int f*cos(pi*i*f) df + d_const * int cos(pi*i*f) df ]

            if i == 0:
                # int cos(0) df = int 1 df = f2 - f1
                # int f*cos(0) df = int f df = (f2^2 - f1^2)/2
                integral_cos = f2 - f1
                integral_fcos = 0.5 * (f2**2 - f1**2)
            else:
                pi_i = np.pi * i
                # int cos(pi*i*f) df = sin(pi*i*f)/(pi*i)
                integral_cos = (np.sin(np.pi * i * f2) - np.sin(np.pi * i * f1)) / pi_i
                # int f*cos(pi*i*f) df = f*sin(pi*i*f)/(pi*i) + cos(pi*i*f)/(pi*i)^2
                integral_fcos = (
                    f2 * np.sin(np.pi * i * f2) / pi_i
                    - f1 * np.sin(np.pi * i * f1) / pi_i
                    + (np.cos(np.pi * i * f2) - np.cos(np.pi * i * f1)) / (pi_i**2)
                )

            c[i] += w * (s * integral_fcos + d_const * integral_cos)

            for j in range(i, m_half + 1):
                # Q[i, j] = w * int cos(pi*i*f)*cos(pi*j*f) df
                # = w/2 * [ int cos(pi*(i-j)*f) df + int cos(pi*(i+j)*f) df ]

                if i == j:
                    # int cos(0) df = f2 - f1
                    integral_diff = f2 - f1
                else:
                    pi_diff = np.pi * (i - j)
                    integral_diff = (
                        np.sin(np.pi * (i - j) * f2) - np.sin(np.pi * (i - j) * f1)
                    ) / pi_diff

                pi_sum = np.pi * (i + j)
                if i + j == 0:
                    integral_sum = f2 - f1
                else:
                    integral_sum = (
                        np.sin(np.pi * (i + j) * f2) - np.sin(np.pi * (i + j) * f1)
                    ) / pi_sum

                q_mat[i, j] += w * 0.5 * (integral_diff + integral_sum)
                if i != j:
                    q_mat[j, i] = q_mat[i, j]

    # Solve the normal equations
    try:
        a = np.linalg.solve(q_mat, c)
    except np.linalg.LinAlgError:
        a, _, _, _ = np.linalg.lstsq(q_mat, c, rcond=None)

    # Reconstruct filter coefficients from amplitude coefficients
    h = np.zeros(numtaps, dtype=np.float64)
    h[m_half] = a[0]
    for j in range(1, m_half + 1):
        h[m_half - j] = a[j] / 2.0
        h[m_half + j] = a[j] / 2.0

    return h
