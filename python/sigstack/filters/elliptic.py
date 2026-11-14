"""Complete elliptic integral of the first kind and Jacobi elliptic functions."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

__all__ = ["ellipj", "ellipk"]


def ellipk(m: float) -> float:
    r"""Compute the complete elliptic integral of the first kind.

    Parameters
    ----------
    m : float
        Parameter $m = k^2$ with $0 \le m < 1$.

    Returns
    -------
    float
        The value of $K(m) = \int_0^{\pi/2} (1 - m \sin^2\theta)^{-1/2}\,d\theta$.

    Raises
    ------
    ValueError
        If $m < 0$, $m \ge 1$, or $m$ is NaN.

    Notes
    -----
    Uses the arithmetic-geometric mean (AGM) algorithm,
    $K(m) = \pi / (2\,\mathrm{AGM}(1, \sqrt{1 - m}))$.

    References
    ----------
    .. [1] M. Abramowitz and I. A. Stegun, "Handbook of Mathematical
           Functions", NBS, 1964, Sec. 17.6.
    """
    if not (0.0 <= m < 1.0):
        raise ValueError(f"m must satisfy 0 <= m < 1, got {m}")
    if m == 0.0:
        return float(math.pi / 2.0)

    a = 1.0
    b = math.sqrt(1.0 - m)
    eps = np.finfo(float).eps

    for _ in range(64):
        a_new = 0.5 * (a + b)
        b = math.sqrt(a * b)
        a = a_new
        if abs(a - b) <= 4.0 * eps * max(abs(a), abs(b)) or b == 0.0:
            break

    return float(math.pi / (2.0 * a))


def ellipj(
    u: float | np.ndarray, m: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Compute Jacobi elliptic functions sn, cn, dn.

    Parameters
    ----------
    u : float or numpy.ndarray
        Argument(s).
    m : float
        Parameter $m = k^2$ with $0 \le m < 1$.

    Returns
    -------
    tuple of numpy.ndarray
        ``(sn, cn, dn)`` arrays with the same shape as ``u``.

    Raises
    ------
    ValueError
        If $m < 0$, $m \ge 1$, or $m$ is NaN.

    Notes
    -----
    Uses the arithmetic-geometric mean (AGM) method of Abramowitz and
    Stegun, section 16.4 (AGM scale as in 17.6).  For $m = 0$ the
    trigonometric limits are returned directly.  For $0 < m < 1$ the AGM
    iterates are computed scalar-wise and the angular recursion is
    evaluated vectorially over ``u``.

    The identity $dn^2 = 1 - m\,sn^2 = (1 - m) + m\,cn^2$ is used to compute
    dn without cancellation near $u = (2j+1)K$, where the quotient form
    $cn / \cos(\phi_{n-1} - \phi_n)$ becomes 0/0.
    """
    if not (0.0 <= m < 1.0):
        raise ValueError(f"m must satisfy 0 <= m < 1, got {m}")

    u_arr = np.asarray(u, dtype=np.float64)

    if m == 0.0:
        return np.sin(u_arr), np.cos(u_arr), np.ones_like(u_arr)

    a_list: list[float] = [1.0]
    b_list: list[float] = [math.sqrt(1.0 - m)]
    c_list: list[float] = [math.sqrt(m)]

    for _ in range(40):
        a_prev = a_list[-1]
        b_prev = b_list[-1]
        a_new = 0.5 * (a_prev + b_prev)
        b_new = math.sqrt(a_prev * b_prev)
        c_new = 0.5 * (a_prev - b_prev)
        a_list.append(a_new)
        b_list.append(b_new)
        c_list.append(c_new)
        if abs(c_new) < 1e-16 * a_new:
            break

    n = len(a_list) - 1
    phi = (2.0**n) * a_list[n] * u_arr

    for idx in range(n, 0, -1):
        ratio = c_list[idx] / a_list[idx]
        phi = 0.5 * (phi + np.arcsin(ratio * np.sin(phi)))

    sn = np.sin(phi)
    cn = np.cos(phi)
    dn = np.sqrt((1.0 - m) + m * cn**2)

    return sn, cn, dn
