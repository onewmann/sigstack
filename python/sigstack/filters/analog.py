"""Analog filter prototypes.

This module provides functions to compute analog lowpass filter prototypes
in zero-pole-gain form. These are used as building blocks for digital filters
via frequency transformations and the bilinear transform.

The prototypes follow SciPy's conventions:
- Butterworth: cutoff 1 rad/s, -3 dB at cutoff
- Chebyshev I: cutoff 1 rad/s, passband edge at -rp dB
- Chebyshev II: cutoff 1 rad/s, stopband edge at -rs dB
- Elliptic: cutoff 1 rad/s, passband edge at -rp dB, stopband edge at -rs dB

All poles are guaranteed to have negative real parts (stable).
"""

from __future__ import annotations

import math
import numbers

import numpy as np
from numpy.typing import NDArray

from sigstack.filters.elliptic import ellipj, ellipk

__all__ = ["buttap", "cheb1ap", "cheb2ap", "ellipap"]


def _check_order(n: object) -> int:
    """Validate a filter order and return it as a Python ``int``.

    Parameters
    ----------
    n : object
        Candidate order. Must be an integer (``bool`` is rejected); numpy
        integers are accepted.

    Returns
    -------
    int
        The order, converted to a Python ``int``.

    Raises
    ------
    TypeError
        If ``n`` is not an integer (in particular if it is a ``bool``).
    ValueError
        If ``n < 1``.
    """
    if isinstance(n, bool) or not isinstance(n, numbers.Integral):
        raise TypeError("Order must be an integer")
    value = int(n)
    if value < 1:
        raise ValueError("Order must be >= 1")
    return value


def _agm_k(a: float, b: float) -> float:
    r"""Complete elliptic integral $K$ via the arithmetic-geometric mean.

    Computes $K(m)$ for a parameter given by the initial AGM pair
    $(a, b)$ with $m = 1 - (b/a)^2$, i.e. $K = \pi / (2\,\mathrm{AGM}(a, b))$.
    Starting at $a = 1$, $b = \sqrt{m_0}$ yields $K(1 - m_0)$ without the
    loss of precision that evaluating $\ellipk(1 - m_0)$ directly suffers for
    small $m_0$.

    Parameters
    ----------
    a : float
        Initial arithmetic mean term (typically 1).
    b : float
        Initial geometric mean term (typically $\sqrt{m_0}$).

    Returns
    -------
    float
        The value of the complete elliptic integral.

    Notes
    -----
    With $k' = b/a$ for the starting pair, $K = \pi / (2\,\mathrm{AGM}(1, k'))$.

    References
    ----------
    .. [1] M. Abramowitz and I. A. Stegun, "Handbook of Mathematical
           Functions", NBS, 1964, Sec. 17.6.
    """
    eps = np.finfo(float).eps
    for _ in range(64):
        a_new = 0.5 * (a + b)
        b = math.sqrt(a * b)
        a = a_new
        if abs(a - b) <= 4.0 * eps * max(abs(a), abs(b)) or b == 0.0:
            break
    return float(math.pi / (2.0 * a))


def _ellipj_comp(u: float, m1: float) -> tuple[float, float, float]:
    r"""Jacobi elliptic functions at the complementary parameter $1 - m_1$.

    Computes $(\operatorname{sn}, \operatorname{cn}, \operatorname{dn})$ of a real
    argument $u$ for the parameter $m' = 1 - m_1$ without ever forming the
    floating-point value $1 - m_1$, which rounds to $1$ (and is therefore
    rejected by :func:`sigstack.filters.elliptic.ellipj`) once $m_1 < \varepsilon/2$.

    Parameters
    ----------
    u : float
        Real argument.
    m1 : float
        Parameter with $0 < m_1 \le 1$; the functions are evaluated at
        $m' = 1 - m_1$.

    Returns
    -------
    tuple of float
        ``(sn, cn, dn)`` at parameter $1 - m_1$.

    Raises
    ------
    ValueError
        If $m_1 \le 0$ or $m_1 > 1$.

    Notes
    -----
    Uses the descending Landen transformation (A&S Sec. 16.4). The AGM chain
    starts at $a_0 = 1$, $b_0 = \sqrt{m_1}$, $c_0 = \sqrt{1 - m_1}$ and iterates

    .. math::
        a_{i+1} = \tfrac{1}{2}(a_i + b_i),\quad
        b_{i+1} = \sqrt{a_i b_i},\quad
        c_{i+1} = \tfrac{1}{2}(a_i - b_i)

    until $c_N \le \varepsilon\, a_N$ (at most 64 steps). The angle is set to
    $\phi_N = 2^N a_N u$ and unwound for $i = N, \ldots, 1$:

    .. math::
        \phi_{i-1} = \tfrac{1}{2}\left(\phi_i + \arcsin\!\big((c_i/a_i)\sin\phi_i\big)\right).

    Then $\operatorname{sn} = \sin\phi_0$, $\operatorname{cn} = \cos\phi_0$ and
    $\operatorname{dn} = \sqrt{m_1 + (1 - m_1)\,\operatorname{cn}^2}$, the last from
    $dn^2 = 1 - m'\,sn^2$ with $m' = 1 - m_1$, which avoids cancellation near
    $u = K(1 - m_1)$.

    References
    ----------
    .. [1] M. Abramowitz and I. A. Stegun, "Handbook of Mathematical Functions",
           NBS, 1964, Sec. 16.4.
    """
    if not (0.0 < m1 <= 1.0):
        raise ValueError(f"m1 must satisfy 0 < m1 <= 1, got {m1}")

    eps = np.finfo(float).eps
    a_list: list[float] = [1.0]
    b_list: list[float] = [math.sqrt(m1)]
    c_list: list[float] = [math.sqrt(1.0 - m1)]

    for _ in range(64):
        a_prev = a_list[-1]
        b_prev = b_list[-1]
        a_new = 0.5 * (a_prev + b_prev)
        b_new = math.sqrt(a_prev * b_prev)
        c_new = 0.5 * (a_prev - b_prev)
        a_list.append(a_new)
        b_list.append(b_new)
        c_list.append(c_new)
        if abs(c_new) <= eps * a_new:
            break

    n_steps = len(a_list) - 1
    phi = (2.0**n_steps) * a_list[n_steps] * u

    for idx in range(n_steps, 0, -1):
        ratio = c_list[idx] / a_list[idx]
        phi = 0.5 * (phi + math.asin(ratio * math.sin(phi)))

    sn = math.sin(phi)
    cn = math.cos(phi)
    dn = math.sqrt(m1 + (1.0 - m1) * cn * cn)

    return sn, cn, dn


def buttap(n: int) -> tuple[NDArray[np.complex128], NDArray[np.complex128], float]:
    r"""Compute Butterworth analog lowpass prototype.

    The Butterworth filter has a maximally flat response in the passband.
    The poles are located on a circle of radius 1 in the left half-plane.

    Parameters
    ----------
    n : int
        Filter order (must be >= 1).

    Returns
    -------
    z : ndarray
        Zeros of the filter (empty array for Butterworth).
    p : ndarray
        Poles of the filter.
    k : float
        Gain of the filter.

    Raises
    ------
    TypeError
        If ``n`` is not an integer.
    ValueError
        If ``n < 1``.

    Notes
    -----
    The poles are given by:

    .. math::
        p_\ell = \exp\left( j \frac{(2\ell + n - 1)\pi}{2n} \right),
        \qquad \ell = 1, \ldots, n,

    the left-half-plane poles among the $2n$ roots of Eq. (B.3) in [2]. The gain is 1.

    References
    ----------
    .. [1] S. K. Mitra, *Digital Signal Processing: A Computer-Based Approach*, 4th ed.,
           McGraw-Hill, 2011, App. A.2, Eq. (A.11).
    .. [2] A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed.,
           Pearson, 2010, App. B.1, Eq. (B.3).
    """
    order = _check_order(n)

    # Poles on the unit circle in the left half-plane; the lower half-plane poles are the
    # exact conjugates of the upper ones, so conjugate pairs match to the last bit
    poles = np.empty(order, dtype=np.complex128)
    for i in range(order // 2):
        angle = (2 * i + 1) * np.pi / (2 * order)
        poles[i] = complex(-np.sin(angle), np.cos(angle))
        poles[order - 1 - i] = np.conj(poles[i])
    if order % 2:
        poles[order // 2] = -1.0

    return np.array([], dtype=np.complex128), poles, 1.0


def cheb1ap(n: int, rp: float) -> tuple[NDArray[np.complex128], NDArray[np.complex128], float]:
    r"""Compute Chebyshev Type I analog lowpass prototype.

    The Chebyshev Type I filter has an equiripple response in the passband
    and a monotonic response in the stopband. The passband edge is at 1 rad/s
    with ripple ``rp`` dB.

    Parameters
    ----------
    n : int
        Filter order (must be >= 1).
    rp : float
        Passband ripple in dB (must be > 0 and finite).

    Returns
    -------
    z : ndarray
        Zeros of the filter (empty array for Chebyshev Type I).
    p : ndarray
        Poles of the filter, all with negative real part.
    k : float
        Gain of the filter, chosen so that ``H(0) = 1`` for odd ``n`` and
        ``H(0) = 10**(-rp/20)`` for even ``n``.

    Raises
    ------
    TypeError
        If ``n`` is not an integer.
    ValueError
        If ``n < 1``, ``rp`` is not finite, ``rp <= 0``, or the ripple factor
        $10^{rp/10} - 1$ underflows to zero (``rp`` too small to resolve).

    Notes
    -----
    The poles lie on an ellipse in the left half-plane with semi-axes
    $\sinh(\mu)$ (real) and $\cosh(\mu)$ (imaginary), where

    .. math::
        \varepsilon = \sqrt{10^{rp/10} - 1}, \quad
        \mu = \frac{1}{n}\operatorname{arsinh}(1/\varepsilon)

    and the pole angles are $\theta_i = m_i \pi / (2n)$ with
    $m_i = 2i - 1 - n$ for $i = 1, \ldots, n$. The poles are

    .. math::
        p_i = -\sinh(\mu + j\theta_i)

    References
    ----------
    .. [1] S. K. Mitra, *Digital Signal Processing: A Computer-Based Approach*, 4th ed.,
           McGraw-Hill, 2011, App. A.3, Eqs. (A.18)-(A.19).
    """
    order = _check_order(n)
    if not math.isfinite(rp) or rp <= 0:
        raise ValueError(f"Ripple must be finite and > 0, got {rp}")

    # Ripple factor and ellipse parameter
    eps2 = 10.0 ** (rp / 10.0) - 1.0
    if eps2 == 0.0:
        raise ValueError("Passband ripple too small to resolve in double precision")
    eps = math.sqrt(eps2)
    mu = np.arcsinh(1.0 / eps) / order

    # Integer indices m_i = 2*i - 1 - n and angles theta_i
    m = np.arange(-order + 1, order, 2, dtype=np.float64)
    theta = np.pi * m / (2 * order)

    # Poles on the ellipse in the left half-plane
    poles = -np.sinh(mu + 1j * theta)

    # No finite zeros for Chebyshev Type I
    zeros = np.array([], dtype=np.complex128)

    # Gain so that H(0) = 1 (odd n) or 10**(-rp/20) (even n)
    k = float(np.real(np.prod(-poles)))
    if order % 2 == 0:
        k = k / math.sqrt(1.0 + eps2)

    return zeros, poles.astype(np.complex128), k


def cheb2ap(n: int, rs: float) -> tuple[NDArray[np.complex128], NDArray[np.complex128], float]:
    r"""Compute Chebyshev Type II analog lowpass prototype.

    The Chebyshev Type II filter has a maximally flat response in the
    passband (``H(0) = 1``) and an equiripple response in the stopband.
    The stopband edge is at 1 rad/s, where the attenuation is exactly
    ``rs`` dB.

    Parameters
    ----------
    n : int
        Filter order (must be >= 1).
    rs : float
        Stopband attenuation in dB (must be > 0 and finite).

    Returns
    -------
    z : ndarray
        Zeros of the filter, on the imaginary axis. There are ``n`` zeros
        for even ``n`` and ``n - 1`` for odd ``n`` (none for ``n = 1``).
    p : ndarray
        Poles of the filter, all with negative real part.
    k : float
        Gain of the filter, chosen so that ``H(0) = 1``.

    Raises
    ------
    TypeError
        If ``n`` is not an integer.
    ValueError
        If ``n < 1``, ``rs`` is not finite, ``rs <= 0``, or the attenuation
        factor $10^{rs/10} - 1$ underflows to zero (``rs`` too small).

    Notes
    -----
    The inverse Chebyshev (Type II) prototype is obtained by inverting the
    poles of a Chebyshev Type I ellipse and placing finite zeros on the
    imaginary axis. Let

    .. math::
        \delta_e = \frac{1}{\sqrt{10^{rs/10} - 1}},\quad
        \mu = \frac{1}{n}\operatorname{arsinh}(1/\delta_e)

    and, for ``i = 1, ..., n``, the integers ``m_i = 2i - 1 - n`` with
    angles ``\theta_i = m_i \pi / (2n)``. The poles are the inverses of
    the Type I ellipse points:

    .. math::
        p_i = -\frac{1}{\sinh(\mu + j\theta_i)}

    and the finite zeros lie on the imaginary axis:

    .. math::
        z_i = \frac{j}{\sin(\theta_i)},\qquad m_i \neq 0.

    For odd ``n`` the middle index has ``m_i = 0`` (a zero at infinity) and
    is dropped. The gain is

    .. math::
        k = \operatorname{Re}\left(\frac{\prod_i (-p_i)}{\prod_i (-z_i)}\right)

    so that the DC gain is exactly 1.

    References
    ----------
    .. [1] S. K. Mitra, *Digital Signal Processing: A Computer-Based Approach*, 4th ed.,
           McGraw-Hill, 2011, App. A.3, Eqs. (A.20)-(A.24).
    .. [2] A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed.,
           Pearson, 2010, App. B.2, Eq. (B.10).
    """
    order = _check_order(n)
    if not math.isfinite(rs) or rs <= 0:
        raise ValueError(f"Stopband attenuation must be finite and > 0, got {rs}")

    # Inverse ripple factor and ellipse parameter
    ds2 = 10.0 ** (rs / 10.0) - 1.0
    if ds2 == 0.0:
        raise ValueError("Stopband attenuation too small to resolve in double precision")
    de = 1.0 / math.sqrt(ds2)
    mu = np.arcsinh(1.0 / de) / order

    # Integer indices m_i = 2*i - 1 - n and angles theta_i
    m = np.arange(-order + 1, order, 2, dtype=np.float64)
    theta = np.pi * m / (2 * order)

    # Poles: inverses of the Chebyshev Type I ellipse points
    poles = -1.0 / np.sinh(mu + 1j * theta)

    # Zeros on the imaginary axis, dropping the zero at infinity (m_i == 0)
    mask = m != 0
    zeros = 1j / np.sin(theta[mask])

    # Gain so that H(0) = 1; np.prod of an empty array is 1.0
    k = float(np.real(np.prod(-poles) / np.prod(-zeros)))

    return zeros.astype(np.complex128), poles.astype(np.complex128), k


def ellipap(
    n: int, rp: float, rs: float
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], float]:
    r"""Compute elliptic (Cauer) analog lowpass prototype.

    The elliptic filter has an equiripple response in both the passband and
    the stopband, achieving the steepest transition for a given order. The
    passband edge is at 1 rad/s with ripple ``rp`` dB and the stopband edge
    is at 1 rad/s with attenuation ``rs`` dB (SciPy convention).

    Parameters
    ----------
    n : int
        Filter order (must be >= 1).
    rp : float
        Passband ripple in dB (must be > 0 and finite).
    rs : float
        Stopband attenuation in dB (must be > ``rp`` and finite).

    Returns
    -------
    z : ndarray
        Zeros of the filter, on the imaginary axis. There are ``n`` zeros
        for even ``n`` and ``n - 1`` for odd ``n`` (none for ``n = 1``).
    p : ndarray
        Poles of the filter, all with negative real part.
    k : float
        Gain of the filter.

    Raises
    ------
    TypeError
        If ``n`` is not an integer.
    ValueError
        If ``n < 1``, ``rp`` is not finite or ``<= 0``, the ripple factor
        $10^{rp/10} - 1$ underflows to zero, ``rs`` is not finite, or
        ``rs <= rp``.

    Notes
    -----
    The squared magnitude is (Mitra, App. A.4, Eq. A.26)

    .. math::
        |H(j\omega)|^2 = \frac{1}{1 + \varepsilon^2 R_n^2(\omega)}

    where $R_n$ is the $n$-th degree Jacobi elliptic function and
    $\varepsilon = \sqrt{10^{rp/10} - 1}$ is the passband ripple factor.
    The selectivity modulus $k$ solves the degree equation

    .. math::
        n\,K(k') / K(k) = K(k_1') / K(k_1),

    with $k_1^2 = \varepsilon^2 / (10^{rs/10} - 1)$ the discrimination
    modulus. The solution uses the nome $q = \exp(-\pi K'/K)$ (A&S 16.27)
    and the theta-function series for $K$ and $kK$ (A&S 16.38.5, 16.38.7),
    whose ratio gives $k$. Zeros are

    .. math::
        z_j = \frac{j}{\sqrt{m}\,\operatorname{sn}(u_j | m)},\quad
        u_j = j K(k) / n,

    and poles follow from $\operatorname{sn}$ of a complex argument
    (A&S 16.21.2). The gain is chosen so that $H(0) = 1$ for odd ``n`` and
    $H(0) = 10^{-rp/20}$ for even ``n``.

    References
    ----------
    .. [1] S. K. Mitra, *Digital Signal Processing: A Computer-Based Approach*, 4th ed.,
           McGraw-Hill, 2011, App. A.4, Eq. (A.26).
    .. [2] M. Abramowitz and I. A. Stegun, "Handbook of Mathematical
           Functions", NBS, 1964, ch. 16-17.
    """
    order = _check_order(n)
    if not math.isfinite(rp) or rp <= 0:
        raise ValueError(f"Ripple must be finite and > 0, got {rp}")
    if not math.isfinite(rs) or rs <= rp:
        raise ValueError(
            f"Stopband attenuation must be finite and > passband ripple, got rs={rs}, rp={rp}"
        )

    # Passband ripple factor and discrimination modulus k1^2 = m1.
    eps2 = 10.0 ** (rp / 10.0) - 1.0
    if eps2 == 0.0:
        raise ValueError("Passband ripple too small to resolve in double precision")
    eps = math.sqrt(eps2)
    m1 = eps2 / (10.0 ** (rs / 10.0) - 1.0)

    # Order one: a single real pole, no finite zeros.
    if order == 1:
        z = np.array([], dtype=np.complex128)
        p = np.array([-1.0 / eps], dtype=np.complex128)
        return z, p, float(1.0 / eps)

    # Complete elliptic integrals for the discrimination modulus. K(k1) is
    # evaluated directly; K(k1') = K(1 - m1) via the AGM to keep precision.
    kk1 = ellipk(m1)
    kk1p = _agm_k(1.0, math.sqrt(m1))

    # Solve the degree equation for the selectivity modulus k using the
    # nome q (A&S 16.27) and the theta series for K and kK (A&S 16.38.5, 16.38.7).
    q1 = math.exp(-math.pi * kk1p / kk1)
    q = q1 ** (1.0 / order)
    num = sum(q ** (i * (i + 1)) for i in range(8))
    den = 1.0 + 2.0 * sum(q ** (i * i) for i in range(1, 9))
    m = 16.0 * q * (num / den) ** 4
    kk = ellipk(m)

    # Argument indices jj: 1,3,..,n-1 (even n) or 0,2,..,n-1 (odd n).
    jj = np.arange(1 - order % 2, order, 2, dtype=np.float64)
    s, c, d = ellipj(jj * kk / order, m)

    # Zeros on the imaginary axis; jj == 0 is the zero at infinity.
    nz = jj != 0
    z = 1j / (math.sqrt(m) * s[nz])
    z = np.concatenate((z, np.conj(z)))

    # Shift v0: solve sn(r|1-m1)/cn(r|1-m1) = 1/eps for r in (0, kk1p).
    def g(r: float) -> float:
        sv, cv, _ = _ellipj_comp(r, m1)
        return float(sv - cv / eps)

    lo, hi = 0.0, kk1p
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if g(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)
    v0 = kk * r / (order * kk1)

    # Poles via sn of a complex argument (A&S 16.21.2).
    sv, cv, dv = _ellipj_comp(v0, m)
    p = (-(c * d * sv * cv + 1j * s * dv) / (1.0 - (d * sv) ** 2)).astype(np.complex128)
    p = np.concatenate((p, np.conj(p[jj != 0])))

    # Gain so that H(0) = 1 (odd n) or 10**(-rp/20) (even n).
    k = float(np.real(np.prod(-p) / np.prod(-z)))
    if order % 2 == 0:
        k = k / math.sqrt(1.0 + eps2)

    return z.astype(np.complex128), p.astype(np.complex128), k
