"""Zero-pole-gain and second-order-section utilities for IIR filters."""

from __future__ import annotations

import math
import numbers

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "bilinear_zpk",
    "lp2bp_zpk",
    "lp2bs_zpk",
    "lp2hp_zpk",
    "lp2lp_zpk",
    "sos_response",
    "zpk2sos",
]


def _as_complex_1d(values: ArrayLike) -> NDArray[np.complex128]:
    """Coerce input to a 1-D complex128 array.

    Parameters
    ----------
    values : array_like
        Input values.

    Returns
    -------
    NDArray[np.complex128]
        1-D array with dtype complex128.

    Raises
    ------
    ValueError
        If the input is not 1-D (i.e., has more than one dimension).
    """
    arr = np.asarray(values, dtype=np.complex128)
    if arr.ndim > 1:
        raise ValueError("z and p must be 1-D")
    return np.atleast_1d(arr)


def _check_finite_roots(roots: NDArray[np.complex128], name: str) -> None:
    """Raise ``ValueError`` if any root is not finite.

    Parameters
    ----------
    roots : NDArray[np.complex128]
        1-D array of roots to check.
    name : str
        Name of the argument, used in the error message.

    Raises
    ------
    ValueError
        If any root has a non-finite real or imaginary part.
    """
    if not np.all(np.isfinite(roots)):
        raise ValueError(f"{name} must contain only finite values")


def _check_finite_gain(k: object) -> float:
    """Validate the gain and return it as a Python ``float``.

    Parameters
    ----------
    k : object
        Gain to check: a real number (Python or NumPy).

    Returns
    -------
    float
        The gain.

    Raises
    ------
    ValueError
        If the gain is complex (also with a zero imaginary part) or not finite.
    TypeError
        If the gain is not a number.
    """
    if isinstance(k, (complex, np.complexfloating)):
        raise ValueError(f"k must be real, got {k!r}")
    if not isinstance(k, numbers.Real):
        raise TypeError(f"k must be a real number, got {type(k).__name__}")
    value = float(k)
    if not math.isfinite(value):
        raise ValueError(f"k must be finite, got {value}")
    return value


def bilinear_zpk(
    z: ArrayLike, p: ArrayLike, k: float, fs: float
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], float]:
    r"""Map an analog prototype to the digital domain via the bilinear transform.

    The substitution $s = 2 f_s \frac{z - 1}{z + 1}$ maps the imaginary axis of the
    $s$-plane onto the unit circle in the $z$-plane (and the left half-plane into its
    interior), so an analog filter with zeros $z$, poles $p$ and gain $k$ becomes a
    digital filter whose zeros and poles are the images of the originals under that map.

    With $f_s$ the sampling frequency and $d = \text{len}(p) - \text{len}(z)$ the
    number of zeros at infinity, the transformed quantities are

    .. math::

        z_i' = \frac{2 f_s + z_i}{2 f_s - z_i},\qquad
        p_j' = \frac{2 f_s + p_j}{2 f_s - p_j},

    together with $d$ additional zeros at $z = -1$, and the gain is scaled by the
    ratio of the products over the original roots,

    .. math::

        k' = k\,\operatorname{Re}\!\left(
            \prod_i (2 f_s - z_i)\Big/\prod_j (2 f_s - p_j)\right).

    Prewarping of the prototype frequencies is the caller's responsibility.

    Parameters
    ----------
    z : array_like
        Analog zeros of the prototype filter.
    p : array_like
        Analog poles of the prototype filter.
    k : float
        Gain of the prototype filter.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    tuple[NDArray[np.complex128], NDArray[np.complex128], float]
        The digital zeros, poles and gain.

    Raises
    ------
    ValueError
        If ``z`` or ``p`` is not 1-D, if there are more zeros than poles, or if any
        root or the gain is not finite.

    References
    ----------
    A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed.,
    Pearson, 2010, Sec. 7.2.2, Eq. (7.18).
    """
    z = _as_complex_1d(z)
    p = _as_complex_1d(p)

    if len(z) > len(p):
        raise ValueError("need at least as many poles as zeros")

    _check_finite_roots(z, "z")
    _check_finite_roots(p, "p")
    k = _check_finite_gain(k)

    d = len(p) - len(z)
    fs2 = 2.0 * float(fs)

    zeros = np.concatenate(((fs2 + z) / (fs2 - z), -np.ones(d, dtype=np.complex128)))
    poles = (fs2 + p) / (fs2 - p)
    gain = k * float(np.real(np.prod(fs2 - z) / np.prod(fs2 - p)))

    return zeros, poles, gain


def lp2lp_zpk(
    z: ArrayLike, p: ArrayLike, k: float, wo: float
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], float]:
    r"""Scale an analog lowpass prototype to a new cutoff frequency.

    A lowpass prototype with cutoff $\omega_0 = 1$ is mapped to one with cutoff
    $\omega_o$ by the frequency scaling $s \mapsto s / \omega_o$. Every zero and
    pole is therefore multiplied by $\omega_o$, and the gain is scaled so that the
    response at the new cutoff matches the prototype's response at its own.

    With $d = \text{len}(p) - \text{len}(z)$ the number of zeros at infinity,

    .. math::

        z_i' = z_i\,\omega_o,\qquad
        p_j' = p_j\,\omega_o,\qquad
        k' = k\,\omega_o^{d}.

    Parameters
    ----------
    z : array_like
        Zeros of the lowpass prototype.
    p : array_like
        Poles of the lowpass prototype.
    k : float
        Gain of the lowpass prototype.
    wo : float
        Desired cutoff frequency in rad/s.

    Returns
    -------
    tuple[NDArray[np.complex128], NDArray[np.complex128], float]
        The transformed zeros, poles and gain.

    Raises
    ------
    ValueError
        If ``z`` or ``p`` is not 1-D, if there are more zeros than poles, or if any
        root or the gain is not finite.

    References
    ----------
    S. K. Mitra, *Digital Signal Processing: A Computer-Based Approach*, 4th ed.,
    McGraw-Hill, 2011, App. A.2, Eq. (A.11).
    """
    z = _as_complex_1d(z)
    p = _as_complex_1d(p)

    if len(z) > len(p):
        raise ValueError("need at least as many poles as zeros")

    _check_finite_roots(z, "z")
    _check_finite_roots(p, "p")
    k = _check_finite_gain(k)

    d = len(p) - len(z)
    wo = float(wo)

    zeros = z * wo
    poles = p * wo
    gain = k * wo**d

    return zeros, poles, gain


def lp2hp_zpk(
    z: ArrayLike, p: ArrayLike, k: float, wo: float
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], float]:
    r"""Transform an analog lowpass prototype into a highpass filter.

    The substitution $s \mapsto \omega_o / s$ inverts the frequency axis, turning
    a lowpass response into a highpass one with cutoff $\omega_o$. Each zero and
    pole is mapped to its reciprocal scaled by $\omega_o$, and the $d =
    \text{len}(p) - \text{len}(z)$ zeros at infinity become zeros at the origin.

    .. math::

        z_i' = \frac{\omega_o}{z_i},\qquad
        p_j' = \frac{\omega_o}{p_j},

    with $d$ additional zeros at $0$, and a gain scaled by the ratio of the
    products over the original roots,

    .. math::

        k' = k\,\operatorname{Re}\!\left(
            \prod_i (-z_i)\Big/\prod_j (-p_j)\right).

    Parameters
    ----------
    z : array_like
        Zeros of the lowpass prototype.
    p : array_like
        Poles of the lowpass prototype.
    k : float
        Gain of the lowpass prototype.
    wo : float
        Desired cutoff frequency in rad/s.

    Returns
    -------
    tuple[NDArray[np.complex128], NDArray[np.complex128], float]
        The transformed zeros, poles and gain.

    Raises
    ------
    ValueError
        If ``z`` or ``p`` is not 1-D, if there are more zeros than poles, or if any
        root or the gain is not finite.

    References
    ----------
    S. K. Mitra, *Digital Signal Processing: A Computer-Based Approach*, 4th ed.,
    McGraw-Hill, 2011, App. B.1, Eq. (B.1).
    """
    z = _as_complex_1d(z)
    p = _as_complex_1d(p)

    if len(z) > len(p):
        raise ValueError("need at least as many poles as zeros")

    _check_finite_roots(z, "z")
    _check_finite_roots(p, "p")
    k = _check_finite_gain(k)

    d = len(p) - len(z)
    wo = float(wo)

    zeros = np.concatenate((wo / z, np.zeros(d, dtype=np.complex128)))
    poles = wo / p
    gain = k * float(np.real(np.prod(-z) / np.prod(-p)))

    return zeros, poles, gain


def lp2bp_zpk(
    z: ArrayLike, p: ArrayLike, k: float, wo: float, bw: float
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], float]:
    r"""Transform an analog lowpass prototype into a bandpass filter.

    The substitution $s \mapsto (s^2 + \omega_o^2) / (b_w s)$ maps the lowpass
    cutoff (1 rad/s) to a passband centred at $\omega_o$ with bandwidth $b_w$. Each
    zero and pole is replaced by the two roots of the quadratic in $s$,

    .. math::

        z_i' = \frac{z_i b_w}{2} \pm
            \sqrt{\left(\frac{z_i b_w}{2}\right)^{2} - \omega_o^{2}},

    and likewise for the poles, with $d = \text{len}(p) - \text{len}(z)$ extra
    zeros at the origin. The gain is scaled by $b_w^{d}$:

    .. math::

        k' = k\,b_w^{d}.

    Parameters
    ----------
    z : array_like
        Zeros of the lowpass prototype.
    p : array_like
        Poles of the lowpass prototype.
    k : float
        Gain of the lowpass prototype.
    wo : float
        Centre frequency of the passband in rad/s.
    bw : float
        Bandwidth of the passband in rad/s.

    Returns
    -------
    tuple[NDArray[np.complex128], NDArray[np.complex128], float]
        The transformed zeros, poles and gain.

    Raises
    ------
    ValueError
        If ``z`` or ``p`` is not 1-D, if there are more zeros than poles, or if any
        root or the gain is not finite.

    References
    ----------
    S. K. Mitra, *Digital Signal Processing: A Computer-Based Approach*, 4th ed.,
    McGraw-Hill, 2011, App. B.2, Eq. (B.3).
    """
    z = _as_complex_1d(z)
    p = _as_complex_1d(p)

    if len(z) > len(p):
        raise ValueError("need at least as many poles as zeros")

    _check_finite_roots(z, "z")
    _check_finite_roots(p, "p")
    k = _check_finite_gain(k)

    d = len(p) - len(z)
    wo = float(wo)
    bw = float(bw)

    zl = z * (bw / 2.0)
    pl = p * (bw / 2.0)

    zeros = np.concatenate(
        (zl + np.sqrt(zl**2 - wo**2), zl - np.sqrt(zl**2 - wo**2), np.zeros(d, dtype=np.complex128))
    )
    poles = np.concatenate((pl + np.sqrt(pl**2 - wo**2), pl - np.sqrt(pl**2 - wo**2)))
    gain = k * bw**d

    return zeros, poles, gain


def lp2bs_zpk(
    z: ArrayLike, p: ArrayLike, k: float, wo: float, bw: float
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], float]:
    r"""Transform an analog lowpass prototype into a bandstop filter.

    The substitution $s \mapsto b_w s / (s^2 + \omega_o^2)$ maps the lowpass
    cutoff (1 rad/s) to a stopband centred at $\omega_o$ with bandwidth $b_w$. Each
    zero and pole is replaced by the two roots of the quadratic in $s$,

    .. math::

        z_i' = \frac{b_w / 2}{z_i} \pm
            \sqrt{\left(\frac{b_w / 2}{z_i}\right)^{2} - \omega_o^{2}},

    and likewise for the poles, with $d = \text{len}(p) - \text{len}(z)$ extra
    zeros at each of $+j\omega_o$ and $-j\omega_o$. The gain is scaled by the ratio
    of the products over the original roots:

    .. math::

        k' = k\,\operatorname{Re}\!\left(
            \prod_i (-z_i)\Big/\prod_j (-p_j)\right).

    Parameters
    ----------
    z : array_like
        Zeros of the lowpass prototype.
    p : array_like
        Poles of the lowpass prototype.
    k : float
        Gain of the lowpass prototype.
    wo : float
        Centre frequency of the stopband in rad/s.
    bw : float
        Bandwidth of the stopband in rad/s.

    Returns
    -------
    tuple[NDArray[np.complex128], NDArray[np.complex128], float]
        The transformed zeros, poles and gain.

    Raises
    ------
    ValueError
        If ``z`` or ``p`` is not 1-D, if there are more zeros than poles, or if any
        root or the gain is not finite.

    References
    ----------
    S. K. Mitra, *Digital Signal Processing: A Computer-Based Approach*, 4th ed.,
    McGraw-Hill, 2011, App. B.3, Eq. (B.6).
    """
    z = _as_complex_1d(z)
    p = _as_complex_1d(p)

    if len(z) > len(p):
        raise ValueError("need at least as many poles as zeros")

    _check_finite_roots(z, "z")
    _check_finite_roots(p, "p")
    k = _check_finite_gain(k)

    d = len(p) - len(z)
    wo = float(wo)
    bw = float(bw)

    zh = (bw / 2.0) / z
    ph = (bw / 2.0) / p

    zeros = np.concatenate(
        (
            zh + np.sqrt(zh**2 - wo**2),
            zh - np.sqrt(zh**2 - wo**2),
            np.full(d, 1j * wo),
            np.full(d, -1j * wo),
        )
    )
    poles = np.concatenate((ph + np.sqrt(ph**2 - wo**2), ph - np.sqrt(ph**2 - wo**2)))
    gain = k * float(np.real(np.prod(-z) / np.prod(-p)))

    return zeros, poles, gain


def _classify_roots(roots: NDArray[np.complex128]) -> list[list[np.complex128]]:
    r"""Classify roots into real roots and exact conjugate pairs.

    For each root $r$: if $|\operatorname{Im}(r)| \le 100\,\varepsilon\,
    \max(1, |r|)$ it is treated as real. Otherwise the nearest remaining root to
    $\overline{r}$ must be within that tolerance; otherwise a ``ValueError`` is
    raised. Valid complex pairs are symmetrised so that both members are exact
    conjugates of their midpoint (with positive imaginary part).

    Parameters
    ----------
    roots : NDArray[np.complex128]
        1-D array of roots to classify.

    Returns
    -------
    list[list[np.complex128]]
        Groups of roots: each group is either a single real root ``[r]`` or an
        exact conjugate pair ``[q, conj(q)]`` with ``Im(q) > 0``.

    Raises
    ------
    ValueError
        If a complex root has no conjugate partner within tolerance.
    """
    eps = np.finfo(np.float64).eps
    remaining = list(roots)
    groups: list[list[np.complex128]] = []

    while remaining:
        r = remaining.pop(0)
        tol = 100.0 * eps * max(1.0, abs(r))
        if abs(r.imag) <= tol:
            groups.append([r.real + 0j])
        else:
            # Find the nearest remaining root to conj(r)
            best_idx = -1
            best_dist = float("inf")
            for i, cand in enumerate(remaining):
                dist = abs(cand - np.conj(r))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx < 0 or best_dist > tol:
                raise ValueError(
                    f"complex root {r} has no conjugate partner within tolerance {tol:.3e}"
                )
            partner = remaining.pop(best_idx)
            # Symmetrise: midpoint with positive imaginary part
            q = (r + np.conj(partner)) / 2.0
            if q.imag < 0:
                q = np.conj(q)
            groups.append([q, np.conj(q)])

    return groups


def _coeffs_from_roots(roots: list[np.complex128]) -> NDArray[np.float64]:
    """Compute monic polynomial coefficients from a list of at most two roots.

    Parameters
    ----------
    roots : list[np.complex128]
        List of 0, 1, or 2 roots.

    Returns
    -------
    NDArray[np.float64]
        Coefficient array of length 3: [c0, c1, c2] for the polynomial
        $c_0 + c_1 z^{-1} + c_2 z^{-2}$ (i.e., the numerator/denominator
        coefficients in SOS form).

    Notes
    -----
    - Empty list: ``[1, 0, 0]``
    - One real root $r$: ``[1, -r.real, 0]``
    - Two roots $r_1, r_2$: ``[1, -(r_1+r_2).real, (r_1*r_2).real]``
    """
    if len(roots) == 0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    elif len(roots) == 1:
        r = roots[0]
        return np.array([1.0, -r.real, 0.0], dtype=np.float64)
    else:
        r1, r2 = roots[0], roots[1]
        c1 = -(r1 + r2).real
        c2 = (r1 * r2).real
        return np.array([1.0, c1, c2], dtype=np.float64)


def _nearest(candidates: list[np.complex128], ref: complex) -> tuple[int, float]:
    """Index of the candidate nearest to ``ref`` and its distance; ``(-1, inf)`` if none."""
    best_i, best_d = -1, math.inf
    for i, c in enumerate(candidates):
        dist = abs(c - ref)
        if dist < best_d:
            best_i, best_d = i, dist
    return best_i, best_d


def zpk2sos(
    z: NDArray[np.complex128],
    p: NDArray[np.complex128],
    k: float,
) -> NDArray[np.float64]:
    r"""Convert zero-pole-gain representation to second-order sections.

    The overall transfer function is realised as a cascade of second-order
    (biquad) sections:

    .. math::

        H(z) = k \prod_{i=1}^{m}
            \frac{b_{0,i} + b_{1,i}\,z^{-1} + b_{2,i}\,z^{-2}}
                 {1 + a_{1,i}\,z^{-1} + a_{2,i}\,z^{-2}},

    where each section is normalised so that its denominator leading coefficient
    equals 1. The gain $k$ is absorbed into the first section's numerator.

    Parameters
    ----------
    z : array_like
        Zeros of the filter.
    p : array_like
        Poles of the filter.
    k : float
        Gain of the filter.

    Returns
    -------
    NDArray[np.float64]
        Second-order sections array of shape ``(m, 6)`` where each row is
        ``[b0, b1, b2, 1.0, a1, a2]``. The number of sections is
        ``ceil(len(p) / 2)`` (or one trivial section when both ``z`` and ``p`` are
        empty).

    Raises
    ------
    ValueError
        If ``z`` or ``p`` is not 1-D, if there are more zeros than poles, if any
        root or the gain is not finite, if the gain is complex, or if a complex
        root has no conjugate partner within tolerance.
    TypeError
        If the gain is not a number.

    Notes
    -----
    Pairing and ordering follow Jackson's rules (Oppenheim and Schafer, 2010,
    Sec. 6.9.3):

    1. **Pole grouping.** Conjugate pole pairs form second-order groups. Real
       poles are sorted by decreasing magnitude; two per group, with the odd
       smallest-magnitude pole forming a first-order group if present.

    2. **One-pass zero assignment.** All pole groups (second-order and first-order
       alike) are served in a single pass in decreasing order of the group radius
       $\max |p|$, i.e. the group closest to the unit circle is served first,
       whatever its kind. For a second-order group the reference pole is the member
       with non-negative imaginary part and the largest magnitude (for two real
       poles, the one with the larger magnitude). The distance from the reference
       pole to the nearest remaining conjugate zero pair (measured to its
       $\operatorname{Im} > 0$ member) is compared with the distance to the nearest
       remaining real zero; the pair is taken when it is nearer or equally near,
       otherwise that real zero is taken together with the next-nearest remaining
       real zero to the same reference pole. Zeros at infinity are padded as zeros
       at the origin first and compete as real zeros. A pair is also taken when the
       remaining zero pairs would otherwise outnumber the second-order groups still
       to be served, so that no complex pair is left for a first-order group. A
       first-order group (a lone real pole) takes the nearest remaining real zero
       and never a complex pair, since a first-order section cannot hold one.
       Groups of equal radius are served in construction order (real-pole groups
       before conjugate pairs), so ties follow the input order.

    3. **Section ordering.** Sections are stable-sorted by increasing group radius
       $\max |p|$, so that the highest-Q (largest-radius) section appears last in
       the cascade. The gain $k$ multiplies the numerator of the first row only, and
       $a_0 = 1$ in every row.

    The cascade realises $z^{d} H(z)$ where $d = \text{len}(p) - \text{len}(z)$ is
    the number of zeros at infinity (padded as zeros at the origin). When both
    ``z`` and ``p`` are empty the result is the single trivial section
    ``[k, 0, 0, 1, 0, 0]``.

    References
    ----------
    A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed.,
    Pearson, 2010, Sec. 6.3.2 (cascade form) and Sec. 6.9.3 (pairing and ordering
    of poles and zeros).
    """
    z = _as_complex_1d(z)
    p = _as_complex_1d(p)

    if len(z) > len(p):
        raise ValueError("need at least as many poles as zeros")

    _check_finite_roots(z, "z")
    _check_finite_roots(p, "p")
    k = _check_finite_gain(k)

    # Empty z and p: the constant transfer function k is one trivial section.
    if len(p) == 0:
        return np.array([[k, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

    # Pad zeros with zeros at the origin so that len(z) == len(p).
    d = len(p) - len(z)
    if d > 0:
        z = np.concatenate((z, np.zeros(d, dtype=np.complex128)))

    # Classify poles into groups.
    pole_groups = _classify_roots(p)

    # Separate conjugate pairs and real poles.
    pole_pairs: list[list[np.complex128]] = []
    real_poles: list[np.complex128] = []
    for g in pole_groups:
        if len(g) == 2:
            pole_pairs.append(g)
        else:
            real_poles.append(g[0])

    # Sort real poles by decreasing magnitude.
    real_poles.sort(key=lambda r: abs(r), reverse=True)

    # Group real poles: two per group, odd smallest forms first-order.
    pole_section_groups: list[list[np.complex128]] = []
    for i in range(0, len(real_poles), 2):
        if i + 1 < len(real_poles):
            pole_section_groups.append([real_poles[i], real_poles[i + 1]])
        else:
            pole_section_groups.append([real_poles[i]])

    # Add conjugate pairs.
    for pair in pole_pairs:
        pole_section_groups.append(pair)

    # Classify zeros into groups.
    zero_groups = _classify_roots(z)

    # Separate zero conjugate pairs and real zeros.
    zero_pairs: list[list[np.complex128]] = []
    real_zeros: list[np.complex128] = []
    for g in zero_groups:
        if len(g) == 2:
            zero_pairs.append(g)
        else:
            real_zeros.append(g[0])

    # Order pole groups by decreasing max|pole| (closest to the unit circle first).
    def _max_pole_mag(group: list[np.complex128]) -> float:
        return max(abs(r) for r in group)

    pole_section_groups.sort(key=_max_pole_mag, reverse=True)

    # Assign zeros to pole groups.
    zero_assignments: list[list[np.complex128]] = [[] for _ in range(len(pole_section_groups))]

    # Track remaining real zeros and complex zero pairs.
    rem_real_zeros = list(real_zeros)
    rem_zero_pairs = list(zero_pairs)

    # Serve every pole group in one pass, closest to the unit circle first. Counting
    # argument (zeros were padded to len(p)): with P free zero pairs, R free real zeros, G
    # second-order groups and L in {0, 1} lone real poles still to serve, R = 2 (G - P) + L,
    # so P <= G. When P == G the real zeros are exactly the ones the lone pole needs and this
    # group must take a pair; otherwise taking two real zeros still leaves R >= L.
    groups_left = sum(1 for g in pole_section_groups if len(g) == 2)
    for idx, grp in enumerate(pole_section_groups):
        if len(grp) == 1:
            # A lone real pole takes the nearest remaining real zero (a first-order
            # section cannot hold a complex pair; the argument above guarantees one).
            i, _ = _nearest(rem_real_zeros, grp[0])
            zero_assignments[idx].append(rem_real_zeros.pop(i))
            continue
        # Second-order group: the reference pole is the member with Im >= 0 and the
        # largest |p| (real poles carry Im = +0.0).
        ref_pole = max((r for r in grp if r.imag >= 0), key=abs)
        real_i, real_d = _nearest(rem_real_zeros, ref_pole)
        pair_i, pair_d = _nearest([max(q, key=lambda r: r.imag) for q in rem_zero_pairs], ref_pole)
        # The nearer zero group wins (the pair on a tie); a pair is forced when the free
        # pairs would otherwise outnumber the second-order groups left.
        use_pair = pair_i >= 0 and (pair_d <= real_d or len(rem_zero_pairs) >= groups_left)
        groups_left -= 1
        if use_pair:
            zero_assignments[idx].extend(rem_zero_pairs.pop(pair_i))
        else:
            zero_assignments[idx].append(rem_real_zeros.pop(real_i))
            # then the next-nearest remaining real zero to the same reference pole
            i, _ = _nearest(rem_real_zeros, ref_pole)
            if i >= 0:
                zero_assignments[idx].append(rem_real_zeros.pop(i))

    # Build sections, stable-sorted by increasing max|pole|.
    order = sorted(
        range(len(pole_section_groups)), key=lambda i: _max_pole_mag(pole_section_groups[i])
    )

    sos = np.empty((len(order), 6), dtype=np.float64)
    for new_i, old_i in enumerate(order):
        pole_grp = pole_section_groups[old_i]
        zero_grp = zero_assignments[old_i]

        z_coeff = _coeffs_from_roots(zero_grp)
        p_coeff = _coeffs_from_roots(pole_grp)

        sos[new_i, 0:3] = z_coeff
        sos[new_i, 3] = 1.0
        sos[new_i, 4:6] = p_coeff[1:3]

    # Absorb the overall gain into the first section's numerator.
    sos[0, 0:3] *= k

    return sos


def sos_response(
    sos: ArrayLike,
    n: int = 512,
    fs: float | None = None,
    whole: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    r"""Compute the frequency response of a second-order-section filter.

    The overall transfer function is the product of the per-section responses,

    .. math::

        H(e^{j\omega}) = \prod_{i=1}^{m}
            \frac{b_{0,i} + b_{1,i}\,e^{-j\omega} + b_{2,i}\,e^{-2j\omega}}
                 {1 + a_{1,i}\,e^{-j\omega} + a_{2,i}\,e^{-2j\omega}},

    where each section is normalised so that its denominator leading coefficient
    (the $a_{0,i}$ column) equals 1.

    Parameters
    ----------
    sos : array_like
        Second-order sections array of shape ``(m, 6)``, where each row is
        ``[b0, b1, b2, 1.0, a1, a2]``. The fourth column must be exactly 1.0
        (normalised sections).
    n : int, optional
        Number of frequency points to compute. Must be at least 1. Default is 512.
    fs : float or None, optional
        Sampling frequency in Hz. If ``None``, uses $2\pi$ (normalized). Must be
        strictly positive and finite when given.
    whole : bool, optional
        If True, compute over the full range $[0, 2\pi)$; otherwise $[0, \pi)$.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.complex128]]
        f : NDArray[np.float64]
            Frequency values in Hz (or normalized if ``fs`` is None), shape ``(n,)``.
        h : NDArray[np.complex128]
            Complex frequency response at each frequency point, shape ``(n,)``.

    Raises
    ------
    ValueError
        If ``sos`` has a complex dtype, does not have shape ``(m, 6)``, contains a
        section whose fourth column is not exactly 1.0, if ``n < 1``, or if ``fs``
        is given but not strictly positive and finite (including zero, negative,
        NaN and infinite values).

    References
    ----------
    A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed.,
    Pearson, 2010, Sec. 6.3.2 (cascade form: the response is the product of the
    second-order section responses).
    """
    sos_arr = np.asarray(sos, dtype=np.complex128)

    if np.iscomplexobj(sos):
        raise ValueError("sos must be real-valued; complex sections are not supported")
    if sos_arr.ndim != 2 or sos_arr.shape[1] != 6:
        raise ValueError(f"sos must have shape (m, 6), got {sos_arr.shape}")
    if not np.all(sos_arr[:, 3] == 1.0):
        raise ValueError("every section's fourth column (a0) must be exactly 1.0")

    n = int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    if fs is not None and not (np.isfinite(fs) and float(fs) > 0.0):
        raise ValueError(f"fs must be strictly positive and finite when given, got {fs!r}")

    if fs is None:
        fs = 2.0 * np.pi

    if whole:
        w = 2.0 * np.pi * np.arange(n) / n
    else:
        w = np.pi * np.arange(n) / n

    f = w * fs / (2.0 * np.pi)

    # Compute exp(-1j*w) and exp(-2j*w)
    e1 = np.exp(-1j * w)
    e2 = e1 * e1

    # Initialize response as ones
    h = np.ones(n, dtype=np.complex128)

    # Multiply by each section's transfer function
    for i in range(sos_arr.shape[0]):
        b0, b1, b2, _, a1, a2 = sos_arr[i]
        numerator = b0 + b1 * e1 + b2 * e2
        denominator = 1.0 + a1 * e1 + a2 * e2
        h *= numerator / denominator

    return f.astype(np.float64), h.astype(np.complex128)
