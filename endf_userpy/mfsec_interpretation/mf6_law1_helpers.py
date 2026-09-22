"""Physics helpers shared by the MF6 LAW=1 discrete-lines port and
the (upcoming) LAW=1 continuum port.

Contents mirror the Fortran routines in ``endf_userpy/fortran/endf6.f90``:

- :func:`mf6cm2lab_disc` -- inverse CM->LAB map for discrete-line
  configurations (Fortran ``mf6cm2lab_disc``, line 259). Given a
  fixed eval-frame outgoing energy ``tp`` and a user LAB cosine
  ``u``, solve for the LAB outgoing energy ``ep`` at which the
  peak is observed, the corresponding eval-frame cosine ``w``, and
  the eval->LAB Jacobian ``dinv``. Returns ``dinv = 0`` on the
  "no physical solution" branch (below threshold, or discriminant
  negative).
- :func:`bachaa` -- Kalbach-Mann slope parameter ``a(E, E')``
  (Fortran ``bachaa``, line 1501). Semi-empirical fit adapted from
  NJOY2016 by D. Lopez Aldama.
- :func:`ykalbach` -- Kalbach-Mann angular distribution ``f(u)``
  (Fortran ``ykalbach``, line 1448).
- :func:`f6law1_dis_amplitude` -- angular-weighted amplitude for a
  single discrete line at a single (E, tp, w), dispatching on
  LANG (Fortran ``f6law1_dis``, line 712). LANG=1 uses the
  Legendre recurrence from ``primitives.interpolation``; LANG=2
  uses Kalbach-Mann; LANG=11..15 uses tabulated
  ``interp_tab1``.

All routines are scalar in the arguments that scan the physical
axes (E, tp, w, u). The caller loops over incident energies and
mu queries in Python; vectorising per query is defensive but the
discrete-line grid is inherently small (typically a few lines
per subsection) and Python-loop overhead is not a hot spot.
"""
from __future__ import annotations

import numpy as np


# --- CM<->LAB inverse for discrete-line kinematics -----------------

_MF6CM_D2_MIN = 1.0e-38
_MF6CM_C_MIN = 1.0e-19


def mf6cm2lab_disc(awr: float, awi: float, awp: float, lct: int,
                    e: float, tp: float, u: float):
    """Inverse of ``mf6lab2cm`` for a discrete-line configuration.

    Given a fixed eval-frame outgoing energy ``tp`` (a discrete
    peak position in the file's evaluation frame) and a user LAB
    cosine ``u``, solve for the LAB outgoing energy ``ep`` at which
    the peak is observed, the corresponding eval-frame cosine
    ``w``, and the eval->LAB Jacobian ``dinv`` such that
    ``density_LAB(ep, u) = density_eval(tp, w) * dinv``.

    Returns ``(ep, w, dinv)``. ``dinv = 0.0`` is the sentinel for
    "no physical solution" (below kinematic threshold, or the
    discriminant of the CM->LAB quadratic is negative).

    LCT=1 (LAB) and LCT=3 with light ejectile (``awp < 4``) go
    through the identity map. LCT=2 and LCT=3 with heavy ejectile
    solve the classical CM<->LAB quadratic

        y^2 - 2 c0 sqrt(e) u y + (c0^2 e - tp) = 0,  y = sqrt(ep)

    with ``c0 = sqrt(awi awp)/(awi + awr)``. The forward-branch
    root (+) is chosen when positive; if not, the (-) root is tried
    as a fallback; if neither is positive, ``dinv = 0`` is
    returned.

    Direct transliteration of the Fortran reference (endf6.f90
    line 259).
    """
    # Note: the Fortran condition is
    #   (lct.eq.2.or.(lct.eq.3.and.awp.lt.4)) and e*tp > 0
    if not ((lct == 2 or (lct == 3 and awp < 4.0)) and e * tp > 0.0):
        return tp, u, 1.0

    c0 = np.sqrt(awi * awp) / (awi + awr)
    bcoef = c0 * np.sqrt(e) * u
    ccoef = c0 * c0 * e - tp
    disc = bcoef * bcoef - ccoef
    if disc < 0.0:
        return 0.0, 0.0, 0.0
    root = np.sqrt(disc)
    y = bcoef + root
    if y <= 0.0:
        y = bcoef - root
        if y <= 0.0:
            return 0.0, 0.0, 0.0
    ep = y * y
    # Recover w and dinv via the forward LAB->eval formulas.
    c = c0 * np.sqrt(e / ep)
    d2 = 1.0 + c * c - 2.0 * c * u
    if d2 < _MF6CM_D2_MIN:
        d2 = _MF6CM_D2_MIN
        c = u - _MF6CM_C_MIN
    dinv = 1.0 / np.sqrt(d2)
    w = dinv * (u - c)
    if w > 1.0:
        w = 1.0
    elif w < -1.0:
        w = -1.0
    return ep, w, dinv


# --- Kalbach-Mann angular distribution -----------------------------

# Isotope-averaged natural-A fallbacks for the semi-empirical
# binding-energy formula (elements without a single dominant
# isotope: 6-C, 12-Mg, 14-Si, ...). Direct copy of the Fortran
# ``iza`` table at endf6.f90:1538-1560.
_BACHAA_ISOTOPE_TABLE = {
    6000: 6012,  12000: 12024, 14000: 14028, 16000: 16032,
    17000: 17035, 19000: 19039, 20000: 20040, 22000: 22048,
    23000: 23051, 24000: 24052, 26000: 26056, 28000: 28058,
    29000: 29063, 31000: 31069, 40000: 40090, 42000: 42096,
    48000: 48112, 49000: 49115, 50000: 50120, 63000: 63151,
    72000: 72178, 74000: 74184, 82000: 82208,
}

_BACHAA_EPS = 1.0e-3
_BACHAA_EMC2 = 939.56542052539
_BACHAA_EMEV = 1.0e6


def bachaa(zai: float, zap: float, zat: float, ee: float, epe: float) -> float:
    """Kalbach-Mann parameter ``a = a(E, E')`` (endf6.f90 line
    1501; adapted from NJOY2016).

    All energies in eV. Returns the Kalbach-Mann slope parameter
    ``a`` for the angular distribution
    ``f(u) = a f0 (cosh(a u) + r sinh(a u)) / (2 sinh(a))``.

    ZA convention: ``za = 1000 * Z + A``. Elements written with
    A=0 (natural isotopic mix) are mapped to a dominant-isotope A
    via the ``_BACHAA_ISOTOPE_TABLE``.
    """
    iza1i = int(zai + _BACHAA_EPS)
    iza2 = int(zap + _BACHAA_EPS)
    izat = int(zat + _BACHAA_EPS)
    e = ee / _BACHAA_EMEV
    ep = epe / _BACHAA_EMEV
    iza1 = iza1i if iza1i != 0 else 1
    iza = _BACHAA_ISOTOPE_TABLE.get(izat, izat)
    aa = iza % 1000
    if aa == 0:
        raise ValueError(
            f'bachaa: dominant isotope not known for ZA={iza}'
        )
    za = iza // 1000
    ac = aa + (iza1 % 1000)
    zc = za + (iza1 // 1000)
    ab = ac - (iza2 % 1000)
    zb = zc - (iza2 // 1000)
    na = aa - za
    nb = ab - zb
    nc = ac - zc

    third = 1.0 / 3.0
    twoth = 2.0 / 3.0
    fourth = 4.0 / 3.0
    c1, c2, c3 = 15.68, -28.07, -18.56
    c4, c5, c6 = 33.22, -0.717, 1.211
    s2, s3, s4, s5 = 2.22, 8.48, 7.72, 28.3
    b1, b2, b3 = 0.04, 1.8e-6, 6.7e-7
    d1 = 9.3
    ea1, ea2 = 41.0, 130.0

    sa = (c1 * (ac - aa)
          + c2 * ((nc - zc) ** 2 / ac - (na - za) ** 2 / aa)
          + c3 * (ac ** twoth - aa ** twoth)
          + c4 * ((nc - zc) ** 2 / ac ** fourth
                  - (na - za) ** 2 / aa ** fourth)
          + c5 * (zc ** 2 / ac ** third - za ** 2 / aa ** third)
          + c6 * (zc ** 2 / ac - za ** 2 / aa))
    if iza1 == 1002:
        sa -= s2
    elif iza1 == 1003:
        sa -= s3
    elif iza1 == 2003:
        sa -= s4
    elif iza1 == 2004:
        sa -= s5

    sb = (c1 * (ac - ab)
          + c2 * ((nc - zc) ** 2 / ac - (nb - zb) ** 2 / ab)
          + c3 * (ac ** twoth - ab ** twoth)
          + c4 * ((nc - zc) ** 2 / ac ** fourth
                  - (nb - zb) ** 2 / ab ** fourth)
          + c5 * (zc ** 2 / ac ** third - zb ** 2 / ab ** third)
          + c6 * (zc ** 2 / ac - zb ** 2 / ab))
    if iza2 == 1002:
        sb -= s2
    elif iza2 == 1003:
        sb -= s3
    elif iza2 == 2003:
        sb -= s4
    elif iza2 == 2004:
        sb -= s5

    ecm = aa * e / ac
    ea = ecm + sa
    eb = ep * ac / ab + sb
    x1 = eb if ea <= ea2 else ea2 * eb / ea
    x3 = eb if ea <= ea1 else ea1 * eb / ea
    fa = 0.0 if iza1 == 2004 else 1.0
    if iza2 == 1:
        fb = 0.5
    elif iza2 == 2004:
        fb = 2.0
    else:
        fb = 1.0
    bb = b1 * x1 + b2 * x1 ** 3 + b3 * fa * fb * x3 ** 4
    if iza1i == 0:
        # Incident photon: extra factor
        fact = d1 if ep == 0.0 else d1 / np.sqrt(ep)
        fact = max(fact, 1.0)
        fact = min(fact, 4.0)
        bb = bb * np.sqrt(e / (2.0 * _BACHAA_EMC2)) * fact
    return bb


_KALBACH_AMIN = 1.0e-38


def ykalbach(zai: float, zap: float, zat: float,
              e: float, ep: float, u: float,
              b: np.ndarray, na: int) -> float:
    """Kalbach-Mann angular distribution at ``(E, E', u)`` (endf6.f90
    line 1448).

    ``f(u) = a f0 (cosh(a u) + r sinh(a u)) / (2 sinh(a))``

    with ``f0 = b[0]``, ``r = b[1]`` (for na >= 1), and either
    ``a = bachaa(...)`` (na=1) or ``a = b[2]`` (na=2). For na=0
    both r and a are zero and the distribution collapses to the
    constant ``f0 / 2``.
    """
    f0 = float(b[0])
    if na == 1:
        r = float(b[1])
        a = bachaa(zai, zap, zat, e, ep)
    elif na == 2:
        r = float(b[1])
        a = float(b[2])
    else:
        return 0.5 * f0

    if abs(a) > _KALBACH_AMIN:
        au = a * u
        return 0.5 * a * f0 * (np.cosh(au) + r * np.sinh(au)) / np.sinh(a)
    return 0.5 * f0


# --- Amplitude dispatcher for one discrete line --------------------


def f6law1_dis_amplitude(e: float, tp: float, w: float,
                          za: float, zai: float, zap: float,
                          lang: int, nd: int, na: int,
                          ep_panel: np.ndarray,
                          b_panel: np.ndarray) -> float:
    """Angular-weighted amplitude of one discrete line at ``(E, tp, w)``
    on one incident-energy panel (endf6.f90 line 712).

    ``ep_panel`` (shape ``(nep_total,)``) and ``b_panel``
    (shape ``(nep_total, na+1)``) are the panel's full tabulated
    outgoing-energy mesh and angular parameters. ``nd`` is the
    discrete-lines count; only the first ``nd`` entries of
    ``ep_panel`` are eligible.

    Dispatches on ``lang``:
    - 1: Legendre (see :func:`_legendre_eval_at_mu`)
    - 2: Kalbach-Mann (see :func:`ykalbach`)
    - 11-15: tabulated (u, p(u)) pairs

    Returns 0.0 when the query ``tp`` does not match any of the
    first ``nd`` entries of ``ep_panel``. This mirrors the Fortran
    ``imatch`` semantics: the caller is responsible for making sure
    ``tp`` equals one of the tabulated discrete Ep values (usually
    trivially true because ``tp`` came from
    ``yintp(e1, ep1[k], e2, ep2[k], law, e)`` where ``ep1[k] ==
    ep2[k]``).
    """
    # imatch: return the index of the first ep_panel[i] == tp
    # within the discrete slots [0, nd).
    idx = _imatch(tp, ep_panel, nd)
    if idx < 0:
        return 0.0

    if lang in (1, 2):
        nt = na + 1
        a_coeffs = np.asarray(b_panel[idx, :nt], dtype=float)
        if lang == 1:
            return _legendre_eval_at_mu(w, a_coeffs, na)
        return ykalbach(zai, zap, za, e, tp, w, a_coeffs, na)
    if 11 <= lang <= 15:
        f0 = float(b_panel[idx, 0])
        if na > 0:
            nmu = na // 2
            mu_arr = np.empty(nmu, dtype=float)
            y_arr = np.empty(nmu, dtype=float)
            k = 1
            for j in range(nmu):
                mu_arr[j] = float(b_panel[idx, k])
                k += 1
                y_arr[j] = 2.0 * f0 * float(b_panel[idx, k])
                k += 1
            lmu = lang - 10
            from ..primitives.interpolation import interp_tab1
            tab = {
                'mu': mu_arr, 'f': y_arr,
                'INT': np.array([lmu], dtype=int),
                'NBT': np.array([nmu], dtype=int),
            }
            return float(interp_tab1(np.array([w]), tab, 'mu', 'f')[0])
        return 0.5 * f0
    raise NotImplementedError(f'MF6 LAW=1 LANG={lang} not supported')


def _imatch(x0: float, x: np.ndarray, n: int) -> int:
    """Return the first index ``i`` in ``[0, n)`` with ``x[i] == x0``,
    or -1 if none. Mirrors the Fortran ``imatch`` semantics used by
    ``f6law1_dis`` (endf6.f90 line 1981).
    """
    for i in range(min(n, x.shape[0])):
        if float(x[i]) == float(x0):
            return i
    return -1


def _legendre_eval_at_mu(mu: float, a_from_1: np.ndarray, na: int) -> float:
    """Legendre expansion evaluation matching Fortran ``yleg``
    (endf6.f90 line 1394): ``sum_{L=0..na} (L+0.5) a_L P_L(mu)``,
    with ``a_0 == 1`` prepended and ``a_1, ..., a_na`` supplied in
    ``a_from_1``.

    ``a_from_1`` has shape ``(na+1,)`` where the first entry is
    ``a_0`` from the file (in the discrete-line context this comes
    from the tabulated ``b`` row; the Fortran wrapper picks
    ``b(i, j)`` for ``j = 1..nt`` where ``nt = na + 1``). We follow
    that same convention: pass the full ``na+1``-entry row.

    Direct closed-form Bonnet recurrence, matching
    :func:`_eval_legendre_series` in
    :mod:`primitives.interpolation` but at a scalar ``mu``.
    """
    n = na + 1
    if n == 0:
        return 0.0
    # Bonnet recurrence up to L = na
    p_prev = 1.0                # P_0
    total = (0 + 0.5) * float(a_from_1[0]) * p_prev
    if n == 1:
        return total
    p_curr = float(mu)          # P_1
    total += (1 + 0.5) * float(a_from_1[1]) * p_curr
    for L in range(1, n - 1):
        p_next = ((2 * L + 1) * mu * p_curr - L * p_prev) / (L + 1)
        total += (L + 1 + 0.5) * float(a_from_1[L + 1]) * p_next
        p_prev = p_curr
        p_curr = p_next
    return total
