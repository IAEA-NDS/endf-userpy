#===================================================================================
# Resonance Formalism Calculation
# Authors: Daniel Lopez Aldama & Georg Schnabel
# Date: 21.03.2026
# Ver: 1.0.0
#===================================================================================

import os
import jax
jax.config.update("jax_enable_x64", True)

# Persist compiled XLA executables across Python sessions.
# Override by setting the JAX_CACHE_DIR environment variable.
_jax_cache_dir = os.environ.get(
    "JAX_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "jax_compilation_cache")
)
os.makedirs(_jax_cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _jax_cache_dir)

import jax.numpy as jnp
from jax import lax
from jax import tree_util
import flax
import numpy as np

#===================================================================================
# Configurable data for resonance calculation applying the jax package
#===================================================================================

#-----------------------------------------------------------------------------------
# Resonance data configurable parameters
#-----------------------------------------------------------------------------------

MAX_ISO = 1                      # Max. isotopes per material (typical: NIS=1)
MAX_RANGES = 2                   # Max. energy ranges (typical NER=2, 1 RRR + 1 URR)
N_TOTAL = MAX_ISO * MAX_RANGES   # Max. number of isotope/energy ranges
NLMAX = 8                        # Max. number of L-values (0,1,2, ... NLMAX-1)
NRS_SIZE = 2048                  # Default fix blocksize for resonance data
ER_VAL = 1.0e-12                 # Er-value used for padded and missed channels

#----------------------------------------------------------------------------------
# Channel configurable parameters
# CH_KEY = CHN_STRIDE * L + CHN_OFFSET + 2 * J
#----------------------------------------------------------------------------------

MAX_CHN = 32                     # Maximum number of channels
CHN_STRIDE = 1000                # Channel key stride (even number)
CHN_OFFSET = CHN_STRIDE / 2      # Channel key offset
MISSED_2J = CHN_OFFSET - 1       # Value of 2*J assigned to a missed J-channel
CHN_PADDED = 99999               # Key value assigned to padded channels

#----------------------------------------------------------------------------------
# TAB1 configurable parameters
#----------------------------------------------------------------------------------

NP_TAB1 = 2048                   # Default size of NP for a padded TAB1 record
NR_TAB1 = 20                     # Default size of NR for a padded TAB1 record

#=====================================================================================
# Physical constants from ENDF-6 Formats Manual, Appendix H.
#=====================================================================================

#-------------------------------------------------------------------------------------
# Fundamental physical constants
#-------------------------------------------------------------------------------------

mn      = 1.008664915950         # neutron mass [uma]
ev_amu  = 9.3149410242e+8        # Energy equivalence of the atomic mass unit [eV/amu]
fm      = 1.0e-12                # fm to cm conversion factor (1 fm = 1.0e-12 cm)
h_bar   = 6.582119569e-16        # Reduced Planck's constant = h/(2*pi) [eV.s]
c_light = 2.99792458e+10         # Speed of light in vacuum [cm/s]
particle_data = {                # Incident particle mass and spin: tuple(mass, spin)
       1: (mn,             0.5), # Neutron
    1001: (1.007276466621, 0.5), # Proton
    1002: (2.013553212745, 1.0), # Deuteron
    1003: (3.015500716210, 0.5), # Triton
    2003: (3.014932247175, 0.5), # Helion
    2004: (4.001506179127, 0.0), # Alpha
       0: (0.000000000000, 1.0)  # Photon
}

#----------------------------------------------------------------------------------
# Derived constant for computing the k-wave number k(E):
# kn   = sqrt(2 * mn) / hbar
# ki   = kn * sqrt(awi) * awri / (awri + awi)
# k(E) = ki * sqrt(E)
#----------------------------------------------------------------------------------

kn = jnp.sqrt(2.0 * mn * ev_amu) * fm / (h_bar * c_light)

#==================================================================================
# Dataclasses
#==================================================================================

@flax.struct.dataclass
class Padded_TAB1:
    x:    jnp.array
    y:    jnp.array
    nbt:  jnp.array
    intp: jnp.array
    np:   jnp.array

@flax.struct.dataclass
class BKG_DataClass:
    tot: Padded_TAB1
    sct: Padded_TAB1
    cap: Padded_TAB1
    fis: Padded_TAB1

@flax.struct.dataclass
class Dummy_DataClass:
    ap : jnp.array

@flax.struct.dataclass
class Special_DataClass:
    abn : jnp.array
    ap  : jnp.array

@flax.struct.dataclass
class BW_DataClass:
    abn : jnp.array
    spi : jnp.array
    ki  : jnp.array
    qx  : jnp.array
    r_a : Padded_TAB1
    r_ap: Padded_TAB1
    nch : jnp.array
    ch  : jnp.array
    gch : jnp.array
    ich : jnp.array
    lr  : jnp.array
    er  : jnp.array
    gn  : jnp.array
    gg  : jnp.array
    gf  : jnp.array
    gx  : jnp.array

@flax.struct.dataclass
class RM_DataClass: # TODO
    abn : jnp.array

@flax.struct.dataclass
class AA_DataClass: # TODO
    abn : jnp.array

@flax.struct.dataclass #TODO
class RML_DataClass:
    abn : jnp.array

@flax.struct.dataclass #TODO
class URR_DataClass:
    abn : jnp.array

#==================================================================================
#  Compute total spin J and number of sequences for each J
#==================================================================================

def get_J_spins(I, s, l):
    """
    Compute the list of possible J-values for given I, s and l
    and the number of sequences for each J.
    Input parameters:
        I : targen spin
        s : incident particle intrinsic spin
        l : orbital moment number
    Return:
        J_spins : array of J values
        J_seq: number of sequences for each J-values
        nJ_tot: total number of sequences
    """

    #---------
    # J range
    #---------

    Jmin = round(2 * abs(abs(I - l) - s)) / 2
    Jmax = round(2 * (I + l + s)) / 2

    nJ = int(round(2 * (Jmax - Jmin)) / 2) + 1
    J_spins = Jmin + np.arange(nJ)

    #---------
    # j range
    #---------

    jmin = round(2 * abs(I - s)) / 2
    jmax = round(2 * (I + s)) / 2
    mj = int(round(2 * (jmax - jmin)) / 2) + 1

    #--------------------------------------------
    # count allowed j for each J (triangle rule)
    #--------------------------------------------

    low  = np.maximum(jmin, np.abs(J_spins - l))
    high = np.minimum(jmax, J_spins + l)

    J_seq = np.maximum(0, high - low + 1).astype(np.int32)

    nJ_tot = J_seq.sum()

    return nJ_tot, J_spins, J_seq

#==================================================================================
# TAB1 conversion to jax Padded TAB1
#==================================================================================

def pad_tab1(x_endf, y_endf, nbt_endf, intp_endf, NP_BLOCK=NP_TAB1, NR_BLOCK=NR_TAB1):
    """
    Convert an ENDF-6/TAB1 to record using jax arrays
    Input:
        x_endf, y_endf, nbt_endf, intp_endf : TAB1 data
    Return:
        Padded_TAB1 : padded TAB1 data (x, y, nbt, intp)
        np : Number of energy points (length of x_endf)
    """

    x_jax = jnp.array(x_endf, dtype=jnp.float64)
    y_jax = jnp.array(y_endf, dtype=jnp.float64)
    nbt_jax = jnp.array(nbt_endf, dtype=jnp.int32) - 1
    intp_jax = jnp.array(intp_endf, dtype=jnp.int32)
    np = jnp.array(x_jax.shape[0], dtype=jnp.int32)
    nr = nbt_jax.shape[0]
    nbt_jax = nbt_jax.at[-1].set(np-1)


    npad = jnp.maximum(NP_BLOCK, np) - np
    x_pad = jnp.concatenate([x_jax, jnp.full((npad,), x_jax[-1])])
    y_pad = jnp.concatenate([y_jax, jnp.full((npad,), y_jax[-1])])

    npad = jnp.maximum(NR_BLOCK, nr) - nr
    nbt_pad = jnp.concatenate([nbt_jax, jnp.full((npad,), nbt_jax[-1])])
    intp_pad = jnp.concatenate([intp_jax, jnp.full((npad,), intp_jax[-1])])

    return Padded_TAB1(x_pad, y_pad, nbt_pad, intp_pad, np)

#===================================================================================
# Segment scalar interpolation
#===================================================================================

def segment_intp_scalar(ilaw, x, x1, x2, y1, y2):
    """
    Interpolation of y(x) in (x1, x2] according to
    to the ENDF-6 interpolation laws (ilaw)
    Input:
        ilaw : ENDF_interpolation laws
        x  : value of the abscissa
        x1 : lower interval boundary
        x2 : upper interval boundary
        y1 : y(x1)
        y2 : y(x2)
    Return:
        y(x)
    Handles:
    - discontinuities: X[i] == X[i-1] (ilaw == 1)
    - constant values: Y[i] == Y[i-1] (ilaw == 1)
    - out-of-range values (ilaw == 0)
    - ENDF-6 interpolation laws: (ilaw == 1|2|3|4|5)
      1: const, 2: lin-lin, 3: lin-log, 4: log-lin, 5: log-log
    """

    EPS = jnp.array(1.0e-30, dtype=jnp.float64)

    def outlier(_):
        return 0.0

    def const(_):
        return y1

    def linlin(_):
        return y1 + (x - x1)*(y2 - y1)/(x2 - x1)

    def linlog(_):
        x1s = jnp.where(x1==0.0, EPS*x2, x1)
        return y1 + jnp.log(x/x1s) * (y2 - y1) / jnp.log(x2/x1s)

    def loglin(_):
        y1s = jnp.where(y1==0.0, EPS*y2, y1)
        return y1s * jnp.exp((x-x1) * jnp.log(y2/y1s) / (x2-x1))

    def loglog(_):
        x1s = jnp.where(x1==0.0, EPS*x2, x1)
        y1s = jnp.where(y1==0.0, EPS*y2, y1)
        return (
            y1s * jnp.exp(jnp.log(x/x1s) * jnp.log(y2/y1s) / jnp.log(x2/x1s))
        )

    intp_laws = [outlier, const, linlin, linlog, loglin, loglog]

    return lax.switch(ilaw, intp_laws, operand=None)

#===================================================================================
# Vectorization of segment_intp_scalar in all axes (jit version)
# ilaw, x, x1, x2, y1, y2 --> vectors
#===================================================================================

segment_intp_vec = jax.vmap(segment_intp_scalar, in_axes=(0,0,0,0,0,0))

#===================================================================================
# TAB1 interpolation of a vector x_vec
#===================================================================================

def tab1_intp_vec(x_vec, tab1):
    """
    Evaluation of a padded TAB1 record at x_vec,
    fully safe and JAX compatible,
    using the ENDF-6 interpolation laws:
        Y-const: ilaw == 1
        lin-lin: ilaw == 2
        lin-log: ilaw == 3
        log-lin: ilaw == 4
        log-log: ilaw == 5
        discontinuity or constant value: ilaw == 1
        outliers: ilaw == 0
    Input:
        x_vec: x vector
        tab1: Padded_TAB1 record
    Return:
        y_vec: y(x_vect)
    """

    i = jnp.clip(jnp.searchsorted(tab1.x, x_vec, side="left"), 1, tab1.np - 1)
    k = jnp.searchsorted(tab1.nbt, i, side="left")
    x1 = tab1.x[i-1]
    y1 = tab1.y[i-1]
    x2 = tab1.x[i]
    y2 = tab1.y[i]

    ilaw = tab1.intp[k] % 10
    ilaw = jnp.where((x1 == x2) | (y1 == y2), 1, ilaw)
    ilaw = jnp.where((x_vec < x1) | (x_vec > x2), 0, ilaw)

    y_vec = segment_intp_vec(ilaw, x_vec, x1, x2, y1, y2)

    return y_vec

def tab1_intp_scalar(x, tab1):
    """Evaluate a padded TAB1 record at scalar x."""
    i = jnp.clip(jnp.searchsorted(tab1.x, x, side="left"), 1, tab1.np - 1)
    k = jnp.searchsorted(tab1.nbt, i, side="left")
    x1, x2 = tab1.x[i-1], tab1.x[i]
    y1, y2 = tab1.y[i-1], tab1.y[i]
    ilaw = tab1.intp[k] % 10
    ilaw = jnp.where((x1 == x2) | (y1 == y2), 1, ilaw)
    ilaw = jnp.where((x < x1) | (x > x2), 0, ilaw)
    return segment_intp_scalar(ilaw, x, x1, x2, y1, y2)

#======================================================================
# Evaluate background cross sections at a single scalar energy e
#======================================================================

def bkg_xs_scalar(e, bkg_data):
    """
    Evaluate MF3 background cross sections at scalar incident energy e.
    Returns scalars: tot, sct, cap, fis
    """
    tot = tab1_intp_scalar(e, bkg_data.tot)
    sct = tab1_intp_scalar(e, bkg_data.sct)
    cap = tab1_intp_scalar(e, bkg_data.cap)
    fis = tab1_intp_scalar(e, bkg_data.fis)
    return tot, sct, cap, fis

#==================================================================================
# Compute rho = k(e) * r(e) for a given radius TAB1 record r
# Covers both scattering radius (r_ap) and channel radius (r_a)
#==================================================================================

def rho(e, ki, r):
    """
    Compute rho = k(|e|) * r(|e|)
    Input parameters:
        e: incident or resonance energy vector (NE,)
        ki = kn * sqrt(awi) * awri / (awri + awi)
        r: radius TAB1 record (channel or scattering)
    Return:
        rho (NE,)
    """

    ee = jnp.abs(e)
    return ki * jnp.sqrt(ee) * tab1_intp_vec(ee, r)

def rho_scalar(e, ki, r):
    """
    Compute rho = k(|e|) * r(|e|)
    Input parameters:
        e: incident energy (scalar)
        ki = kn * sqrt(awi) * awri / (awri + awi)
        r: radius TAB1 record (channel or scattering)
    Return:
        rho (scalar)
    """

    ee = jnp.abs(e)
    return ki * jnp.sqrt(ee) * tab1_intp_scalar(ee, r)

#==================================================================================
# Compute the channel radius at energies e and er for the competitive width gx
#==================================================================================

def rhox_chn_res(er, ki, r_a, qx):
    """
    Compute channel rho for the competitive width
    Input parameters:
        er: resonance energy vector (NE,)
        ki = kn * sqrt(awi) * awri / (awri + awi)
        r_a: channel radius (TAB1 record)
        qx: Q-value of the competitive width
    Return:
        rhox (NE,)
    """

    ee = jnp.abs(er + qx)
    return ki * jnp.sqrt(ee) * tab1_intp_vec(ee, r_a)

def rhox_chn_scalar(e, ki, r_a, qx):
    """
    Compute channel rho for the competitive width
    Input parameters:
        e: incident energy
        ki = kn * sqrt(awi) * awri / (awri + awi)
        r_a: channel radius (TAB1 record)
        qx: Q-value of the competitive width
    Return:
        rhox (scalar)
    """

    ee = jnp.clip(e + qx, 0.0)
    return ki * jnp.sqrt(ee) * tab1_intp_scalar(ee, r_a)

#==================================================================================
#  Compute de l-values for the competitive width (approximation for lx)
#==================================================================================

def lx_values(l_vec, spi):
    """
    Compute the l-value of the competitive width
    Input parameters:
        l_vec : l-values (vector)
        spi: target spin
    Return:
        lx (vector)
    """

    return l_vec + (jnp.abs(l_vec - 2) - l_vec) * (spi == 0)

# =================================================================================
# Scalar version of penetration and shift Factors (PNT,SHF)
# =================================================================================

def pnt_shf_scalar(rho, L):
    """
    Compute the penetration factor for L
    Input:
        rho : k(E)*a(E) where a(E) = channel radius
        L : angular moment order (scalar)
    Return:
      pnt : penetration factor
    """

    r2 = rho * rho

    def f0():
        return rho, jnp.array(0.0, dtype=jnp.float64)

    def f1():
        den = 1.0 + r2
        return rho * r2 / den, -1.0 / den

    def f2():
        den = 9.0 + r2 * (3.0 + r2)
        return rho * r2**2 / den, -(18.0 + 3.0 * r2) / den

    def f3():
        den = 225.0 + r2 * (45.0 + r2 * (6.0 + r2))
        return rho * r2**3 / den, -(675.0 + r2 * (90.0 + 6.0 * r2)) / den

    def f4():
        den = 11025.0 + r2 * (1575.0 + r2 * (135.0 + r2 * (10.0 + r2)))
        return rho * r2**4 / den, -(44100.0 + r2 * (4725.0 + r2 * (270.0 + 10.0 * r2))) / den

    def f5():
        den = 893025.0 + r2 * (
            99225.0 + r2 * (6300.0 + r2 * (315.0 + r2 * (15.0 + r2)))
        )
        return (
            rho * r2**5 / den,
            -(4465125.0 + r2 * (396900.0 + r2 * (18900.0 + r2 * (630.0 + 15.0 * r2)))) / den
        )

    def flow(L):
        return lax.switch(L, (f0, f1, f2, f3, f4, f5))

    def fhigh(L):

        p_prev, s_prev = f5()

        def step(carry, ll):

            p_prev, s_prev = carry

            sdif = ll - s_prev
            ratio = r2 / (sdif * sdif + p_prev * p_prev)

            p_new = ratio * p_prev
            s_new = ratio * sdif - ll

            update = ll <= L

            p_out = jnp.where(update, p_new, p_prev)
            s_out = jnp.where(update, s_new, s_prev)

            return (p_out, s_out), None

        (p_final, s_final), _ = lax.scan(
            step,
            (p_prev, s_prev),
            jnp.arange(6, NLMAX)
        )

        return p_final, s_final

    return lax.cond(L < 6, flow, fhigh, L)

#===================================================================================
# Vectorization 1 to 1 (rho(Er), Lr) --> pnt(NR,), shf(NR,)
#===================================================================================

def get_pnt_shf_vec(rho_r, L_r):
    """
    Compute pnt y shf factor for the tupla (rho_er[i], Lr[i])
    Inputs:
        rho_r : vector jnp.float64, shape (NR,)
        L_r : vector jnp.int32, shape (NR,)
    Return:
        pnt : vector (NR,)
        shf : vector (NR,)
    """

    # vmap 1 to 1
    vmapped = jax.vmap(pnt_shf_scalar, in_axes=(0,0))

    return vmapped(rho_r, L_r)

#===================================================================================
# pnt_scalar: penetration factor only (used for the competitive width gx)
# Delegates to pnt_shf_scalar and discards the shift factor; JAX DCE removes it.
#===================================================================================

def pnt_scalar(rho, L):
    return pnt_shf_scalar(rho, L)[0]

def get_pnt_vec(rho_r, L_r):
    return get_pnt_shf_vec(rho_r, L_r)[0]

#===================================================================================
# Potential phase Shift (PHS)
#===================================================================================

def phs_scalar(rho, L):
    """
    Compute the potential phase shift factor for L
    Input:
        rho : k(E)*a(E) (a = potential scattering radius)
        L : angular moment order (scalar)
    Return:
        phs : phase factor
    """

    r2 = rho * rho

    def f0():
        return rho

    def f1():
        return rho - jnp.atan(rho)

    def f2():
        return rho - jnp.atan2(3.0 * rho, 3.0 - r2)

    def f3():
        return rho - jnp.atan2(rho * (15.0 - r2), 15.0 - 6.0 * r2)

    def f4():
        return rho - jnp.atan2(rho * (105.0 - 10.0 * r2), 105.0 - r2 * (45.0 - r2))

    def f5():
        return rho - jnp.atan2(
            rho * (945.0 - r2 * (105.0 - r2)),
            945.0 - r2 * (420.0 - 15.0 * r2)
        )

    def flow(L):
        return lax.switch(L, (f0, f1, f2, f3, f4, f5))

    def fhigh(L):

        ph_prev = f5()

        den = 893025.0 + r2 * (
            99225.0 + r2 * (6300.0 + r2 * (315.0 + r2 * (15.0 + r2)))
        )

        p_prev = rho * r2**5 / den

        s_prev = -(
            4465125.0
            + r2 * (396900.0 + r2 * (18900.0 + r2 * (630.0 + 15.0 * r2)))
        ) / den

        def step(carry, ll):

            ph_prev, p_prev, s_prev = carry

            sdif = ll - s_prev
            ratio = r2 / (sdif * sdif + p_prev * p_prev)

            ph_new = ph_prev - jnp.atan2(p_prev, sdif)
            p_new = ratio * p_prev
            s_new = ratio * sdif - ll

            update = ll <= L

            ph_out = jnp.where(update, ph_new, ph_prev)
            p_out = jnp.where(update, p_new, p_prev)
            s_out = jnp.where(update, s_new, s_prev)

            return (ph_out, p_out, s_out), None

        (ph_final, _, _), _ = lax.scan(
            step,
            (ph_prev, p_prev, s_prev),
            jnp.arange(6, NLMAX)
        )

        return ph_final

    return lax.cond(L < 6, flow, fhigh, L)

#===================================================================================
# Special case (only ap, with NIS > 1)
#===================================================================================

def special(data: Special_DataClass, e: jnp.float64):
    spot = data.abn * 4.0 * jnp.pi * data.ap * data.ap
    pot  = jnp.where(e > 0.0, spot, jnp.array(0.0, dtype=jnp.float64))
    zero = jnp.array(0.0, dtype=jnp.float64)
    return pot, zero, zero, pot, zero

#===================================================================================
# Single Level Breit Wigner (SLBW)
#===================================================================================

def slbw(data: BW_DataClass, e: jnp.float64):
    zero = jnp.array(0.0, dtype=jnp.float64)
    # TODO: Code SLBW core
    return zero, zero, zero, zero, zero

#===================================================================================
# Multi-Levels Breit Wigner (MLBW)
#===================================================================================

def mlbw(data: BW_DataClass, e: jnp.float64):
    """
    Compute resonance cross sections for the MLBW formalism at a single
    incident energy e (scalar) for one isotope/energy range.
    Returns scalars: sct, cap, fis, pot, rxx
    """

    abn  = data.abn
    spi  = data.spi
    ki   = data.ki
    qx   = data.qx
    r_a  = data.r_a
    r_ap = data.r_ap
    nch  = data.nch
    ch   = data.ch
    gch  = data.gch
    ich  = data.ich
    lr   = data.lr
    er   = data.er
    gn   = data.gn
    gg   = data.gg
    gf   = data.gf
    gx   = data.gx

    EPS = jnp.array(1.0e-38, dtype=jnp.float64)

    #-------------
    # Valid masks
    #-------------

    valid_res = ich < nch
    valid_ch  = ch < CHN_PADDED

    #------------------------------------------------
    # Resonance-energy penetration and shift factors
    #------------------------------------------------

    rho_r = rho(er, ki, r_a)                                              # [nr]
    pnt_r, shf_r = get_pnt_shf_vec(rho_r, lr)                             # [nr]

    rho_xr = rhox_chn_res(er, ki, r_a, qx)                                # [nr]
    lx = jnp.where(valid_res, lx_values(lr, spi), 0)                      # [nr]
    pntx_r = get_pnt_vec(rho_xr, lx)                                      # [nr]

    #----------------
    # Reduced widths
    #----------------

    gn0 = jnp.where(pnt_r > EPS, gn / pnt_r, 0.0)                         # [nr]
    gx0 = jnp.where(pntx_r > EPS, gx / pntx_r, 0.0)                       # [nr]

    #--------------------------------------
    # Incident-energy factors (all scalar)
    #--------------------------------------

    e_safe   = jnp.where(e > 0.0, e, 0.0)
    k2       = (ki * jnp.sqrt(jnp.where(e > 0.0, e, 1.0)))**2
    pi_k2    = abn * jnp.where(e > 0.0, jnp.pi / k2, 0.0)
    twopi_k2 = 2.0 * pi_k2

    #--------------------------------------------------
    # Energy-dependent penetration and shift factors
    # rho_e is scalar; vmap pnt_shf_scalar over lr
    #--------------------------------------------------

    rho_e  = rho_scalar(e_safe, ki, r_a)                                  # scalar
    pnt_e, shf_e = jax.vmap(pnt_shf_scalar, in_axes=(None, 0))(rho_e, lr) # [nr]

    rho_xe = rhox_chn_scalar(e_safe, ki, r_a, qx)                         # scalar
    pntx_e = jax.vmap(pnt_scalar, in_axes=(None, 0))(rho_xe, lx)          # [nr]

    #-------------------------------------
    # Widths and shifted resonance energy
    #-------------------------------------

    gn_e = pnt_e * gn0                                                    # [nr]
    gx_e = pntx_e * gx0                                                   # [nr]
    gt   = gn_e + gg + gf + gx_e                                          # [nr]
    erp  = er + 0.5 * (shf_r - shf_e) * gn0                               # [nr]

    de      = 2.0 * (e_safe - erp)                                        # [nr]
    denom   = gt**2 + de**2                                               # [nr]
    bw_ratio = jnp.where(denom > EPS, 2.0 * gn_e / denom, 0.0)            # [nr]

    #------------------------
    # Resonance terms (MLBW)
    #------------------------

    A_r   = bw_ratio * gt                                                 # [nr]
    B_r   = bw_ratio * de                                                 # [nr]
    cap_r = bw_ratio * gg                                                 # [nr]
    fis_r = bw_ratio * gf                                                 # [nr]
    rxx_r = bw_ratio * gx_e                                               # [nr]

    #---------------------------------
    # Sum resonances by channel index
    #---------------------------------

    def sum_by_channel(sig_r):
        return jax.ops.segment_sum(sig_r, ich, ch.shape[0])               # [mch]

    A_ch   = sum_by_channel(A_r)
    B_ch   = sum_by_channel(B_r)
    cap_ch = sum_by_channel(cap_r)
    fis_ch = sum_by_channel(fis_r)
    rxx_ch = sum_by_channel(rxx_r)

    #----------------------------------------------
    # Channel L-values recovered from unique key
    # ch = CHN_STRIDE * L + CHN_OFFSET + 2 * J
    #----------------------------------------------

    lch = jnp.where(valid_ch, ch // CHN_STRIDE, 0)                        # [mch]

    #--------------------------------------------------------
    # Potential scattering phase and contribution by channel
    #--------------------------------------------------------

    rho_s   = rho_scalar(e_safe, ki, r_ap)                                # scalar
    phi2_ch = 2.0 * jax.vmap(phs_scalar, in_axes=(None, 0))(rho_s, lch)   # [mch]
    phi2_ch = jnp.where(valid_ch, phi2_ch, 0.0)                           # [mch]

    pot_ch  = 1.0 - jnp.cos(phi2_ch)                                      # [mch]

    #--------------------------------------------------
    # Elastic scattering by channel
    #--------------------------------------------------

    sct_ch = (pot_ch - A_ch)**2 + (jnp.sin(phi2_ch) + B_ch)**2            # [mch]

    #---------------------------------------
    # Weighted sum over channels -> scalars
    #---------------------------------------

    gmask = jnp.where(valid_ch, gch, 0.0)                                 # [mch]

    sct = jnp.sum(sct_ch * gmask) * pi_k2                                 # scalar
    cap = jnp.sum(cap_ch * gmask) * twopi_k2                              # scalar
    fis = jnp.sum(fis_ch * gmask) * twopi_k2                              # scalar
    pot = jnp.sum(pot_ch * gmask) * twopi_k2                              # scalar
    rxx = jnp.sum(rxx_ch * gmask) * twopi_k2                              # scalar

    return sct, cap, fis, pot, rxx

#===================================================================================
# Reich-Moore (reich_moore)
#===================================================================================

def reich_moore(data: RM_DataClass, e: jnp.float64):
    zero = jnp.array(0.0, dtype=jnp.float64)
    # TODO: Code R_M core
    return zero, zero, zero, zero, zero

#===================================================================================
# Adler- Adler (adler_adler)
#===================================================================================

def adler_adler(data: AA_DataClass, e: jnp.float64):
    zero = jnp.array(0.0, dtype=jnp.float64)
    # TODO: Code A-A core
    return zero, zero, zero, zero, zero

#===================================================================================
# R-Matrix Limited(rml)
#===================================================================================

def rml(data: RML_DataClass, e: jnp.float64):
    zero = jnp.array(0.0, dtype=jnp.float64)
    # TODO: Code RML core
    return zero, zero, zero, zero, zero

#===================================================================================
# Unresolved Resonance Region - SLBW (urrxs)
#===================================================================================

def urrxs(data: URR_DataClass, e: jnp.float64):
    zero = jnp.array(0.0, dtype=jnp.float64)
    # TODO: Code unified URRXS core
    return zero, zero, zero, zero, zero

#===================================================================================
# Dummy calculation (no resonances data, NIS = 1)
# (used with ap = 0.0 for dummy ranges)
#===================================================================================

def dummy(data: Dummy_DataClass, e: jnp.float64):
    spot = 4.0 * jnp.pi * data.ap * data.ap
    pot  = jnp.where(e > 0.0, spot, jnp.array(0.0, dtype=jnp.float64))
    zero = jnp.array(0.0, dtype=jnp.float64)
    return zero, zero, zero, pot, zero

#===================================================================================