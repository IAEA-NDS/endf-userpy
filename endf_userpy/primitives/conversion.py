import numpy as np


def _correct_mu_cm(mu_cm):
    """Correct numerical issues in mu_cm inplace"""
    # TODO: discuss with Daniel production of NaN
    #       and the interpretation of the situation
    pass


def _correct_r2(r2):
    """Correct issues with r2 inplace"""
    r2min = 1e-76
    r2[r2 < r2min] = r2min


def compute_r2(E_lab, awi, awr, awp, q):
    return awr*(awr+awi-awp)/(awi*awp)*(1.0+(awr+awi)/awr*q/E_lab)


def convert_angcos_to_cmsys(mu_lab, r2):
    """Map a LAB-frame cos(theta) to CM-frame cos(theta) given
    `r2 = compute_r2(...)`. Kinematically forbidden LAB angles
    (where `z = mu_lab^2 + r^2 - 1 < 0` for r < 1, i.e. the
    equal-mass / light-ejectile back-scatter region) produce NaN
    -- the caller is expected to clip NaN to 0 in the LAB return
    since no scattering into unreachable angles is physically
    possible. `np.errstate` here just suppresses the harmless
    `RuntimeWarning: invalid value encountered in sqrt` that the
    guaranteed-negative-in-the-forbidden-region argument emits;
    the NaN itself is exactly what downstream code needs to
    identify the forbidden region.
    """
    mu_lab = mu_lab.reshape(1, -1)
    r2 = r2.reshape(-1, 1)
    _correct_r2(r2)
    r = np.sqrt(r2)
    u = mu_lab
    u2 = np.square(mu_lab)
    z = u2 + r2 - 1.0
    z1 = (1.0-u2-r2*u2)
    with np.errstate(invalid='ignore'):
        z2 = (r*(u2-1.0-u*np.sqrt(z)))
    mu_cm = z1 / z2
    _correct_mu_cm(mu_cm)
    return mu_cm


def convert_angdist_to_labsys(mu_cm, f_cm, r2):
    mu_cm = np.asarray(mu_cm)
    f_cm = np.asarray(f_cm)
    if mu_cm.ndim == 1:
        mu_cm = mu_cm.reshape(1, -1)
        f_cm = f_cm.reshape(1, -1)
    r2 = r2.reshape(-1, 1)
    _correct_r2(r2)
    r = np.sqrt(r2)
    xw = 1.0 + 2.0*r*mu_cm + r2
    f_lab = f_cm * xw * np.sqrt(xw) / (r2*(r+mu_cm))
    return f_lab
