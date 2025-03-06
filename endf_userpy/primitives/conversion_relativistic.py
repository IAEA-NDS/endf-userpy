import numpy as np


# Relativistic conversion from kinetic energy of emitted particle
# to cosine angle of emitted particle and the opposite direction
# (all in laboratory system). 
#
# Meaning of variable names
#
# Ekin_i: kinetic energy of incident particle in MeV
#
# m_i: mass of incident particle in MeV
# m_t: mass of target nucleus in MeV
# m_r: mass of residual nucleus in MeV
# m_e: mass of ejectile in MeV
#
# Ekin: kinetic energy of ejectile in MeV
# cos_phi: cosine of scattering angle of ejectile

def compute_Ekin_from_cos_phi(cos_phi, Ekin_i, m_i, m_t, m_e, m_r):
    x0 = m_e**2
    x1 = cos_phi**2
    x2 = Ekin_i**2 + 2*Ekin_i*m_i
    x3 = x1*x2
    x4 = m_i**2
    x5 = x2 + x4
    x6 = m_t**2
    x7 = np.sqrt(x5)
    x8 = m_t*x7
    x9 = 2*x8
    x10 = x6 + x9
    x11 = m_r**2
    x12 = m_e**4
    x13 = m_r**4
    x14 = m_i**4
    x15 = 2*x0
    x16 = m_t**4
    x17 = 2*x11
    x18 = x11*x15
    x19 = x0*x6
    x20 = 12*x4
    x21 = x11*x6
    x22 = 4*x2**2
    x23 = x0*x22
    x24 = x2*x4
    x25 = 6*x7
    x26 = 4*x3
    x27 = m_t**3*x7
    x28 = 8*x0
    x29 = 8*x11
    x30 = x4*x8
    x31 = x0*x8
    x32 = 4*x11
    x33 = 12*x2
    Ekin_result = (
        -m_e + np.sqrt(x0 +
        ((1/2)*cos_phi*np.sqrt(x2)*(x0 + x10 - x11 + x4) -
        1/2*np.sqrt(m_i**6 + m_t**6 + m_t**5*x25 + m_t*x14*x25 -
        6*x0*x24 + x0*x26*x4 + x1*x23 + x12*x2 + x12*x4 + x12*x6 +
        x12*x9 + x13*x2 + x13*x4 + x13*x6 + x13*x9 - x14*x15 -
        x14*x17 + x14*x2 + 15*x14*x6 - x15*x16 - x15*x21 - x16*x17 +
        13*x16*x2 + 15*x16*x4 - x17*x24 - x18*x2 - x18*x4 -
        14*x19*x2 - x19*x20 + x19*x26 - 10*x2*x21 - x2*x32*x8 -
        x20*x21 + x22*x6 - x23 + 18*x24*x6 + 4*x24*x8 - x27*x28 -
        x27*x29 + x27*x33 + 20*x27*x4 + x28*x3*x8 - x28*x30 -
        x29*x30 - x31*x32 - x31*x33))**2/(x10 - x3 + x5)**2)
    )
    return Ekin_result


def compute_cos_phi_from_Ekin(Ekin, Ekin_i, m_i, m_t, m_e, m_r):
    x0 = Ekin**2 + 2*Ekin*m_e
    x1 = Ekin_i**2 + 2*Ekin_i*m_i
    cos_phi_result = (
        -1/2*(m_r**2 + x0 + x1 - (m_t -
        np.sqrt(m_e**2 + x0) + np.sqrt(m_i**2 +
        x1))**2)/(np.sqrt(x0)*np.sqrt(x1))
    )
    return cos_phi_result


def compute_dEkin_dcos_phi(cos_phi, Ekin_i, m_i, m_t, m_e, m_r):
    x0 = m_e**2
    x1 = cos_phi**2
    x2 = Ekin_i**2 + 2*Ekin_i*m_i
    x3 = x1*x2
    x4 = m_i**2
    x5 = x2 + x4
    x6 = m_t**2
    x7 = np.sqrt(x5)
    x8 = m_t*x7
    x9 = 2*x8
    x10 = x6 + x9
    x11 = x10 - x3 + x5
    x12 = x11**(-2)
    x13 = m_r**2
    x14 = np.sqrt(x2)*(x0 + x10 - x13 + x4)
    x15 = m_e**4
    x16 = m_r**4
    x17 = m_i**4
    x18 = 2*x0
    x19 = m_t**4
    x20 = 2*x13
    x21 = x13*x18
    x22 = x0*x6
    x23 = 12*x4
    x24 = x13*x6
    x25 = 4*x2**2
    x26 = x0*x25
    x27 = x2*x4
    x28 = x0*x27
    x29 = x2*x22
    x30 = 6*x7
    x31 = 4*x3
    x32 = m_t**3*x7
    x33 = 8*x0
    x34 = 8*x13
    x35 = x4*x8
    x36 = x0*x8
    x37 = 4*x13
    x38 = 12*x2
    x39 = x2*x8
    x40 = (
        np.sqrt(m_i**6 + m_t**6 + m_t**5*x30 + m_t*x17*x30 +
        x0*x31*x4 + x1*x26 + x15*x2 + x15*x4 + x15*x6 + x15*x9 +
        x16*x2 + x16*x4 + x16*x6 + x16*x9 - x17*x18 + x17*x2 -
        x17*x20 + 15*x17*x6 - x18*x19 - x18*x24 + 13*x19*x2 -
        x19*x20 + 15*x19*x4 - x2*x21 - 10*x2*x24 - x20*x27 - x21*x4
        - x22*x23 + x22*x31 - x23*x24 + x25*x6 - x26 + 18*x27*x6 +
        4*x27*x8 - 6*x28 - 14*x29 + x3*x33*x8 - x32*x33 - x32*x34 +
        x32*x38 + 20*x32*x4 - x33*x35 - x34*x35 - x36*x37 - x36*x38
        - x37*x39)
    )
    x41 = (1/2)*cos_phi*x14 - 1/2*x40
    x42 = x41**2
    x43 = 4*cos_phi
    dEkin_dcos_phi_result = (
        (2*cos_phi*x2*x42/x11**3 +
        (1/2)*x12*x41*(x14 - (cos_phi*x26 + cos_phi*x33*x39 +
        x28*x43 + x29*x43)/x40))/np.sqrt(x0 + x12*x42)
    )
    return dEkin_dcos_phi_result


def compute_dcos_phi_dEkin(Ekin, Ekin_i, m_i, m_t, m_e, m_r):
    x0 = 2*Ekin
    x1 = Ekin**2 + m_e*x0
    x2 = Ekin + m_e
    x3 = np.sqrt(m_e**2 + x1)
    x4 = Ekin_i**2 + 2*Ekin_i*m_i
    x5 = m_t - x3 + np.sqrt(m_i**2 + x4)
    x6 = (1/2)/np.sqrt(x4)
    dcos_phi_dEkin_result = (
        -x6*(2*m_e + x0 +
        2*x2*x5/x3)/np.sqrt(x1) + x2*x6*(m_r**2 + x1 + x4 -
        x5**2)/x1**(3/2)
    )
    return dcos_phi_dEkin_result


def convert_energydist_to_angdist(
    dist_vals, Ekin, Ekin_i, m_i, m_t, m_e, m_r  
):
    # p(Ekin) dEkin = p(mu) dmu
    # p(Ekin) = p(mu) dmu/dEkin
    cos_phi_vals = compute_cos_phi_from_Ekin(
        Ekin, Ekin_i, m_i, m_t, m_e, m_r
    )  
    funcdet_vals = compute_dcos_phi_dEkin(
        Ekin, Ekin_i, m_i, m_t, m_e, m_r
    )
    new_dist_vals * np.abs(funcdet_vals) 
    return new_dist_vals


def test_Ekin():
    MeV_per_amu = 9.3149410242e8 / 1e6  # eV / c^2  
    m_i = 1.0 * MeV_per_amu
    m_t = 56 * MeV_per_amu
    m_r = 53.001 * MeV_per_amu
    m_e = 4.0 * MeV_per_amu
    Ekin_i = 5
    phi = 0 * 3.14159 / 180.0 #  sp.pi 
    cos_phi = np.cos(phi)
    print(f'cos_phi: {cos_phi}')
    val = compute_Ekin_from_cos_phi(cos_phi, Ekin_i, m_i, m_t, m_e, m_r)
    print(f'Ekin: {val}')


def test_cos_phi():
    MeV_per_amu = 9.3149410242e8 / 1e6  # eV / c^2  
    m_i = 1.0 * MeV_per_amu
    m_t = 56 * MeV_per_amu
    m_r = 53.001 * MeV_per_amu
    m_e = 4.0 * MeV_per_amu
    Ekin_i = 5
    Ekin = 3.40494307103972
    val = compute_cos_phi_from_Ekin(Ekin, Ekin_i, m_i, m_t, m_e, m_r)
    print(val)


def test_limits():
    MeV_per_amu = 9.3149410242e8 / 1e6  # eV / c^2  
    m_i = 1.0 * MeV_per_amu
    m_t = 56 * MeV_per_amu
    m_r = 53.005273305 * MeV_per_amu
    m_e = 4.0 * MeV_per_amu
    Ekin_i = 5
    phi = 0 * 3.14159 / 180.0 #  sp.pi 
    cos_phi = np.cos(phi)
    lims = get_cos_phi_limit(Ekin_i, m_i, m_t, m_e, m_r)
    print(f'limits: {lims}')
    print(f'cos_phi: {cos_phi}')
    val = compute_Ekin_from_cos_phi(cos_phi, Ekin_i, m_i, m_t, m_e, m_r)
    print(f'Ekin: {val}')


# compute_dEkin_dmu(m_i, m_t, m_e, Q, cos_phi):
# test_Ekin()
# test_cos_phi()
# test_limits()
