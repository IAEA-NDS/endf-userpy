import numpy as np


def compute_r2(E_lab, awi, awr, awp, q):
    return awr*(awr+awi-awp)/(awi*awp)*(1.0+(awr+awi)/awr*q/E_lab) 


def convert_angcos_to_cmsys(mu_lab, r2): 
    mu_lab = mu_lab.reshape(1, -1)
    r2 = r2.reshape(-1, 1)
    r = np.sqrt(r2)
    u = mu_lab
    u2 = np.square(mu_lab) 
    mu_cm = (1.0-u2-r2*u2)/(r*(u2-1.0-u*np.sqrt(u2+r2-1.0))) 
    return mu_cm


def convert_angdist_to_labsys(mu_cm, f_cm, r2):
    mu_cm = np.array(mu_cm, copy=None)
    f_cm = np.array(f_cm, copy=None)
    if mu_cm.ndim == 1:
        mu_cm = mu_cm.reshape(1, -1)
        f_cm = f_cm.reshape(1, -1)
    r2 = r2.reshape(-1, 1)
    r = np.sqrt(r2)
    xw = 1.0 + 2.0*r*mu_cm + r2
    f_lab = f_cm * xw * np.sqrt(xw) / (r2*(r+mu_cm))
    return f_lab
