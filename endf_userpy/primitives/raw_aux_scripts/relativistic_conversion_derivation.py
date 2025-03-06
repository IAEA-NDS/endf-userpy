import sympy as sp

m_i, m_t, m_e, m_r = sp.symbols('m_i m_t m_e m_r', positive=True)

p_i, p_r, p_e = sp.symbols('p_i p_r p_e')
cos_phi, cos_theta, sin_phi, sin_theta = sp.symbols('cos_phi cos_theta sin_phi sin_theta')
c = sp.symbols('c', positive=True)

MeV_per_amu = 9.3149410242e8 / 1e6  # eV / c^2  

c_val = 1.0
m_i_val = 3.0
m_t_val = 7.0
m_e_val = 1.0
m_r_val = 0.8
p_i_val = 30.0
phi_val = 10 * 3.14159 / 180.0 #  sp.pi 


# Energy of projectile and (E_i) and target (E_t) before collision
E_i = sp.sqrt(m_i**2 + p_i**2) 
E_t = m_t
# Energy of ejectile (E_e) and residual nucleus (E_r) after collision
E_e = sp.sqrt(m_e**2 + p_e**2)
E_r = sp.sqrt(m_r**2 + p_r**2)

# corresponding four-momentum vectors
p4_i = sp.Matrix([[E_i], [p_i], [0.0]])
p4_t = sp.Matrix([[E_t], [0.0], [0.0]])
p4_e = sp.Matrix([[E_e], [p_e * cos_phi], [p_e * sin_phi]])
p4_r = sp.Matrix([[E_r], [p_r * cos_theta], [p_r * sin_theta]])

eq = p4_i + p4_t - p4_e - p4_r

# eliminate sin_theta

sin_theta_eq = eq[2] 
sin_theta_expr = sp.solve(sin_theta_eq, sin_theta)[0]

# check if it works as expected
p_e_val = 12
p_r_val = 18
sin_phi_val = 0.003
sin_theta_val = sin_theta_expr.subs({'sin_phi': sin_phi_val, 'p_e': p_e_val, 'p_r': p_r_val})
sin_theta_eq.subs({'sin_phi': sin_phi_val, 'p_e': p_e_val, 'p_r': p_r_val, 'sin_theta': sin_theta_val})

# eliminate p_r 
eq_x = eq[1].subs(cos_theta, sp.sqrt(1 - sin_theta_expr**2))
p_r_expr = sp.solve(eq_x, p_r)[1]

# check if works as expected
p_i_val = 18.23
p_e_val = 12.12
phi = 3.02 
sin_phi_val = sp.sin(0.001).evalf()
cos_phi_val = sp.cos(0.001).evalf()
p_r_val = p_r_expr.subs({'cos_phi': cos_phi_val, 'sin_phi': sin_phi_val, 'p_e': p_e_val, 'p_i': p_i_val})
eq_x.subs({'cos_phi': cos_phi_val, 'sin_phi': sin_phi_val, 'p_e': p_e_val, 'p_i': p_i_val, 'p_r': p_r_val})

# eliminate p_e

energy_eq = eq[0].subs({'p_r': p_r_expr, 'sin_phi': sp.sqrt(1-cos_phi**2)})

p_e_expr = sp.solve(energy_eq, p_e)[0]
cos_phi_expr = sp.solve(energy_eq, cos_phi)[0]

# check if works as expected as function of cos_phi
p_e_val = p_e_expr.xreplace({c: c_val, m_i: m_i_val, m_t: m_t_val, m_e: m_e_val, cos_phi: cos_phi_val, p_i: p_i_val, m_r: m_r_val})
energy_eq.xreplace({c: c_val, m_i: m_i_val, m_t: m_t_val, m_e: m_e_val, cos_phi: cos_phi_val, p_i: p_i_val, p_e: p_e_val, m_r: m_r_val})

# check if works as expected as function of p_e
cos_phi_val = cos_phi_expr.xreplace({c: c_val, m_i: m_i_val, m_t: m_t_val, m_e: m_e_val, p_e: p_e_val, p_i: p_i_val, m_r: m_r_val})
energy_eq.xreplace(
    {c: c_val, m_i: m_i_val, m_t: m_t_val, m_e: m_e_val, p_e: p_e_val, p_i: p_i_val, p_e: p_e_val, cos_phi: cos_phi_val, m_r: m_r_val}
)

m_i_val = 1.0 * MeV_per_amu
m_t_val = 56 * MeV_per_amu
m_r_val = 56 * MeV_per_amu
m_e_val = 1.0 * MeV_per_amu
phi_val = 10 * 3.14159 / 180.0 #  sp.pi 
Ekin_i_val = m_i_val + 1

substis  = {
    m_i: m_i_val, 
    m_t: m_t_val, 
    m_r: m_r_val,
    m_e: m_e_val, 
    p_i: sp.sqrt((m_i_val + Ekin_i_val)**2 - m_i_val**2),
    sin_phi: sp.sin(phi_val).evalf(),
    cos_phi: sp.cos(phi_val).evalf(),
    cos_theta: sp.sqrt(1 - sin_theta_expr),
}


# check that equations solved correctly, should give a vector of zeros
check_eq = eq.copy()
(check_eq
    .xreplace({sin_theta: sin_theta_expr, cos_theta: sp.sqrt(1-sin_theta_expr**2)})
    .xreplace({p_r: p_r_expr})
    .xreplace({p_e: p_e_expr})
).xreplace(substis)


# here we are done!!!
# p_e_expr contains the formula to calculate p_e as a function of c, m_t, cos_phi, m_i, m_t, m_e
p_e_expr.free_symbols

# replace p_i by E_i_kin
Ekin_i = sp.symbols('Ekin_i', positive=True)
p_i_expr2 = sp.sqrt(Ekin_i**2 + 2*Ekin_i*m_i)

Ekin_func = E_e.xreplace({p_e: p_e_expr}).xreplace({p_i: p_i_expr2}) - m_e


# E**2 = m**2 + p**2 * c**2
# (E**2 - m**2) / c**2 = p**2
# E = m + T
# p = sqrt(E**2 - m**2) / c
# p = sqrt((T+m)**2 - m**2) / c
Ekin = sp.symbols('Ekin', positive=True)
p_e_expr2 = sp.sqrt(Ekin**2 + 2*Ekin*m_e)

# substitute p_e by kinetic energy of ejectile
cos_phi_func = -cos_phi_expr.xreplace({p_e: p_e_expr2}).xreplace({p_i: p_i_expr2})


m_i_val = 1.0 * MeV_per_amu
m_t_val = 56 * MeV_per_amu
m_r_val = 53.001 * MeV_per_amu
m_e_val = 4.0 * MeV_per_amu
phi_val = 0 * 3.14159 / 180.0 #  sp.pi 
Ekin_i_val = 5

substis  = {
    m_i: m_i_val, 
    m_t: m_t_val, 
    m_r: m_r_val,
    m_e: m_e_val, 
    # p_i: sp.sqrt((m_i_val + Ekin_i_val)**2 - m_i_val**2),
    Ekin_i: Ekin_i_val,
    sin_phi: sp.sin(phi_val).evalf(),
    cos_phi: sp.cos(phi_val).evalf(),
    cos_theta: sp.sqrt(1 - sin_theta_expr),
}


print(f'orig cos_phi: {substis[cos_phi]}')
Ekin_val = Ekin_func.xreplace(substis)
print(f'Ekin_val: {Ekin_val}')
cos_phi_val = cos_phi_func.xreplace({Ekin: Ekin_val}).xreplace(substis)
print(f'cos_phi_val: {cos_phi_val}')
recon_Ekin_val = Ekin_func.xreplace({cos_phi: cos_phi_val}).xreplace(substis).evalf()
print(f'recon. Ekin_val: {recon_Ekin_val}')


# transformation of probability distribution
p(mu) dmu = p(E) dE

p(E) = p(mu) * dmu/dE


dE_dmu = sp.diff(Ekin_func, cos_phi)
dmu_dE = sp.diff(cos_phi_func, Ekin)


# prepare the python code

import textwrap
from sympy.printing import pycode
from sympy.codegen import Assignment


def print_code(expr):
    code = pycode(expr).replace('math', 'np')
    print(textwrap.fill(code, width=60, break_long_words=False))


Ekin_result = sp.symbols('Ekin_result')
z = sp.cse(Ekin_func)
print('\n'.join([pycode(Assignment(*x)) for x in z[0]]))
code = pycode(Assignment(Ekin_result, z[1][0]))
print(textwrap.fill(code, width=60, break_long_words=False).replace('math', 'np'))  


cos_phi_result = sp.symbols('cos_phi_result')
z = sp.cse(cos_phi_func)
print('\n'.join([pycode(Assignment(*x)) for x in z[0]]))
code = pycode(Assignment(cos_phi_result, z[1][0]))
print(textwrap.fill(code, width=60, break_long_words=False).replace('math', 'np'))  


dEkin_dcos_phi_result = sp.symbols('dEkin_dcos_phi_result')
z = sp.cse(dE_dmu)
code = '\n'.join([pycode(Assignment(*x)) for x in z[0]]) 
print(textwrap.fill(code, width=60, break_long_words=False, replace_whitespace=False).replace('math', 'np'))  
code = pycode(Assignment(dEkin_dcos_phi_result, z[1][0]))
print(textwrap.fill(code, width=60, break_long_words=False).replace('math', 'np'))  


dcos_phi_dEkin_result = sp.symbols('dcos_phi_dEkin_result')
z = sp.cse(dmu_dE)
code = '\n'.join([pycode(Assignment(*x)) for x in z[0]]) 
print(textwrap.fill(code, width=60, break_long_words=False, replace_whitespace=False).replace('math', 'np'))  
code = pycode(Assignment(dcos_phi_dEkin_result, z[1][0]))
print(textwrap.fill(code, width=60, break_long_words=False).replace('math', 'np'))  


# Try to find kinematic limits by looking at the square roots


def get_radicands(expr):
    radicands = [e.args[0] for e in sp.preorder_traversal(expr)
            if isinstance(e, sp.Pow) and abs(e.exp.evalf()) == 0.5]
    return radicands


def find_subvars_for_replacement(elem_var, subvars, subexprs):
    repl_symbols = {v: e for v, e in zip(subvars, subexprs) if elem_var in e.free_symbols}
    for v, e in zip(subvars, subexprs):
        if any(s in e.free_symbols for s in repl_symbols):
            repl_symbols[v] = e
    return repl_symbols


def substitute_subvars(repl_dict, expr): 
    new_expr = expr.xreplace(repl_dict)
    while new_expr != expr:
        expr = new_expr
        new_expr = expr.xreplace(repl_dict)
    return new_expr


z = sp.cse(Ekin_func)
subexprs, redexpr = z

all_subvars = [v for v, e in z[0]]
all_subexprs = [e for v, e in z[0]]  

repl_dict = find_subvars_for_replacement(cos_phi, all_subvars, all_subexprs)

all_subexpr_radicands = [get_radicands(e) for e in all_subexprs]
all_subexpr_radicands = sum(all_subexpr_radicands, start=[])

rads_in_expr_using_subvars = get_radicands(z[1])
rs = rads_in_expr_using_subvars
rs = [substitute_subvars(repl_dict, r) for r in rs]
rs = [r for r in rs if cos_phi in r.free_symbols]

# rs[0] is the outer square root
# rs[1] is the inner square root

cos_phi_limits = sp.solve(rs[0], cos_phi)
print_code(cos_phi_limits[0])
print_code(cos_phi_limits[1])
