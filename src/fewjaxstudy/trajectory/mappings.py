import numpy as np
import jax.numpy as jnp
from jax import jit, custom_jvp, jacobian, vmap

XMIN = 0.05
AMAX = 0.999
AMIN = -AMAX
DELTAPMIN = 0.001
DELTAPMAX = 9 + DELTAPMIN
EMAX = 0.9
ESEP = 0.25

ALPHA_FLUX = 1.0 / 2.0
BETA_FLUX = 2.0
ALPHA_AMP = 1.0 / 3.0
BETA_AMP = 3.0

DPC_REGIONB = DELTAPMAX - 0.001
PMAX_REGIONB = 200
AMAX_REGIONB = 0.999
AMIN_REGIONB = -AMAX_REGIONB
EMAX_REGIONB = 0.9
DELTAPMIN_REGIONB = 9

@jit
def sep_analytic(a, e, x):
    ats = a**2 * (1 + e)

    del13 = (-ats**3*(-3+e)**3 + (3+e)**6 - 3*ats*(3+e)**3*(15+e**2) + 3*ats**2*(153 + 6*e**2 + e**4) + 24*3**0.5*jnp.sqrt(-ats**3*(-1+ats-e)*(ats*(-3+e)**3-(-1+e)*(3+e)**3)))**(1/3)
    TermOneSq = ats*(1 - e/3) + 1/3 * (3+e)**2 + (1/3*ats**2*(-3+e)**2 + 1/3*(3+e)**4 - 2/3*ats*(3+e)*(15+e**2))/del13 + del13/3
    TermOne = jnp.sqrt(TermOneSq)
    TermTwo = jnp.sqrt((3+e)**2 + ats*(3-e+16/TermOne) - TermOneSq)
    return 3 + e + TermOne - x*TermTwo

@jit
def dsepde(a, e, x):
    ps = sep_analytic(a, e, x)
    return (
        a**4*(-3+e)*(-1+e)*(1+e) +
        ps*(-a**2*(7+e*(2+3*e)) + (a**2*(-1+e)+2*(3+e)-ps)*ps)
        ) /\
        (
            a**2*(1+e)*(7+e**2) - ps*(a**2*(-3+e)*(1+e)+2*(3+e)**2 - 3*(3+e)*ps + ps**2)
        )

@jit
def u_of_p(p, pLSO):
    check_term = (jnp.log(p - pLSO + DELTAPMAX - 2 * DELTAPMIN) - jnp.log(DELTAPMAX - DELTAPMIN)) / np.log(2)
    sgn = jnp.sign(check_term)
    return sgn * (sgn * check_term)**ALPHA_FLUX

@jit
def chi_of_a(a):
    return (1 - a) ** (1 / 3)


@jit
def chi2_of_a(a):
    return (1 - a) ** (2 / 3)

@jit
def z_of_a(a):
    chimax = chi_of_a(AMIN)
    chimin = chi_of_a(AMAX)
    return (chi_of_a(a) - chimin) / (chimax - chimin)

@jit
def z2_of_a(a):
    chimax = chi2_of_a(AMIN)
    chimin = chi2_of_a(AMAX)
    return (chi2_of_a(a) - chimin) / (chimax - chimin)

@jit
def Secc_of_uz(
    u,
    z,
):  
    check_part = z + u**BETA_FLUX * (1 - z)
    sgn = jnp.sign(check_part)
    return ESEP + (EMAX - ESEP) * sgn * jnp.sqrt(sgn*check_part)

@jit
def w_of_euz(e, u, z):
    return e / Secc_of_uz(u, z)

@jit
def w_of_euz_flux(e, u, z):
    return e / Secc_of_uz(u, z)

@jit
def p_of_u_flux(u, pLSO):
    return (pLSO + DELTAPMIN) + (DELTAPMAX - DELTAPMIN) * (np.exp(u ** (1 / ALPHA_FLUX) * log(2)) - 1)

@jit
def a_of_chi2(chi):
    return 1 - chi ** (1.5)

@jit
def a_of_z2(z):
    chimax = chi2_of_a(AMIN)
    chimin = chi2_of_a(AMAX)
    return a_of_chi2(chimin + z * (chimax - chimin))

@jit
def a_of_chi(chi):
    return 1 - chi**3

@jit
def a_of_z(z):
    chimax = chi_of_a(AMIN)
    chimin = chi_of_a(AMAX)
    return a_of_chi(chimin + z * (chimax - chimin))


@jit
def e_of_uwz(u, w, z):
    return Secc_of_uz(u, z) * w

@jit
def uwz_of_ape(
    a,
    p,
    e,
):  
    pLSO = sep_analytic(a, e, 1.)
    u = u_of_p(p, pLSO)
    z = z_of_a(a)
    w = w_of_euz(e, u, z)
    return jnp.asarray([u, w, z])


uwz_of_ape_jac = vmap(jacobian(uwz_of_ape, argnums=(1, 2)), in_axes=(0, 0, 0), out_axes=1)

