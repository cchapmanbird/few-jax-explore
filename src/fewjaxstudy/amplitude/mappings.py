import jax.numpy as jnp
from ..trajectory.mappings import get_separatrix

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
PMAX_REGIONB = 200.0
AMAX_REGIONB = 0.999
AMIN_REGIONB = -AMAX_REGIONB
EMAX_REGIONB = 0.9
DELTAPMIN_REGIONB = 9


def u_of_p(p, pLSO):
    check_term = (
        jnp.log(p - pLSO + DELTAPMAX - 2 * DELTAPMIN) - jnp.log(DELTAPMAX - DELTAPMIN)
    ) / jnp.log(2)
    sgn = jnp.sign(check_term)
    return sgn * (sgn * check_term) ** ALPHA_AMP


def chi_of_a(a):
    return (1 - a) ** (1 / 3)

chimax = chi_of_a(AMIN)
chimin = chi_of_a(AMAX)

def z_of_a(a):
    return (chi_of_a(a) - chimin) / (chimax - chimin)

def Secc_of_uz(
    u,
    z,
):  
    check_part = z + u**BETA_AMP * (1 - z)
    sgn = jnp.sign(check_part)
    return ESEP + (EMAX - ESEP) * sgn * jnp.sqrt(sgn*check_part)

def w_of_euz(e, u, z):
    return e / Secc_of_uz(u, z)

def uwz_of_ape_amp(
    a,
    p,
    e,
):
    pLSO = get_separatrix(a, e, 1.)
    u = u_of_p(p, pLSO)
    z = z_of_a(a)
    w = w_of_euz(e, u, z)
    return u, w, z
