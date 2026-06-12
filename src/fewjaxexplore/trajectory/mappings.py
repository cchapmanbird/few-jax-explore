import numpy as np
import jax.numpy as jnp
from jax import jit, custom_jvp, jacobian, vmap, Array
from optimistix import Bisection, root_find
from jax import lax

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
def get_separatrix_analytic_equatorial(a, e, x):
    ats = a**2 * (1 + e)

    del13 = (-ats**3*(-3+e)**3 + (3+e)**6 - 3*ats*(3+e)**3*(15+e**2) + 3*ats**2*(153 + 6*e**2 + e**4) + 24*3**0.5*jnp.sqrt(-ats**3*(-1+ats-e)*(ats*(-3+e)**3-(-1+e)*(3+e)**3)))**(1/3)
    TermOneSq = ats*(1 - e/3) + 1/3 * (3+e)**2 + (1/3*ats**2*(-3+e)**2 + 1/3*(3+e)**4 - 2/3*ats*(3+e)*(15+e**2))/del13 + del13/3
    TermOne = jnp.sqrt(TermOneSq)
    TermTwo = jnp.sqrt((3+e)**2 + ats*(3-e+16/TermOne) - TermOneSq)
    return 3 + e + TermOne - x*TermTwo


def _separatrix_polynomial_full(p, args):
    a = args[0]
    e = args[1]
    x = args[2]
    return (
        -4 * (3 + e) * pow(p, 11)
        + pow(p, 12)
        + pow(a, 12) * pow(-1 + e, 4) * pow(1 + e, 8) * pow(-1 + x, 4) * pow(1 + x, 4)
        - 4
        * pow(a, 10)
        * (-3 + e)
        * pow(-1 + e, 3)
        * pow(1 + e, 7)
        * p
        * pow(-1 + pow(x, 2), 4)
        - 4
        * pow(a, 8)
        * (-1 + e)
        * pow(1 + e, 5)
        * pow(p, 3)
        * pow(-1 + x, 3)
        * pow(1 + x, 3)
        * (
            7
            - 7 * pow(x, 2)
            - pow(e, 2) * (-13 + pow(x, 2))
            + pow(e, 3) * (-5 + pow(x, 2))
            + 7 * e * (-1 + pow(x, 2))
        )
        + 8
        * pow(a, 6)
        * (-1 + e)
        * pow(1 + e, 3)
        * pow(p, 5)
        * pow(-1 + pow(x, 2), 2)
        * (
            3
            + e
            + 12 * pow(x, 2)
            + 4 * e * pow(x, 2)
            + pow(e, 3) * (-5 + 2 * pow(x, 2))
            + pow(e, 2) * (1 + 2 * pow(x, 2))
        )
        - 8
        * pow(a, 4)
        * pow(1 + e, 2)
        * pow(p, 7)
        * (-1 + x)
        * (1 + x)
        * (
            -3
            + e
            + 15 * pow(x, 2)
            - 5 * e * pow(x, 2)
            + pow(e, 3) * (-5 + 3 * pow(x, 2))
            + pow(e, 2) * (-1 + 3 * pow(x, 2))
        )
        + 4
        * pow(a, 2)
        * pow(p, 9)
        * (
            -7
            - 7 * e
            + pow(e, 3) * (-5 + 4 * pow(x, 2))
            + pow(e, 2) * (-13 + 12 * pow(x, 2))
        )
        + 2
        * pow(a, 8)
        * pow(-1 + e, 2)
        * pow(1 + e, 6)
        * pow(p, 2)
        * pow(-1 + pow(x, 2), 3)
        * (
            2 * pow(-3 + e, 2) * (-1 + pow(x, 2))
            + pow(a, 2)
            * (
                pow(e, 2) * (-3 + pow(x, 2))
                - 3 * (1 + pow(x, 2))
                + 2 * e * (1 + pow(x, 2))
            )
        )
        - 2
        * pow(p, 10)
        * (
            -2 * pow(3 + e, 2)
            + pow(a, 2)
            * (
                -3
                + 6 * pow(x, 2)
                + pow(e, 2) * (-3 + 2 * pow(x, 2))
                + e * (-2 + 4 * pow(x, 2))
            )
        )
        + pow(a, 6)
        * pow(1 + e, 4)
        * pow(p, 4)
        * pow(-1 + pow(x, 2), 2)
        * (
            -16 * pow(-1 + e, 2) * (-3 - 2 * e + pow(e, 2)) * (-1 + pow(x, 2))
            + pow(a, 2)
            * (
                15
                + 6 * pow(x, 2)
                + 9 * pow(x, 4)
                + pow(e, 2) * (26 + 20 * pow(x, 2) - 2 * pow(x, 4))
                + pow(e, 4) * (15 - 10 * pow(x, 2) + pow(x, 4))
                + 4 * pow(e, 3) * (-5 - 2 * pow(x, 2) + pow(x, 4))
                - 4 * e * (5 + 2 * pow(x, 2) + 3 * pow(x, 4))
            )
        )
        - 4
        * pow(a, 4)
        * pow(1 + e, 2)
        * pow(p, 6)
        * (-1 + x)
        * (1 + x)
        * (
            -2 * (11 - 14 * pow(e, 2) + 3 * pow(e, 4)) * (-1 + pow(x, 2))
            + pow(a, 2)
            * (
                5
                - 5 * pow(x, 2)
                - 9 * pow(x, 4)
                + 4 * pow(e, 3) * pow(x, 2) * (-2 + pow(x, 2))
                + pow(e, 4) * (5 - 5 * pow(x, 2) + pow(x, 4))
                + pow(e, 2) * (6 - 6 * pow(x, 2) + 4 * pow(x, 4))
            )
        )
        + pow(a, 2)
        * pow(p, 8)
        * (
            -16 * pow(1 + e, 2) * (-3 + 2 * e + pow(e, 2)) * (-1 + pow(x, 2))
            + pow(a, 2)
            * (
                15
                - 36 * pow(x, 2)
                + 30 * pow(x, 4)
                + pow(e, 4) * (15 - 20 * pow(x, 2) + 6 * pow(x, 4))
                + 4 * pow(e, 3) * (5 - 12 * pow(x, 2) + 6 * pow(x, 4))
                + 4 * e * (5 - 12 * pow(x, 2) + 10 * pow(x, 4))
                + pow(e, 2) * (26 - 72 * pow(x, 2) + 44 * pow(x, 4))
            )
        )
    )


def _separatrix_polynomial_polar(p, args):
    a = args[0]
    e = args[1]
    return (
        pow(a, 6) * pow(-1 + e, 2) * pow(1 + e, 4)
        + pow(p, 5) * (-6 - 2 * e + p)
        + pow(a, 2)
        * pow(p, 3)
        * (-4 * (-1 + e) * pow(1 + e, 2) + (3 + e * (2 + 3 * e)) * p)
        - pow(a, 4)
        * pow(1 + e, 2)
        * p
        * (6 + 2 * pow(e, 3) + 2 * e * (-1 + p) - 3 * p - 3 * pow(e, 2) * (2 + p))
    )

def _separatrix_polynomial_equat(p, args):
    a = args[0]
    e = args[1]
    return (
        pow(a, 4) * pow(-3 - 2 * e + pow(e, 2), 2)
        + pow(p, 2) * pow(-6 - 2 * e + p, 2)
        - 2 * pow(a, 2) * (1 + e) * p * (14 + 2 * pow(e, 2) + 3 * p - e * p)
    )

def get_separatrix(a: float, e: float, x: float) -> Array:
    """
    Root-find for the separatrix in Kerr spacetime.

    N.B. the default tolerances need some tuning / rescaling for accuracy in generic.
    """
#     # Detect invalid inputs
    any_nan = jnp.isnan(a) | jnp.isnan(e) | jnp.isnan(x)
    any_inf = jnp.isinf(a) | jnp.isinf(e) | jnp.isinf(x)
    any_invalid = any_nan | any_inf

    is_schw = jnp.isclose(a, 0.0, atol=1e-12)
    is_equat = jnp.isclose(jnp.abs(x), 1.0, atol=1e-12)
    is_polar = jnp.isclose(x, 0.0, atol=1e-12)

    def schwarzschild(a, e, x):
        return 6.0 + 2.0 * e

    def non_schwarzschild(a, e, x):
        a = jnp.where(any_invalid, 0.1, a)
        e = jnp.where(any_invalid, 0.1, e)

        def equatorial(a, e, x):
            def prograde(_):

                x_lo = 1.0 + e
                x_hi = 6.0 + 2.0 * e

                solver = Bisection(rtol=1e-12, atol=1e-12)

                return root_find(
                    _separatrix_polynomial_equat,
                    solver,
                    0.5 * (x_lo + x_hi),
                    (a, e),
                    options=dict(lower=x_lo, upper=x_hi),
                ).value

            def retrograde(_):
                x_lo = 6.0 + 2.0 * e
                x_hi = 5.0 + e + 4.0 * jnp.sqrt(1.0 + e)

                solver = Bisection(
                    rtol=1e-12,
                    atol=1e-12,
                )

                return root_find(
                    _separatrix_polynomial_equat,
                    solver,
                    0.5 * (x_lo + x_hi),
                    (a, e),
                    options=dict(lower=x_lo, upper=x_hi),
                ).value

            return lax.cond(
                a * x > 0.0,
                prograde,
                retrograde,
                operand=None,
            )

        def generic(a, e, x):
            x = jnp.where(any_invalid, 0.1, x)

            # polar separatrix
            polar_lo = (
                1.0
                + jnp.sqrt(3.0)
                + jnp.sqrt(3.0 + 2.0 * jnp.sqrt(3.0))
            )
            polar_hi = 8.0

            polar_solver = Bisection(rtol=1e-12, atol=1e-10)

            polar_p_sep = root_find(
                _separatrix_polynomial_polar,
                polar_solver,
                0.5 * (polar_lo + polar_hi),
                (a, e),
                options=dict(lower=polar_lo, upper=polar_hi),
            ).value

            def polar_case(_):
                return polar_p_sep

            def nonpolar_case(_):

                def positive_x(_):
                    eq_lo = 1.0 + e
                    eq_hi = 6.0 + 2.0 * e

                    eq_solver = Bisection(rtol=1e-12, atol=1e-12)

                    equat_p_sep = root_find(
                        _separatrix_polynomial_equat,
                        eq_solver,
                        0.5 * (eq_lo + eq_hi),
                        (a, e),
                        options=dict(lower=eq_lo, upper=eq_hi),
                    ).value

                    return equat_p_sep, polar_p_sep

                def negative_x(_):
                    return polar_p_sep, 12.0

                x_lo, x_hi = lax.cond(
                    x > 0.0,
                    positive_x,
                    negative_x,
                    operand=None,
                )

                full_solver = Bisection(rtol=1e-10, atol=1e-3)
                return root_find(
                    _separatrix_polynomial_full,
                    full_solver,
                    0.5 * (x_lo + x_hi),
                    (a, e, x),
                    options=dict(lower=x_lo, upper=x_hi),
                ).value
            return lax.cond(
                is_polar,
                polar_case,
                nonpolar_case,
                operand=None,
            )
        return lax.cond(
            is_equat,
            equatorial,
            generic,
            a,
            e,
            x
        )
    result = lax.cond(
        is_schw,
        schwarzschild,
        non_schwarzschild,
        a,
        e,
        x
    )

    return jnp.where(any_nan, jnp.nan, jnp.where(any_inf, jnp.inf, result))


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
    return (pLSO + DELTAPMIN) + (DELTAPMAX - DELTAPMIN) * (np.exp(u ** (1 / ALPHA_FLUX) * jnp.log(2)) - 1)

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
    pLSO = get_separatrix(a, e, 1.)
    u = u_of_p(p, pLSO)
    z = z_of_a(a)
    w = w_of_euz(e, u, z)
    return jnp.asarray([u, w, z])


uwz_of_ape_jac = vmap(jacobian(uwz_of_ape, argnums=(1, 2)), in_axes=(0, 0, 0), out_axes=1)

