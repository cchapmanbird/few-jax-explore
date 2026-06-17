"""
Elliptic function routines written in jax format
"""

from .mappings import get_separatrix, uwz_of_ape
import jax.numpy as jnp
import jax
from jax import jit

def _pdot_PN(p, e, r_isco, p_sep):
    """Leading-order ṗ PN factor (JAX, ODE runtime)."""
    return 8.0 * (1.0 - e ** 2) ** 1.5 * (8.0 + 7.0 * e ** 2) / (5.0 * p * ((p - r_isco) ** 2 - (p_sep - r_isco) ** 2))


def _edot_PN(p, e, r_isco, p_sep):
    """Leading-order ė PN factor (JAX, ODE runtime)."""
    return  (1.0 - e ** 2) ** 1.5 * (304.0 + 121.0 * e ** 2) / (15.0 * p ** 2 * ((p - r_isco) ** 2 - (p_sep - r_isco) ** 2))

@jax.jit
def rc_body_fun(i, args):
    return jax.lax.cond(i == 8, rc_true_fun, rc_false_fun, args)

@jax.jit
def rc_true_fun(args):
    xn, yn = args
    mu = ( xn + yn + yn ) / 3.0
    sn = ( yn + mu ) / mu - 2.0

    c1 = 1.0 / 7.0
    c2 = 9.0 / 22.0
    s = sn * sn * ( 0.3 + sn * ( c1 + sn * ( 0.375 + sn * c2 ) ) )
    value = ( 1.0 + s ) / jnp.sqrt ( mu )
    return jnp.array([value, 0.])

@jax.jit
def rc_false_fun(args):
    xn, yn = args

    lamda = 2.0 * jnp.sqrt ( xn ) * jnp.sqrt ( yn ) + yn
    xn = ( xn + lamda ) * 0.25
    yn = ( yn + lamda ) * 0.25
    return jnp.array([xn, yn])
  
@jax.jit
def jax_rc(x,y):
    xn = x
    yn = y
    return jax.lax.fori_loop(0, 9, rc_body_fun, jnp.array([xn, yn,]))[0]

@jax.jit
def rf_body_fun(i, args):
    return jax.lax.cond(i == 8, rf_true_fun, rf_false_fun, args)

@jax.jit
def rf_true_fun(args):
    xn, yn, zn = args
    mu = ( xn + yn + zn ) / 3.0
    xndev = 2.0 - ( mu + xn ) / mu
    yndev = 2.0 - ( mu + yn ) / mu
    zndev = 2.0 - ( mu + zn ) / mu

    c1 = 1.0 / 24.0
    c2 = 3.0 / 44.0
    c3 = 1.0 / 14.0
    e2 = xndev * yndev - zndev * zndev
    e3 = xndev * yndev * zndev
    s = 1.0 + ( c1 * e2 - 0.1 - c2 * e3 ) * e2 + c3 * e3
    value = s / jnp.sqrt ( mu )

    return jnp.array([value, 0., 0.])

@jax.jit
def rf_false_fun(args):
    xn, yn, zn = args

    xnroot = jnp.sqrt ( xn )
    ynroot = jnp.sqrt ( yn )
    znroot = jnp.sqrt ( zn )
    lamda = xnroot * ( ynroot + znroot ) + ynroot * znroot
    xn = ( xn + lamda ) * 0.25
    yn = ( yn + lamda ) * 0.25
    zn = ( zn + lamda ) * 0.25

    return jnp.array([xn, yn, zn])
  
@jax.jit
def jax_rf(x,y, z):
    xn = x
    yn = y
    zn = z
    return jax.lax.fori_loop(0, 9, rf_body_fun, jnp.array([xn, yn,zn]))[0]

@jax.jit
def rj_body_fun(i, args):
    return jax.lax.cond(i == 8, rj_true_fun, rj_false_fun, args)

@jax.jit
def rj_true_fun(args):
    xn, yn, zn, pn, sigma, power4 = args
    mu = ( xn + yn + zn + pn + pn ) * 0.2
    xndev = ( mu - xn ) / mu
    yndev = ( mu - yn ) / mu
    zndev = ( mu - zn ) / mu
    pndev = ( mu - pn ) / mu

    c1 = 3.0 / 14.0
    c2 = 1.0 / 3.0
    c3 = 3.0 / 22.0
    c4 = 3.0 / 26.0
    ea = xndev * ( yndev + zndev ) + yndev * zndev
    eb = xndev * yndev * zndev
    ec = pndev * pndev
    e2 = ea - 3.0 * ec
    e3 = eb + 2.0 * pndev * ( ea - ec )
    s1 = 1.0 + e2 * ( - c1 + 0.75 * c3 * e2 - 1.5 * c4 * e3 )
    s2 = eb * ( 0.5 * c2 + pndev * ( - c3 - c3 + pndev * c4 ) )
    s3 = pndev * ea * ( c2 - pndev * c3 ) - c2 * pndev * ec
    value = 3.0 * sigma + power4 * ( s1 + s2 + s3 ) / ( mu * jnp.sqrt ( mu ) )

    return jnp.array([value, 0., 0., 0., 0., 0.])

@jax.jit
def rj_false_fun(args):
    xn, yn, zn, pn, sigma, power4 = args

    xnroot = jnp.sqrt ( xn )
    ynroot = jnp.sqrt ( yn )
    znroot = jnp.sqrt ( zn )
    lamda = xnroot * ( ynroot + znroot ) + ynroot * znroot
    alfa = pn * ( xnroot + ynroot + znroot ) + xnroot * ynroot * znroot
    alfa = alfa * alfa
    beta = pn * ( pn + lamda ) * ( pn + lamda )
    value = jax_rc(alfa, beta)#, lamda)
    sigma = sigma + power4 * value
    power4 = power4 * 0.25
    xn = ( xn + lamda ) * 0.25
    yn = ( yn + lamda ) * 0.25
    zn = ( zn + lamda ) * 0.25
    pn = ( pn + lamda ) * 0.25
    return jnp.array([xn, yn, zn, pn, sigma, power4])

@jax.jit
def jax_rj(x,y,z,p):
    xn = x
    yn = y
    zn = z
    pn = p
    sigma = 0.0
    power4 = 1.0
    return jax.lax.fori_loop(0, 9, rj_body_fun, jnp.array([xn, yn, zn, pn, sigma, power4]))[0]

@jax.jit
def rd_body_fun(i, args):
    return jax.lax.cond(i == 8, rd_true_fun, rd_false_fun, args)

@jax.jit
def rd_true_fun(args):
    xn, yn, zn, sigma, power4 = args
    mu = ( xn + yn + 3.0 * zn ) * 0.2
    xndev = ( mu - xn ) / mu
    yndev = ( mu - yn ) / mu
    zndev = ( mu - zn ) / mu
    c1 = 3.0 / 14.0
    c2 = 1.0 / 6.0
    c3 = 9.0 / 22.0
    c4 = 3.0 / 26.0
    ea = xndev * yndev
    eb = zndev * zndev
    ec = ea - eb
    ed = ea - 6.0 * eb
    ef = ed + ec + ec
    s1 = ed * ( - c1 + 0.25 * c3 * ed - 1.5 * c4 * zndev * ef )
    s2 = zndev  * ( c2 * ef + zndev * ( - c3 * ec + zndev * c4 * ea ) )
    value = 3.0 * sigma  + power4 * ( 1.0 + s1 + s2 ) / ( mu * jnp.sqrt ( mu ) )

    return jnp.array([value, 0., 0., 0., 0.])

@jax.jit
def rd_false_fun(args):
    xn, yn, zn, sigma, power4 = args

    xnroot = jnp.sqrt ( xn )
    ynroot = jnp.sqrt ( yn )
    znroot = jnp.sqrt ( zn )
    lamda = xnroot * ( ynroot + znroot ) + ynroot * znroot
    sigma = sigma + power4 / ( znroot * ( zn + lamda ) )
    power4 = power4 * 0.25
    xn = ( xn + lamda ) * 0.25
    yn = ( yn + lamda ) * 0.25
    zn = ( zn + lamda ) * 0.25
    return jnp.array([xn, yn, zn, sigma, power4])

@jax.jit
def jax_rd(x,y,z):
    xn = x
    yn = y
    zn = z
    sigma = 0.0
    power4 = 1.0
    # value, d1, d2, d3, d4, = jax.lax.while_loop(rd_cond_fun, rd_body_fun, jnp.array([xn, yn, zn, sigma, power4]))
    return jax.lax.fori_loop(0, 9, rd_body_fun, jnp.array([xn, yn, zn, sigma, power4]))[0]

# @jax.custom_jvp
@jax.jit
def jax_elliptic_pim ( n, m ):

  x = 0.0
  y = 1.0 - m
  z = 1.0
  p = 1.0 - n

  value1 = jax_rf ( x, y, z)
  value2 = jax_rj ( x, y, z, p)

  value = value1 + n * value2 / 3.0

  return value

# jax_elliptic_pim.defjvps(
#     lambda n_dot, primal_out, n, m: 1 / (2 * (m - n) * (n - 1)) * (jax_elliptic_em(m) + 1/n * ((m - n) * jax_elliptic_km(m) + (n**2 - m)*primal_out)) * n_dot,
#     lambda m_dot, primal_out, n, m: 1 / (2 * (n - m))*(jax_elliptic_em(m) / (m-1) + primal_out) * m_dot
# )

# @jax.custom_jvp
@jax.jit
def jax_elliptic_em ( m ):

  x = 0.0
  y = 1.0 - m
  z = 1.0

  value1 = jax_rf ( x, y, z )
  value2 = jax_rd ( x, y, z )
  value = value1 - m * value2 / 3.0

  return value

# @jax_elliptic_em.defjvp
# def jax_elliptic_em_jvp(primals, tangents):
#     m, = primals
#     mdot, = tangents
#     em = jax_elliptic_em(m)
#     primal_out = em
#     tangent_out = (1 / 2 / m) * (em - jax_elliptic_km(m)) * mdot
#     return primal_out, tangent_out

# @jax.custom_jvp
@jax.jit
def jax_elliptic_km ( m ):

  x = 0.0
  y = 1.0 - m
  z = 1.0

  return jax_rf ( x, y, z )

# @jax_elliptic_km.defjvp
# def jax_elliptic_km_jvp(primals, tangents):
#     m, = primals
#     mdot, = tangents
#     km = jax_elliptic_km(m)
#     primal_out = km
#     tangent_out = (1 / 2 / m) * (jax_elliptic_em(m) / (1 - m) - km) * mdot
#     return primal_out, tangent_out


@jit
def _KerrGeoRadialRoots(a, p, e, En, Q):
    r1 = p / (1 - e)
    r2 = p / (1 + e)
    AplusB = (2) / (1 - (En * En)) - (r1 + r2)
    AB = ((a * a) * Q) / ((1 - (En * En)) * r1 * r2)
    r3 = (AplusB + jnp.sqrt((AplusB * AplusB) - 4 * AB)) / 2
    r4 = AB / r3

    return r1, r2, r3, r4

@jit
def _KerrGeoEnergy(a, p, e, x):
    denom = (
        -4.0 * (a * a) * ((-1 + (e * e)) * (-1 + (e * e)))
        + ((3 + (e * e) - p) * (3 + (e * e) - p)) * p
    )
    numer = (-1 + (e * e)) * (
        (a * a) * (1 + 3 * (e * e) + p)
        + p
        * (
            -3
            - (e * e)
            + p
            - x
            * 2
            * jnp.sqrt(
                (
                    (a * a * a * a * a * a) * ((-1 + (e * e)) * (-1 + (e * e)))
                    + (a * a) * (-4 * (e * e) + ((-2 + p) * (-2 + p))) * (p * p)
                    + 2 * (a * a * a * a) * p * (-2 + p + (e * e) * (2 + p))
                )
                / (p * p * p)
            )
        )
    )

    ratio = numer / denom

    return jnp.sqrt(1.0 - ((1.0 - (e * e)) * (1.0 + ratio)) / p)


@jit
def _KerrGeoAngularMomentum(a, p, e, x, En):
    r1 = p / (1 - e)
    zm = jnp.sqrt(1 - (x * x))

    return (
        -En * _g(r1, a, zm)
        + x *
        jnp.sqrt(
            (
                -_d(r1, a, zm) * _h(r1, a, zm)
                + (En * En) * ((_g(r1, a, zm)**2) + _f(r1, a, zm) * _h(r1, a, zm))
            )
        )
    ) / _h(r1, a, zm)


@jit
def _CapitalDelta(r, a):
    return (r * r) - 2.0 * r + (a * a)


@jit
def _f(r, a, zm):
    return (r * r * r * r) + (a * a) * (r * (r + 2.0) + (zm * zm) * _CapitalDelta(r, a))


@jit
def _g(r, a, zm):
    return 2.0 * a * r


@jit
def _h(r, a, zm):
    return r * (r - 2.0) + (zm * zm) / (1.0 - (zm * zm)) * _CapitalDelta(r, a)


@jit
def _d(r, a, zm):
    return ((r * r) + (a * a) * (zm * zm)) * _CapitalDelta(r, a)

@jit
def pedot_PN(a, p, e, x):
    risco = get_separatrix(a, 0.0, x)
    p_sep = get_separatrix(a, e, x)

    pdot = (8.0 * (1.0 - (e * e))** 1.5 * (8.0 + 7.0 * (e * e))) / (
        5.0 * p * (((p - risco) * (p - risco)) - ((-risco + p_sep) * (-risco + p_sep)))
    )
    edot = (((1.0 - (e * e)) ** 1.5) * (304.0 + 121.0 * (e * e))) / (
        15.0
        * (p * p)
        * (((p - risco) * (p - risco)) - ((-risco + p_sep) * (-risco + p_sep)))
    )

    return jnp.asarray([pdot, edot])

pedot_PN_dot = jax.vmap(jax.jacobian(pedot_PN, argnums=(1, 2)), in_axes=(0, 0, 0, 0), out_axes=1)

EllipPi = jax_elliptic_pim
EllipK = jax_elliptic_km
EllipE = jax_elliptic_em

@jit
def _KerrGeoEquatorialMinoFrequencies(a, p, e, x):
    M = 1.0

    En = _KerrGeoEnergy(a, p, e, x)
    L = _KerrGeoAngularMomentum(a, p, e, x, En)

    r1, r2, r3, r4 = _KerrGeoRadialRoots(a, p, e, En, 0.0)

    # Epsilon0 = (a * a) * (1 - (En * En)) / (L * L)
    # a2zp = ((L * L) + (a * a) * (-1 + (En * En)) * (-1)) / ((-1 + (En * En)) * (-1))
    Epsilon0zp = -(((L * L) + (a * a) * (-1 + (En * En)) * (-1)) / ((L * L) * (-1)))

    zp = (a * a) * (1 - (En * En)) + (L * L)

    kr = jnp.sqrt((r1 - r2) / (r1 - r3) * (r3 - r4) / (r2 - r4))  # (*Eq.(13)*)

    EllK = EllipK(kr**2)
    CapitalUpsilonr = (jnp.pi * jnp.sqrt((1 - (En * En)) * (r1 - r3) * (r2))) / (
        2 * EllK
    )  # (*Eq.(15)*)
    CapitalUpsilonTheta = x * pow(zp, 0.5)  # (*Eq.(15)*)

    rp = M + jnp.sqrt(1.0 - (a * a))
    rm = M - jnp.sqrt(1.0 - (a * a))

    hr = (r1 - r2) / (r1 - r3)
    hp = ((r1 - r2) * (r3 - rp)) / ((r1 - r3) * (r2 - rp))
    hm = ((r1 - r2) * (r3 - rm)) / ((r1 - r3) * (r2 - rm))

    EllipPi_hr_kr = EllipPi(hr, kr**2)
    EllipPi_hp_kr = EllipPi(hp, kr**2)
    EllipPi_hm_kr = EllipPi(hm, kr**2)

    prob1 = (2 * M * En * rp - a * L) * (EllK - (r2 - r3) / (r2 - rp) * EllipPi_hp_kr)

    # This term is zero when r3 - rp == 0.0
    # if abs(prob1) != 0.0:
    #     prob1 = prob1 / (r3 - rp)
    prob1 = prob1 / (r3 - rp)
    CapitalUpsilonPhi = (CapitalUpsilonTheta) / (jnp.sqrt(Epsilon0zp)) + (
        2 * a * CapitalUpsilonr
    ) / (jnp.pi * (rp - rm) * jnp.sqrt((1 - (En * En)) * (r1 - r3) * (r2 - r4))) * (
        prob1
        - (2 * M * En * rm - a * L)
        / (r3 - rm)
        * (EllK - (r2 - r3) / (r2 - rm) * EllipPi_hm_kr)
    )

    # This term is zero when r3 - rp == 0.0
    prob2 = ((4 * 1.0 * En - a * L) * rp - 2 * M * (a * a) * En) * (
        EllK - (r2 - r3) / (r2 - rp) * EllipPi_hp_kr
    )
    # if abs(prob2) != 0.0:
    #     prob2 = prob2 / (r3 - rp)
    prob2 = prob2 / (r3 - rp)

    CapitalGamma = 4 * 1.0 * En + (2 * CapitalUpsilonr) / (
        jnp.pi * jnp.sqrt((1 - (En * En)) * (r1 - r3) * (r2 - r4))
    ) * (
        En
        / 2
        * (
            (r3 * (r1 + r2 + r3) - r1 * r2) * EllK
            + (r2 - r3) * (r1 + r2 + r3 + r4) * EllipPi_hr_kr
            + (r1 - r3) * (r2 - r4) * EllipE(kr**2)
        )
        + 2 * M * En * (r3 * EllK + (r2 - r3) * EllipPi_hr_kr)
        + (2 * M)
        / (rp - rm)
        * (
            prob2
            - ((4 * 1.0 * En - a * L) * rm - 2 * M * (a * a) * En)
            / (r3 - rm)
            * (EllK - (r2 - r3) / (r2 - rm) * EllipPi_hm_kr)
        )
    )

    return jnp.asarray([CapitalGamma, CapitalUpsilonPhi, CapitalUpsilonr])

@jit
def KerrGeoEquatorialCoordinateFrequencies(a, p, e, x):
    Gamma, UpsilonPhi, UpsilonR = _KerrGeoEquatorialMinoFrequencies(a, p, e, x)
    return jnp.asarray([UpsilonPhi / Gamma, UpsilonR / Gamma])


def KerrGeoEquatorialCoordinateFrequencyTimeDerivatives(pdot_interp, edot_interp, a, p, e):
    # chain rule via fluxes
    risco = get_separatrix(a, 0.0, 1.0)
    p_sep = get_separatrix(a, e, 1.0)
    u, w, z = uwz_of_ape(a, p, e)
    pdot = - pdot_interp(u, w, z) * _pdot_PN(p, e, risco, p_sep)
    edot = - edot_interp(u, w, z) * _edot_PN(p, e, risco, p_sep)

    (dOmegaphi_dp, dOmegar_dp), (dOmegaphi_de, dOmegar_de) = jax.jacobian(KerrGeoEquatorialCoordinateFrequencies, argnums=(1, 2,))(a, p, e, 1.0)

    dOmegaphi_dt = dOmegaphi_dp * pdot + dOmegaphi_de * edot
    dOmegar_dt = dOmegar_dp * pdot + dOmegar_de * edot
    return dOmegaphi_dt, dOmegar_dt