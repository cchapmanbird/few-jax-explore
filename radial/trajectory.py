import jax
jax.config.update("jax_enable_x64", True)
from jax import Array, jit
import jax.numpy as jnp
import h5py
import numpy as np

from fewjaxexplore.trajectory.utils import KerrGeoEquatorialCoordinateFrequencies
from optimistix import Newton
from diffrax import Event, Dopri8, SaveAt, PIDController, ODETerm, diffeqsolve, Solution, AbstractAdaptiveSolver
from multispline.spline import BicubicSpline
from fewjaxexplore.constants import YRSID_SI, MTSUN_SI

def pe_to_uw(p, e):
    dp = p - (6 + 2 * e)
    u = (jnp.log10(dp) + 2) / (jnp.log10(13) + 2)
    w = e
    return u, w

def Enorm(p, e, OmegaR):
    Tr = 2 * np.pi / OmegaR
    leadingPN = 32.0 / 5.0 * p ** (-5)

    return Tr / leadingPN

def Lnorm(p, e, OmegaR):
    Tr = 2 * np.pi / OmegaR
    leadingPN = 32.0 / 5.0 * p ** (-3.5)

    return Tr / leadingPN

def get_interpolant_for_coefficients(
    dx: float, dy: float, coefficients: Array
):
    @jit
    def interpolant(x, y):
        tx = x / dx
        ix = jnp.floor(tx).astype(int)
        ty = y / dy
        iy = jnp.floor(ty).astype(int)
        x_diff = tx - ix
        y_diff = ty - iy

        c = coefficients[ix, iy]  # shape (4, 4)

        # Horner over y
        cy = c[:, 3]
        for j in range(2, -1, -1):
            cy = c[:, j] + y_diff * cy  # shape (4,)

        # Horner over x
        result = cy[3]
        for i in range(2, -1, -1):
            result = cy[i] + x_diff * result  # scalar

        return result    
    return interpolant

@jit
def _schwarz_jac_kernel(p, e, Edot, Ldot):
    pdot = (
        -2
        * (
            Edot
            * jnp.sqrt((4 * pow(e, 2) - pow(-2 + p, 2)) / (3 + pow(e, 2) - p))
            * (3 + pow(e, 2) - p)
            * pow(p, 1.5)
            + Ldot * pow(-4 + p, 2) * jnp.sqrt(-3 - pow(e, 2) + p)
        )
    ) / (4 * pow(e, 2) - pow(-6 + p, 2))
    edot = jnp.where(
        e > 0,
        -(
        (
            Edot
            * jnp.sqrt((4 * pow(e, 2) - pow(-2 + p, 2)) / (3 + pow(e, 2) - p))
            * pow(p, 1.5)
            * (18 + 2 * pow(e, 4) - 3 * pow(e, 2) * (-4 + p) - 9 * p + pow(p, 2))
            + (-1 + pow(e, 2))
            * Ldot
            * jnp.sqrt(-3 - pow(e, 2) + p)
            * (12 + 4 * pow(e, 2) - 8 * p + pow(p, 2))
        )
        / (e * (4 * pow(e, 2) - pow(-6 + p, 2)) * p)
        ),
        0.0
        )
    return pdot, edot


def cond_fn(t, y, args, **kwargs):
    p, e = y[:2]
    psep = 6 + 2 * e
    return p < (psep + 2e-2)

stop = Event(cond_fn, root_finder=Newton(rtol=0, atol=1e-10))

def get_trajectory_generator(
        filepath: str, 
        rtol: float = 1e-10, 
        atol: float = 1e-10,
        solver: AbstractAdaptiveSolver = Dopri8(),
        max_steps: int = 256,
        return_interpolants: bool = False
    ):
    f = h5py.File(filepath)
    e_loaded     = np.array(f["e"][:])
    dp_loaded    = (np.log10(np.array(f["dp"][:])) + 2) / (np.log10(13) + 2)
    EdotNorm_loaded   = np.array(f["EdotNorm"][:])   # (n_dp, n_e, N_psi)
    LdotNorm_loaded = np.array(f["LdotNorm"][:])

    multispline_Edot = BicubicSpline(dp_loaded, e_loaded, EdotNorm_loaded)
    coefficients_Edot = jnp.asarray(multispline_Edot.coefficients.reshape(dp_loaded.size - 1, e_loaded.size - 1, 4, 4))
    multispline_Ldot = BicubicSpline(dp_loaded, e_loaded, LdotNorm_loaded)
    coefficients_Ldot = jnp.asarray(multispline_Ldot.coefficients.reshape(dp_loaded.size - 1, e_loaded.size - 1, 4, 4))

    coefficients = jnp.stack([coefficients_Edot, coefficients_Ldot], axis=-1)
    interpolants = get_interpolant_for_coefficients(
        dp_loaded[1] - dp_loaded[0], e_loaded[1] - e_loaded[0], coefficients
    )

    @jit
    def RHS(t, y, args):
        nu, = args
        p, e = y[:2]

        u, w = pe_to_uw(p, e)

        EdotNorm, LdotNorm = interpolants(u, w)
        Omega_phi, Omega_r = KerrGeoEquatorialCoordinateFrequencies(0.0, p, e, 1.)
        Edot = - nu * (EdotNorm / Enorm(p, e, Omega_r))
        Ldot = - nu * (LdotNorm / Lnorm(p, e, Omega_r))
        pdot, edot = _schwarz_jac_kernel(p, e, Edot, Ldot)
        return jnp.array([pdot, edot, Omega_phi, Omega_r])

    term = ODETerm(RHS)
    saveat = SaveAt(t0=True, steps=True, dense=True)

    def solve_dynamics(m1: float, m2: float, p0: float, e0: float, Phi_phi0: float, Phi_r0: float, T: float) -> Solution:
        M = m1 + m2
        nu =  m1 * m2 / M**2
        T_in = T * YRSID_SI / (M * MTSUN_SI)  # in seconds

        stepsize_controller = PIDController(rtol=rtol, atol=jnp.full(4, atol) / jnp.array([1, 1, nu, nu]))
        sol = diffeqsolve(term, solver, t0=0, t1=T_in, dt0=None, y0=jnp.array([p0, e0, Phi_phi0, Phi_r0]), saveat=saveat,
                        stepsize_controller=stepsize_controller, args=(nu,), event=stop, throw=True, max_steps=max_steps)
        return sol
    
    if return_interpolants:
        return solve_dynamics, interpolants
    else:
       return solve_dynamics
