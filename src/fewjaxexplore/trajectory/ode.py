import h5py

import numpy as np
from .utils import KerrGeoEquatorialCoordinateFrequencies

from few.utils.mappings.kerrecceq import apex_of_uwyz
from few.utils.mappings.jacobian import ELdot_to_PEdot_Jacobian
from ..constants import MTSUN_SI, YRSID_SI

import jax
import jax.numpy as jnp
from .mappings import get_separatrix, uwz_of_ape
from optimistix import Newton
from diffrax import Event, Dopri8, SaveAt, PIDController, ODETerm, diffeqsolve, Solution, AbstractAdaptiveSolver
from multispline.spline import TricubicSpline
from .spline import get_interpolant_for_coefficients

def cond_fn(t, y, args, **kwargs):
    p, e = y[:2]
    a, = args

    psep = get_separatrix(a, e, 1.)

    return p < (psep + 2e-3)
    
stop = Event(cond_fn, root_finder=Newton(rtol=0, atol=1e-10))

def get_trajectory_generator(
        filepath: str, 
        rtol: float = 1e-10, 
        atol: float = 1e-10,
        solver: AbstractAdaptiveSolver = Dopri8(),
        max_steps: int = 256
    ):
    regionA = h5py.File(filepath)["regionA"]
    Edot = regionA["Edot"][()]
    Ldot = regionA["Ldot"][()]

    u = np.linspace(0, 1, regionA.attrs["NU"])
    w = np.linspace(0, 1, regionA.attrs["NW"])
    z = np.linspace(0, 1, regionA.attrs["NZ"])

    ugrid, wgrid, zgrid = np.asarray(
        np.meshgrid(u, w, z, indexing="ij")
    ).reshape(3, -1)
    agrid, pgrid, egrid, xgrid = apex_of_uwyz(
        ugrid, wgrid, np.ones_like(zgrid), zgrid
    )


    Edothere = (Edot).flatten()
    Ldothere = (Ldot).flatten()
    xgrid = np.sign(agrid)
    xgrid[xgrid == 0] = 1

    Ldothere = Ldothere * xgrid
    agrid = np.abs(agrid)

    out_pdot_edot = np.asarray(
        [
            ELdot_to_PEdot_Jacobian(
                agrid[i],
                pgrid[i],
                egrid[i],
                xgrid[i],
                Edothere[i],
                Ldothere[i],
            )
            for i in range(Edothere.size)
        ]
    )

    # check whether there are no nans in the output and Edot and Ldot
    if (
        np.isnan(out_pdot_edot).any()
        or np.isnan(Edot).any()
        or np.isnan(Ldot).any()
    ):
        raise ValueError("Interpolation: nans in pdot, edot or Edot, Ldot.")

    pdot = out_pdot_edot[:, 0].reshape(u.size, w.size, z.size)
    edot = out_pdot_edot[:, 1].reshape(u.size, w.size, z.size)

    multispline_pdot = TricubicSpline(u, w, z, pdot)
    coefficients = jnp.asarray(multispline_pdot.coefficients.reshape(u.size - 1, w.size - 1, z.size - 1, 4, 4, 4))
    pdot_interp = get_interpolant_for_coefficients(
        u[1] - u[0], w[1] - w[0], z[1] - z[0], coefficients
    )

    multispline_edot = TricubicSpline(u, w, z, edot)
    coefficients_edot = jnp.asarray(multispline_edot.coefficients.reshape(u.size - 1, w.size - 1, z.size - 1, 4, 4, 4))
    edot_interp = get_interpolant_for_coefficients(
        u[1] - u[0], w[1] - w[0], z[1] - z[0], coefficients_edot
    )

    @jax.jit
    def RHS(t, y, args):
        a, = args
        p, e = y[:2]
        u, w, z = uwz_of_ape(a, p, e)

        pdot = -pdot_interp(u, w, z)
        edot = -edot_interp(u, w, z)

        Omega_phi, Omega_r = KerrGeoEquatorialCoordinateFrequencies(a, p, e, 1.)

        return jnp.array([pdot, edot, Omega_phi, Omega_r])


    term = ODETerm(RHS)
    saveat = SaveAt(t0=True, steps=True, dense=True)
    stepsize_controller = PIDController(rtol=rtol, atol=atol)

    def solve_dynamics(m1: float, m2: float, a: float, p0: float, e0: float, Phi_phi0: float, Phi_r0: float, T: float) -> Solution:
        M = m1 + m2
        Msec = M * MTSUN_SI
        M = m1 + m2
        mu = m1 * m2 / M
        nu = mu / M
        T_in = T * YRSID_SI / Msec * nu  # in seconds

        sol = diffeqsolve(term, solver, t0=0, t1=T_in, dt0=None, y0=jnp.asarray([p0, e0, Phi_phi0, Phi_r0]), saveat=saveat,
                        stepsize_controller=stepsize_controller, args=(a, ), event=stop, throw=True, max_steps=max_steps)
        return sol
    
    return solve_dynamics
