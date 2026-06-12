import few
import h5py

import numpy as np
from .utils import KerrGeoEquatorialCoordinateFrequencies

from few.utils.mappings.kerrecceq import apex_of_uwyz
from few.utils.mappings.jacobian import ELdot_to_PEdot_Jacobian
from few.utils.constants import MTSUN_SI, YRSID_SI

import jax
import jax.numpy as jnp
from interpax import Interpolator3D
from .mappings import get_separatrix, uwz_of_ape
from optimistix import Newton, Bisection
from diffrax import Event, Dopri8, SaveAt, PIDController, ODETerm, diffeqsolve, Solution


regionA = h5py.File(few.get_file_manager().get_file("KerrEccEqFluxData.h5"))["regionA"]
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

pdot_intepr = Interpolator3D(u, w, z, pdot, method="cubic2")
edot_intepr = Interpolator3D(u, w, z, edot, method="cubic2")



def cond_fn(t, y, args, **kwargs):
    p, e = y[:2]
    a, = args

    psep = get_separatrix(a, e, 1.)

    return p < (psep + 2e-3)
    
# stop = Event(cond_fn, root_finder=Bisection(rtol=0, atol=1e-10, flip=True))
    
stop = Event(cond_fn, root_finder=Newton(rtol=0, atol=1e-10))

@jax.jit
def RHS(t, y, args):
    a, = args
    p, e = y[:2]
    u, w, z = uwz_of_ape(a, p, e)

    pdot = -pdot_intepr(u, w, z)
    edot = -edot_intepr(u, w, z)

    Omega_phi, Omega_r = KerrGeoEquatorialCoordinateFrequencies(a, p, e, 1.)

    return jnp.array([pdot, edot, Omega_phi, Omega_r])


term = ODETerm(RHS)
solver = Dopri8()
saveat = SaveAt(t0=True, steps=True, dense=True)
stepsize_controller = PIDController(rtol=1e-10,atol=1e-10)

def solve_dynamics(m1: float, m2: float, a: float, p0: float, e0: float, T: float) -> Solution:
    M = m1 + m2
    Msec = M * MTSUN_SI
    M = m1 + m2
    mu = m1 * m2 / M
    nu = mu / M
    T_in = T * YRSID_SI / Msec * nu  # in seconds

    sol = diffeqsolve(term, solver, t0=0, t1=T_in, dt0=None, y0=jnp.asarray([p0, e0, 0., 0.]), saveat=saveat,
                    stepsize_controller=stepsize_controller, args=(a, ), event=stop, throw=True, max_steps=256)
    return sol