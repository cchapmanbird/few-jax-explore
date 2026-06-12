from .trajectory.ode import solve_dynamics
from .amplitude.ylm import SpinWeightedSphericalHarmonic
# from .amplitude.ampinterp import get_amplitudes

from .summation.spline import get_spline_coefficients_variable

from few.utils.constants import YRSID_SI, MTSUN_SI

import jax
import jax.numpy as jnp
from jax import Array
from functools import partial

# @jax.jit
def build_waveform(get_amplitudes, t_eval: Array, m1: float, m2: float, a: float, p0: float, e0: float, theta: float, T: float, l_modes: Array, m_modes: Array, n_modes: Array) -> Array:
    M = m1 + m2
    Msec = M * MTSUN_SI
    M = m1 + m2
    mu = m1 * m2 / M
    nu = mu / M

    ylms = jax.vmap(SpinWeightedSphericalHarmonic, in_axes=(0, 0, None, None))(l_modes, m_modes, theta, -jnp.pi/2)
    sol = solve_dynamics(m1, m2, a, p0, e0, T)
    length = sol.stats["num_accepted_steps"]
    MAXLEN = 256 + 1

    t = sol.ts * Msec / nu  # convert back to seconds
    p, e, Phi_phi, Phi_r = sol.ys.T

    amplitudes = (jax.vmap(get_amplitudes, in_axes=(None, 0, 0, None, None, None))(a, p, e, l_modes, m_modes, n_modes) * ylms[None, :]).T
    phases = (Phi_phi[None,:] * m_modes[:, None] + Phi_r[None,:] * n_modes[:, None]) / nu + jnp.unwrap(jnp.angle(amplitudes), axis=-1)
    amplitudes = jnp.abs(amplitudes)

    amplitude_coefficients = jax.vmap(get_spline_coefficients_variable, in_axes=(None, 0, None, None))(t, amplitudes, length, MAXLEN)
    phase_coefficients = jax.vmap(get_spline_coefficients_variable, in_axes=(None, 0, None, None))(t, phases, length, MAXLEN)
    return t, length, amplitude_coefficients, phase_coefficients

    