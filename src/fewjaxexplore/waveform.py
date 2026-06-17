from .amplitude.ylm import SpinWeightedSphericalHarmonic
from .summation.spline import get_spline_coefficients_variable
from .summation.interpolatedsum import interpolated_summation_chunked, interpolated_summation_chunked_odesol
from .constants import MTSUN_SI, Gpc, MRSUN_SI

import jax
import jax.numpy as jnp
from jax import Array

def construct_amplitude_phase_splines(traj_gen, amp_gen, m1: float, m2: float, a: float, p0: float, e0: float, Phi_phi0: float, Phi_r0: float, dist: float, theta: float, T: float, l_modes: Array, m_modes: Array, n_modes: Array) -> tuple[Array, Array, Array, Array]:
    M = m1 + m2
    Msec = M * MTSUN_SI
    M = m1 + m2
    mu = m1 * m2 / M

    ylms = jax.vmap(SpinWeightedSphericalHarmonic, in_axes=(0, 0, None, None))(l_modes, m_modes, theta, -jnp.pi/2)
    sol = traj_gen(m1, m2, a, p0, e0, Phi_phi0, Phi_r0, T)
    length = sol.stats["num_accepted_steps"]
    MAXLEN = 256 + 1

    t = sol.ts * Msec  # convert back to seconds
    p, e, Phi_phi, Phi_r = sol.ys.T

    amplitudes = (jax.vmap(amp_gen, in_axes=(None, 0, 0, None, None, None))(a, p, e, l_modes, m_modes, n_modes) * ylms[None, :]).T
    phases = (Phi_phi[None,:] * m_modes[:, None] + Phi_r[None,:] * n_modes[:, None]) + jnp.unwrap(jnp.angle(amplitudes), axis=-1)
    amplitudes = jnp.abs(amplitudes) * (mu * MRSUN_SI  / dist / Gpc)

    amplitude_coefficients = jax.vmap(get_spline_coefficients_variable, in_axes=(None, 0, None, None))(t, amplitudes, length, MAXLEN)
    phase_coefficients = jax.vmap(get_spline_coefficients_variable, in_axes=(None, 0, None, None))(t, phases, length, MAXLEN)
    return t, length, amplitude_coefficients.transpose(1, 2, 0), phase_coefficients.transpose(1, 2, 0)

def construct_amplitudes_sol(traj_gen, amp_gen, m1: float, m2: float, a: float, p0: float, e0: float, Phi_phi0: float, Phi_r0: float, dist: float, theta: float, T: float, l_modes: Array, m_modes: Array, n_modes: Array) -> tuple[Array, Array, Array, Array]:
    M = m1 + m2
    Msec = M * MTSUN_SI
    mu = m1 * m2 / M

    ylms = jax.vmap(SpinWeightedSphericalHarmonic, in_axes=(0, 0, None, None))(l_modes, m_modes, theta, -jnp.pi/2)
    sol = traj_gen(m1, m2, a, p0, e0, Phi_phi0, Phi_r0, T)
    length = sol.stats["num_accepted_steps"]
    MAXLEN = 256 + 1

    t = sol.ts * Msec  # convert back to seconds
    p, e, Phi_phi, Phi_r = sol.ys.T

    amplitudes = (jax.vmap(amp_gen, in_axes=(None, 0, 0, None, None, None))(a, p, e, l_modes, m_modes, n_modes) * ylms[None, :]).T
    amplitudes = amplitudes * (mu * MRSUN_SI  / dist / Gpc)

    amplitude_coefficients = jax.vmap(get_spline_coefficients_variable, in_axes=(None, 0, None, None))(t, amplitudes, length, MAXLEN)
    return t, length, amplitude_coefficients.transpose(1, 2, 0), sol


def build_waveform_from_coefficients(t_eval, t_knots, amplitude_coefficients, phase_coefficients, batch_size):
    return interpolated_summation_chunked(
        t_eval, t_knots, amplitude_coefficients, phase_coefficients, batch_size)

def build_waveform_from_coefficients_and_dense_solution(t_eval, t_knots, amplitude_coefficients, ode_solution, mode_indices, Msec, batch_size):
    return interpolated_summation_chunked_odesol(
        t_eval, t_knots, amplitude_coefficients, ode_solution, mode_indices, Msec, batch_size)
        