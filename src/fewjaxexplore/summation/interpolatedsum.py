from jax import lax, Array
import jax.numpy as jnp
from diffrax import Solution

def get_mode_sum_at_t(idx: int, dt: float, amplitude_coefficients: Array, phase_coefficients: Array) -> Array:
    c_a = amplitude_coefficients[idx]
    c_p = phase_coefficients[idx]
    amp   = c_a[0] + dt*(c_a[1] + dt*(c_a[2] + dt*c_a[3]))
    phase = c_p[0] + dt*(c_p[1] + dt*(c_p[2] + dt*c_p[3]))
    co = jnp.cos(phase)
    si = jnp.sin(phase)
    return jnp.sum(amp * (co - 1j * si))

def interpolated_summation_chunked(t_eval: Array, t_knots: Array, amplitude_coefficients: Array, phase_coefficients: Array, batch_size: int) -> Array:
    """
    Interpolated summation of modes at given evaluation times t_eval, using precomputed spline coefficients for amplitude and phase.
    
    The function processes the evaluation times in batches of size batch_size to manage memory usage, using `lax.map`. This is crucial!

    Parameters
    ----------
    t_eval : Array
        1D array of times at which to evaluate the waveform. Must be sorted in ascending order.
    t_knots : Array
        1D array of knot points corresponding to the spline coefficients. Must be sorted in ascending order.
    amplitude_coefficients : Array
        3D array of shape (num_modes, 4, max_length) containing the amplitude spline coefficients.
    phase_coefficients : Array
        3D array of shape (num_modes, 4, max_length) containing the phase spline coefficients.
    batch_size : int
        Size of batches to process evaluation times in. Must be static at runtime for JIT compilation.
    
    Returns
    -------
    Array
        1D array of complex waveform values at each time in t_eval, computed by summing over modes using the interpolated amplitude and phase at each time.
    """
    indices = jnp.searchsorted(t_knots, t_eval, side="right") - 1
    dt = t_eval - t_knots[indices]
    out = lax.map(
        lambda idx_dt: get_mode_sum_at_t(
            idx_dt[0], idx_dt[1], amplitude_coefficients, phase_coefficients
            ),
        (indices, dt),
        batch_size=batch_size
    )
    return out

def get_mode_sum_at_t_odesol(t: float, idx: int, dt: float, amplitude_coefficients: Array, ode_solution: Solution, mode_indices: Array, Msec: float) -> Array:
    c_a = amplitude_coefficients[idx]
    amp   = c_a[0] + dt*(c_a[1] + dt*(c_a[2] + dt*c_a[3]))
    phase = ((ode_solution.evaluate(t / Msec)[2:,None]) * mode_indices).sum(0)
    co = jnp.cos(phase)
    si = jnp.sin(phase)
    return jnp.sum(amp * (co - 1j * si))

def interpolated_summation_chunked_odesol(t_eval: Array, t_knots: Array, amplitude_coefficients: Array, ode_solution: Solution, mode_indices: Array, Msec: float, batch_size: int) -> Array:
    """
    Interpolated summation of modes at given evaluation times t_eval, using precomputed spline coefficients for amplitude and phase.
    
    The function processes the evaluation times in batches of size batch_size to manage memory usage, using `lax.map`. This is crucial!

    Parameters
    ----------
    t_eval : Array
        1D array of times at which to evaluate the waveform. Must be sorted in ascending order.
    t_knots : Array
        1D array of knot points corresponding to the spline coefficients. Must be sorted in ascending order.
    amplitude_coefficients : Array
        3D array of shape (num_modes, 4, max_length) containing the amplitude spline coefficients.
    ode_solution : Solution
        The solution object from the ODE solver, containing the interpolated state at each time.
    batch_size : int
        Size of batches to process evaluation times in. Must be static at runtime for JIT compilation.
    
    Returns
    -------
    Array
        1D array of complex waveform values at each time in t_eval, computed by summing over modes using the interpolated amplitude and phase at each time.
    """
    indices = jnp.searchsorted(t_knots, t_eval, side="right") - 1
    dt = t_eval - t_knots[indices]
    out = lax.map(
        lambda t_idx_dt: get_mode_sum_at_t_odesol(
            t_idx_dt[0], t_idx_dt[1], t_idx_dt[2], amplitude_coefficients, ode_solution, mode_indices, Msec
            ),
        (t_eval, indices, dt),
        batch_size=batch_size
    )
    return out
