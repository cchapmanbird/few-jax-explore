import jax
import jax.numpy as jnp
from functools import partial

@jax.jit
def get_mode_sum_at_t(idx, dt, amplitude_coefficients, phase_coefficients):
    c_a = amplitude_coefficients[idx]
    c_p = phase_coefficients[idx]
    amp   = c_a[0] + dt*(c_a[1] + dt*(c_a[2] + dt*c_a[3]))
    phase = c_p[0] + dt*(c_p[1] + dt*(c_p[2] + dt*c_p[3]))
    co = jnp.cos(phase)
    si = jnp.sin(phase)
    return jnp.sum(amp * (co - 1j * si))

@partial(jax.jit, static_argnames=['batch_size'])
def interpolated_summation_chunked(t_eval, t_knots, amplitude_coefficients, phase_coefficients, batch_size):
    indices = jnp.searchsorted(t_knots, t_eval, side="right") - 1
    dt = t_eval - t_knots[indices]
    out = jax.lax.map(
        lambda idx_dt: get_mode_sum_at_t(
            idx_dt[0], idx_dt[1], amplitude_coefficients, phase_coefficients
            ),
        (indices, dt),
        batch_size=batch_size
    )
    return out