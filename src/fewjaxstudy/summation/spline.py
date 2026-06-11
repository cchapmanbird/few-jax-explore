import jax
import jax.numpy as jnp
import lineax as lx
from functools import partial
from jax import Array

@partial(jax.jit, static_argnums=(3,))
def build_interp_grid_variable(x: Array, y: Array, len: int, maxlen: int) -> tuple[Array, Array, Array, Array]:
    """
    Build grid for cubic spline tridiagonal solver.

    Parameters
    -----------
    - x: array of shape (maxlen,) - x-coordinates of the data points (padded with extra entries if len < maxlen)
    - y: array of shape (maxlen,) - y-coordinates of the data points (padded with extra entries if len < maxlen)
    - len: integer - number of data points
    - maxlen: integer - maximum length of the arrays. Must be static for JIT compilation.

    Returns
    --------
    - diag: array of shape (maxlen,) - diagonal entries of the tridiagonal matrix
    - ud: array of shape (maxlen - 1,) - upper diagonal entries of the
    - ld: array of shape (maxlen - 1,) - lower diagonal entries of the tridiagonal matrix
    - b: array of shape (maxlen,) - right-hand side vector for the linear system
    """
    diag = jnp.full(maxlen, 1.0)
    ud = jnp.full(maxlen - 1, 0.0)
    ld = jnp.full(maxlen - 1, 0.0)

    b = jnp.full(maxlen , 0.0)

    dx = jnp.diff(x)
    slope = jnp.diff(y) / dx

    # Fill interior
    diag = jax.lax.dynamic_update_slice(diag, 2 * (dx[:-1] + dx[1:]), (1, ))
    ud = jax.lax.dynamic_update_slice(ud, dx[:-1], (1, ))
    ld = jax.lax.dynamic_update_slice(ld, dx[1:], (0, ))
    b = jax.lax.dynamic_update_slice(b, 3 * (dx[1:] * slope[:-1] + dx[:-1] * slope[1:]), (1, ))

    # Left BC
    diag = jax.lax.dynamic_update_index_in_dim(diag, dx[1], 0, 0)
    d = x[2] - x[0]
    ud = jax.lax.dynamic_update_index_in_dim(ud, d, 0, 0)
    b = jax.lax.dynamic_update_index_in_dim(b, ((dx[0] + 2*d) * dx[1] * slope[0] +
            dx[0]**2 * slope[1]) / d, 0, 0)

    # Right BC
    diag = jax.lax.dynamic_update_index_in_dim(diag, dx[len-3], len-1, 0)
    d = x[len-1] - x[len-3]
    ld = jax.lax.dynamic_update_index_in_dim(ld, d, len-2, 0)
    b = jax.lax.dynamic_update_index_in_dim(b, (dx[len-2]**2 * slope[len-3] +
             (2*d + dx[len-2]) * dx[len-3] * slope[len-2]) / d, len-1, 0)

    # Pad missing entries to avoid NaNs in the linear solve
    diag = jnp.where(jnp.arange(maxlen) < len, diag, 1.0)
    ud = jnp.where(jnp.arange(maxlen - 1) < len - 1, ud, 0.0)
    ld = jnp.where(jnp.arange(maxlen - 1) < len - 1, ld, 0.0)

    b = jnp.where(jnp.arange(maxlen) < len, b, 0.0)

    return diag, ud, ld, b

@partial(jax.jit, static_argnums=(3,))
def get_spline_coefficients_variable(x: Array, y: Array, len: int, maxlen: int) -> Array:
    """
    Compute cubic spline coefficients for given data points.

    Parameters
    -----------
    - x: array of shape (maxlen,) - x-coordinates of the data points (padded with extra entries if len < maxlen)
    - y: array of shape (maxlen,) - y-coordinates of the data points (padded with extra entries if len < maxlen)
    - len: integer - number of data points
    - maxlen: integer - maximum length of the arrays. Must be static for JIT compilation.

    Returns
    --------
    - coefficients: array of shape (maxlen, 4) - cubic spline coefficients for each interval (padded with extra entries if len < maxlen)
    """
    diag, ud, ld, b = build_interp_grid_variable(x, y, len, maxlen)
    tridiag_operator = lx.TridiagonalLinearOperator(diag, ld, ud)

    dydx = lx.linear_solve(tridiag_operator, b, solver=lx.Tridiagonal()).value

    coefficients = jnp.full((maxlen, 4), jnp.nan)

    dx = jnp.diff(x)
    slope = jnp.diff(y) / dx

    t = (dydx[:-1] + dydx[1:] - 2 * slope) / dx

    coeff1 = dydx
    coeff2 = (slope - dydx[:-1]) / dx - t
    coeff3 = t / dx

    coefficients = coefficients.at[:, 0].set(y)
    coefficients = coefficients.at[:, 1].set(coeff1)
    coefficients = coefficients.at[:-1, 2].set(coeff2)
    coefficients = coefficients.at[:-1, 3].set(coeff3)
    return coefficients

@jax.jit
def evaluate_spline(x: Array, coefficients: Array, x_eval: Array) -> Array:
    """
    Evaluate the cubic spline at given points.

    Parameters
    -----------
    - x: array of shape (n,) - x-coordinates of the data points
    - coefficients: array of shape (n, 4) - cubic spline coefficients for each interval
    - x_eval: array of shape (m,) - points at which to evaluate the spline

    Returns
    --------
    - y_eval: array of shape (m,) - evaluated spline values at the specified points
    """
    idx = jnp.searchsorted(x, x_eval, side="right") - 1
    idx = jnp.clip(idx, 0, x.size - 2)

    dx = x_eval - x[idx]
    coeffs = coefficients[idx]

    return coeffs[:, 0] + coeffs[:, 1] * dx + coeffs[:, 2] * dx**2 + coeffs[:, 3] * dx**3
