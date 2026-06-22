import jax
jax.config.update("jax_enable_x64", True)
from jax import lax
import jax.numpy as jnp
import h5py
from ndsplines import make_interp_spline
import numpy as np
from functools import partial

def fpbspl(t, k, x, l, h):
    """De Boor-Cox stable recurrence for B-spline basis values on arbitrary knots.

    Fills h[0..k] with the k+1 nonzero B-spline basis function values at x,
    where l is the knot-interval index such that t[l-1] <= x < t[l].

    `k` must be a static Python int. `t`, `x`, `l` may be traced.
    `h` is an input array (e.g. zeros) and the updated array is returned.
    """
    hh = jnp.zeros(k)
    h = h.at[0].set(1.0)
    for j in range(1, k + 1):
        hh = hh.at[:j].set(h[:j])
        h = h.at[0].set(0.0)
        for i in range(j):
            li = l + i
            lj = li - j
            f = hh[i] / (t[li] - t[lj])
            h = h.at[i].add(f * (t[li] - x))
            h = h.at[i + 1].set(f * (x - t[lj]))
    return h

def precompute_domain_idx_and_basis_values(tx, nx, kx, x):
    """Find knot-interval index and compute B-spline basis values.

    Returns (domain_idx, basis_values) where domain_idx is the left
    coefficient index and basis_values has kx+1 nonzero basis values.

    `nx`, `kx` must be static Python ints. `tx`, `x` may be traced/arrays.
    """
    tx = jnp.asarray(tx)
    kx1 = kx + 1
    nkx1 = nx - kx1
    tb = tx[kx1 - 1]
    te = tx[nkx1]
    arg = jnp.clip(x, tb, te)

    # Equivalent to the FITPACK "while arg >= tx[l] and l != nkx1: l += 1"
    # search, but fully vectorized/traceable: find smallest l with
    # arg < tx[l], clipped into [kx1, nkx1].
    l = jnp.searchsorted(tx, arg, side="right")
    l = jnp.clip(l, kx1, nkx1)

    h0 = jnp.zeros(kx + 2)
    h = fpbspl(tx, kx, arg, l, h0)
    domain_idx = l - kx1
    return domain_idx, h[: kx + 1]

# @partial(jax.jit, static_argnums=(3, 4))
# def eval2D_bivariate_fp_bspline(c, tx1, tx2, kx1, kx2, x1, x2):
#     """Evaluate a single 2D FITPACK B-spline at arrays of points.

#     Parameters
#     ----------
#     c : ndarray, shape (nc1, nc2)
#         Coefficient array in natural 2D shape.
#         nc_d = n_d - k_d - 1 per FITPACK convention.
#     tx1, tx2 : ndarray
#         Knot vectors for each dimension.
#     kx1, kx2 : int (static)
#         B-spline degrees in each dimension.
#     x1, x2 : array-like, same length m
#         Evaluation coordinates.

#     Returns
#     -------
#     z : ndarray, shape (m,)
#         Spline values at each (x1[i], x2[i]).
#     """
#     c = jnp.asarray(c)
#     tx1 = jnp.asarray(tx1)
#     tx2 = jnp.asarray(tx2)
#     x1 = jnp.asarray(x1)
#     x2 = jnp.asarray(x2)

#     nx1 = tx1.shape[0]
#     nx2 = tx2.shape[0]

#     def eval_single(x1i, x2i):
#         lx1, wx = precompute_domain_idx_and_basis_values(tx1, nx1, kx1, x1i)
#         lx2, wy = precompute_domain_idx_and_basis_values(tx2, nx2, kx2, x2i)
#         coeff_block = lax.dynamic_slice(c, (lx1, lx2), (kx1 + 1, kx2 + 1))
#         sp = jnp.sum(coeff_block * wx[:, None] * wy[None, :])
#         return sp

#     z = jax.vmap(eval_single)(x1, x2)
#     return z


@partial(jax.jit, static_argnums=(3, 4))
def eval2D_bivariate_fp_bspline(c, tx1, tx2, kx1, kx2, x1, x2):
    """Evaluate a single 2D FITPACK B-spline at arrays of points.

    Parameters
    ----------
    c : ndarray, shape (nc1, nc2)
        Coefficient array in natural 2D shape.
        nc_d = n_d - k_d - 1 per FITPACK convention.
    tx1, tx2 : ndarray
        Knot vectors for each dimension.
    kx1, kx2 : int (static)
        B-spline degrees in each dimension.
    x1, x2 : array-like, same length m
        Evaluation coordinates.

    Returns
    -------
    z : ndarray, shape (m,)
        Spline values at each (x1[i], x2[i]).
    """
    nx1 = tx1.shape[0]
    nx2 = tx2.shape[0]

    lx1, wx = precompute_domain_idx_and_basis_values(tx1, nx1, kx1, x1)
    lx2, wy = precompute_domain_idx_and_basis_values(tx2, nx2, kx2, x2)
    coeff_block = lax.dynamic_slice(c, (lx1, lx2), (kx1 + 1, kx2 + 1))
    return (coeff_block * wx[:, None] * wy[None, :]).sum()


def get_amplitude_interpolant(filename, interp_type='mag-phase', n_fourier_modes=64):
    with h5py.File(filename, 'r') as f:
        psi_r_loaded = f["psi_r"][:]
        e_amp_loaded     = f["e"][:]
        dp_amp_loaded    = f["dp"][:]
        mag_loaded   = f["magnitude"][:]   # (n_dp, n_e, N_psi)
        phase_loaded = np.unwrap(f["phase"][:], axis=-1)

    dp_interp = np.log10(dp_amp_loaded)

    if interp_type == 'mag-phase':
        target1 = mag_loaded
        target2 = phase_loaded
    elif interp_type == 'real-imag':
        target1 = mag_loaded * np.cos(phase_loaded)
        target2 = - mag_loaded * np.sin(phase_loaded)
    elif interp_type == 'fourier':
        coeffs = np.fft.fftshift(np.fft.fft(mag_loaded * np.exp(1j * phase_loaded), axis=-1, norm="forward"), axes=-1)
        num_modes_data = coeffs.shape[-1]
        coeffs = coeffs[:,:,num_modes_data//2 - n_fourier_modes//2 : num_modes_data//2 + n_fourier_modes//2 + 1]

        target1 = np.real(coeffs)
        target2 = np.imag(coeffs)

    
    target1_spl = make_interp_spline((dp_interp, e_amp_loaded), target1)
    target1_coeffs = jnp.asarray(target1_spl.coefficients)
    target2_spl = make_interp_spline((dp_interp, e_amp_loaded), target2)
    target2_coeffs = jnp.asarray(target2_spl.coefficients)

    dp_knots = jnp.asarray(target1_spl.knots[0])
    e_knots = jnp.asarray(target1_spl.knots[1])
    
    delta_psi_r = float(psi_r_loaded[1] - psi_r_loaded[0])

    # test call
    eval2D_bivariate_fp_bspline(
        target1_coeffs[:,:,0], 
        dp_knots, 
        e_knots, 
        3, 
        3, 
        0.2, 
        0.1
    )

    bicubic_vec_fn = jax.jit(
        jax.vmap(
            eval2D_bivariate_fp_bspline,
            in_axes=(2, None, None, None, None, None, None),
        ),
        static_argnums=(3, 4)
    )


    def interpolant(p, e):
        dp = jnp.log10(p - (6 + 2 * e))
        
        y1 = bicubic_vec_fn(
            target1_coeffs, 
            dp_knots, 
            e_knots, 
            3, 
            3, 
            dp, 
            e
            )

        y2 = bicubic_vec_fn(
            target2_coeffs, 
            dp_knots, 
            e_knots, 
            3, 
            3, 
            dp, 
            e
            )
        
        return y1, y2
    
    return interpolant, jnp.asarray(psi_r_loaded), delta_psi_r
