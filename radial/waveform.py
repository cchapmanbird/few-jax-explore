import jax
jax.config.update("jax_enable_x64", True)
from jax import lax
import jax.numpy as jnp
from fewjaxexplore.constants import MTSUN_SI, Gpc, MRSUN_SI
from fewjaxexplore.summation.spline import get_spline_coefficients_variable
    
def y22(theta, phi): 
    return (jnp.exp(2j*phi)*jnp.sqrt(5./jnp.pi)*pow(jnp.cos(theta/2.),4))/2. 

def y2neg2(theta, phi): 
    return (jnp.sqrt(5./jnp.pi)*pow(jnp.sin(theta/2.),4))/(2.*jnp.exp(2j*phi)) 

def to_psi_r(Phi_r, e):
    Phi_r = Phi_r % (2 * jnp.pi)
    psi_r = jnp.arccos(
        (
            -1 + 1 / (
                jnp.sin(Phi_r / 2) * (1 / (1 - e) - 1 / (1 + e)) + 1 / (1 + e)
            )
        ) / e
    )

    psi_r = jnp.where(Phi_r > jnp.pi, 2 * jnp.pi - psi_r, psi_r)
    return psi_r

def get_psi_r_indices(psi_r, delta_psi_r, num_psi_r):
    psi_r_grid_bin = jnp.floor( psi_r / (2 * jnp.pi) * num_psi_r).astype(int)
    psi_r_grid_l = psi_r_grid_bin * delta_psi_r
    psi_r_grid_m = psi_r_grid_l + delta_psi_r
    psi_r_grid_u = psi_r_grid_m + delta_psi_r

    l0 = (psi_r - psi_r_grid_m) * (psi_r - psi_r_grid_u) / (2 * delta_psi_r**2)
    l1 = (psi_r - psi_r_grid_l) * (psi_r - psi_r_grid_u) / (-delta_psi_r**2)
    l2 = (psi_r - psi_r_grid_l) * (psi_r - psi_r_grid_m) / (2 * delta_psi_r**2)

    psi_r_idx_low = psi_r_grid_bin
    psi_r_idx_mid = (psi_r_idx_low + 1) % num_psi_r
    psi_r_idx_up = (psi_r_idx_mid + 1) % num_psi_r

    return jnp.array([l0, l1, l2]), jnp.array([psi_r_idx_low, psi_r_idx_mid, psi_r_idx_up])

def get_re_im_mode_sum_at_t(
        t_dimensionless, 
        idx, 
        dt, 
        real_coeffs,
        imag_coeffs, 
        ode_solution, 
        delta_psi_r,
        num_psi_r,
        y22_,
        y2neg2_,
    ):
    _, e, Phi_phi, Phi_r = ode_solution.evaluate(t_dimensionless)
    psi_r_i = to_psi_r(Phi_r, e)

    # cubic + second-order Lagrange interpolation
    lvec, psi_r_indices = get_psi_r_indices(psi_r_i, delta_psi_r, num_psi_r)

    # index into the spline coefficients relevant for the current time step
    c_re = real_coeffs[psi_r_indices, idx].T
    real_stencil = c_re[0] + dt*(c_re[1] + dt*(c_re[2] + dt*c_re[3]))
    c_im = imag_coeffs[psi_r_indices, idx].T
    imag_stencil = c_im[0] + dt*(c_im[1] + dt*(c_im[2] + dt*c_im[3]))

    # combine the Lagrange interpolation with the cubic spline evaluation
    re = jnp.dot(lvec, real_stencil)
    im = jnp.dot(lvec, imag_stencil)

    azimuthal_phase = - 2 * Phi_phi
    co = jnp.cos(azimuthal_phase)
    si = jnp.sin(azimuthal_phase)
    amp = (re + 1j * im) * (co + 1j * si)
    return amp * y22_ + jnp.conj(amp) * y2neg2_

def get_mag_phase_mode_sum_at_t(
        t_dimensionless, 
        idx, 
        dt, 
        mag_coeffs,
        phase_coeffs, 
        ode_solution, 
        delta_psi_r,
        num_psi_r,
        y22_,
        y2neg2_,
    ):
    _, e, Phi_phi, Phi_r = ode_solution.evaluate(t_dimensionless)
    psi_r_i = to_psi_r(Phi_r, e)

    # cubic + second-order Lagrange interpolation
    lvec, psi_r_indices = get_psi_r_indices(psi_r_i, delta_psi_r, num_psi_r)

    # index into the spline coefficients relevant for the current time step
    c_mag = mag_coeffs[psi_r_indices, idx].T
    mag_stencil = c_mag[0] + dt*(c_mag[1] + dt*(c_mag[2] + dt*c_mag[3]))
    c_phase = phase_coeffs[psi_r_indices, idx].T
    phase_stencil = c_phase[0] + dt*(c_phase[1] + dt*(c_phase[2] + dt*c_phase[3]))

    # combine the Lagrange interpolation with the cubic spline evaluation
    mag = jnp.dot(lvec, mag_stencil)
    phase = jnp.dot(lvec, phase_stencil)

    total_phase = - 2 * Phi_phi - phase
    co = jnp.cos(total_phase)
    si = jnp.sin(total_phase)
    return mag * (
        y22_ * (co + 1j * si) + y2neg2_ * (co - 1j * si)
    )
    
def interp_radial_sum_chunked(
        t_eval, 
        t_knots, 
        a1_coeffs,
        a2_coeffs, 
        ode_solution, 
        delta_psi_r,
        num_psi_r,
        y22_,
        y2neg2_,
        Msec, 
        typesel,
        batch_size
    ):
    indices = jnp.searchsorted(t_knots, t_eval, side="right") - 1
    dt = t_eval - t_knots[indices]
    
    kernel = get_mag_phase_mode_sum_at_t if typesel == 0 else get_re_im_mode_sum_at_t
    out = lax.map(
        lambda t_idx_dt: kernel(
            t_idx_dt[0] / Msec, 
            t_idx_dt[1], 
            t_idx_dt[2], 
            a1_coeffs,
            a2_coeffs,
            ode_solution, 
            delta_psi_r,
            num_psi_r,
            y22_, 
            y2neg2_
            ),
        (t_eval, indices, dt),
        batch_size=batch_size
    )
    return out


def get_radial_waveform_generator(
        traj_gen, 
        amp_gen, 
        delta_psi_r,
        num_psi_r,
        amp_type='mag-phase',
        batch_size=10000
    ):
    
    def radial_waveform_generator(
            t_eval,
            m1, 
            m2,  
            p0, 
            e0, 
            Phi_phi0, 
            Phi_r0,
            theta,
            dist,
            T,
        ):
        M = m1 + m2
        mu = m1 * m2 / M**2
        Msec = M * MTSUN_SI

        sol = traj_gen(m1, m2, p0, e0, Phi_phi0, Phi_r0, T)
        
        amp1, amp2 = amp_gen(sol.ys[:,0], sol.ys[:,1])

        y22_ = y22(theta, - jnp.pi/2)
        y2neg2_ = y2neg2(theta, - jnp.pi/2)

        coeffs1 = jax.vmap(
            get_spline_coefficients_variable,
            in_axes=(None, 1, None, None)
        )(
            sol.ts * Msec, amp1, sol.stats["num_accepted_steps"], 257
        )

        coeffs2 = jax.vmap(
            get_spline_coefficients_variable,
            in_axes=(None, 1, None, None),  
        )(
            sol.ts * Msec, amp2, sol.stats["num_accepted_steps"], 257
        )

        typesel = 0 if amp_type == 'mag-phase' else 1

        waveform = interp_radial_sum_chunked(
            t_eval,
            sol.ts * Msec,
            coeffs1,
            coeffs2,
            sol,
            delta_psi_r,
            num_psi_r,
            y22_,
            y2neg2_,
            Msec,
            typesel,
            batch_size=batch_size
        )

        return waveform * (mu * MRSUN_SI  / dist / Gpc)

    return radial_waveform_generator


def get_radial_fourier_mode_sum_at_t(
        t_dimensionless, 
        idx, 
        dt, 
        mag_coeffs,
        phase_coeffs, 
        ode_solution, 
        mode_indices,
        y22_,
        y2neg2_,
    ):
    _, e, Phi_phi, Phi_r = ode_solution.evaluate(t_dimensionless)
    psi_r = to_psi_r(Phi_r, e)

    # cubic + second-order Lagrange interpolation

    # get all relevant Fourier modes for the current time step
    c_re = mag_coeffs[:, idx].T
    re_int = c_re[0] + dt*(c_re[1] + dt*(c_re[2] + dt*c_re[3]))
    c_imag = phase_coeffs[:, idx].T
    im_int = c_imag[0] + dt*(c_imag[1] + dt*(c_imag[2] + dt*c_imag[3]))

    total_phase = 2 * Phi_phi + mode_indices * psi_r
    co = jnp.cos(total_phase)
    si = jnp.sin(total_phase)

    partial = ((re_int + 1j * im_int) * (co + 1j * si)).sum()
    return jnp.conj(partial * y22_ + jnp.conj(partial) * y2neg2_)
    
def interp_radial_fourier_sum_chunked(
        t_eval, 
        t_knots, 
        a1_coeffs,
        a2_coeffs, 
        ode_solution, 
        mode_indices,
        y22_,
        y2neg2_,
        Msec, 
        batch_size
    ):
    indices = jnp.searchsorted(t_knots, t_eval, side="right") - 1
    dt = t_eval - t_knots[indices]
    
    kernel = get_radial_fourier_mode_sum_at_t
    out = lax.map(
        lambda t_idx_dt: kernel(
            t_idx_dt[0] / Msec, 
            t_idx_dt[1], 
            t_idx_dt[2], 
            a1_coeffs,
            a2_coeffs,
            ode_solution, 
            mode_indices,
            y22_, 
            y2neg2_
            ),
        (t_eval, indices, dt),
        batch_size=batch_size
    )
    return out


def get_radial_fourier_waveform_generator(
        traj_gen, 
        amp_gen, 
        num_modes,
        batch_size=10000
    ):
    
    mode_indices = jnp.arange(-num_modes//2, num_modes//2 + 1)

    def radial_waveform_generator(
            t_eval,
            m1, 
            m2,  
            p0, 
            e0, 
            Phi_phi0, 
            Phi_r0,
            theta,
            dist,
            T,
        ):
        M = m1 + m2
        mu = m1 * m2 / M**2
        Msec = M * MTSUN_SI

        sol = traj_gen(m1, m2, p0, e0, Phi_phi0, Phi_r0, T)
        
        amp1, amp2 = amp_gen(sol.ys[:,0], sol.ys[:,1])

        y22_ = y22(theta, - jnp.pi/2)
        y2neg2_ = y2neg2(theta, - jnp.pi/2)

        coeffs1 = jax.vmap(
            get_spline_coefficients_variable,
            in_axes=(None, 1, None, None)
        )(
            sol.ts * Msec, amp1, sol.stats["num_accepted_steps"], 257
        )

        coeffs2 = jax.vmap(
            get_spline_coefficients_variable,
            in_axes=(None, 1, None, None),  
        )(
            sol.ts * Msec, amp2, sol.stats["num_accepted_steps"], 257
        )

        waveform = interp_radial_fourier_sum_chunked(
            t_eval,
            sol.ts * Msec,
            coeffs1,
            coeffs2,
            sol,
            mode_indices,
            y22_,
            y2neg2_,
            Msec,
            batch_size=batch_size
        )

        return waveform * (mu * MRSUN_SI  / dist / Gpc)

    return radial_waveform_generator

