from jax import Array, vmap
import jax.numpy as jnp
from .mappings import uwz_of_ape_amp
from interpax import interp3d

L_MAX = 5
N_MAX = 55

index_table = jnp.full((L_MAX + 1, 2 * L_MAX + 1, 2 * N_MAX + 1), -1, dtype=int)

ALL_MODES = []

idx = 0
for l in range(2, L_MAX + 1):
    for m in range(-l, l + 1):
        if abs(m) < l - 2:
            idx += 1
            continue
        for n in range(-N_MAX, N_MAX + 1):
            ALL_MODES.append((l, m, n))
            index_table = index_table.at[l, m + L_MAX, n + N_MAX].set(idx)
            idx += 1

xv = jnp.linspace(0, 1, 33)
yv = jnp.linspace(0, 1, 33)
zv = jnp.linspace(0, 1, 33)

def get_amplitude_interpolant(filepath: str):
    """
    Load in data and return a function that takes in (a, p, e) and (l,m,n) arrays and returns the interpolated amplitudes.
    """
    data = jnp.load(filepath)
    def amplitude_interpolant(a: Array, p: Array, e: Array, l_modes: Array, m_modes: Array, n_modes: Array) -> Array:
        u, w, z = uwz_of_ape_amp(a, p, e)
        indices = jnp.atleast_1d(index_table[l_modes, m_modes + L_MAX, n_modes + N_MAX])
        targets = jnp.atleast_1d(data[indices]) 
        amps = vmap(lambda target: interp3d(u, w, z, xv, yv, zv, target, method="cubic2"))(targets)
        return amps
    return amplitude_interpolant