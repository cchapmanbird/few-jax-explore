from jax import Array, jit
import jax.numpy as jnp

def get_interpolant_for_coefficients(
    dx: float, dy: float, dz: float, coefficients: Array
):
    @jit
    def interpolant(x, y, z):
        tx = x / dx
        ix = jnp.floor(tx).astype(int)
        ty = y / dy
        iy = jnp.floor(ty).astype(int)
        tz = z / dz
        iz = jnp.floor(tz).astype(int)

        x_diff = tx - ix
        y_diff = ty - iy
        z_diff = tz - iz

        c = coefficients[ix, iy, iz]  # shape (4, 4, 4)

        # Horner over z
        cz = c[:, :, 3]
        for k in range(2, -1, -1):
            cz = c[:, :, k] + z_diff * cz  # shape (4, 4)

        # Horner over y
        cy = cz[:, 3]
        for j in range(2, -1, -1):
            cy = cz[:, j] + y_diff * cy  # shape (4,)

        # Horner over x
        result = cy[3]
        for i in range(2, -1, -1):
            result = cy[i] + x_diff * result  # scalar

        return result    
    return interpolant
