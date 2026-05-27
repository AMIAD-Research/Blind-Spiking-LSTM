import jax
import jax.numpy as jnp


def all_binary_vectors(m: int) -> jax.Array:
    values = jnp.array([0, 1], dtype=jnp.int32)
    grids = jnp.meshgrid(*[values] * m, indexing="ij")
    return jnp.stack(grids, axis=-1).reshape(-1, m)
