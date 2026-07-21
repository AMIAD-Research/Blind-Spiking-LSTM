from functools import partial

import jax
import jax.numpy as jnp

from cuTFHE.utils import all_binary_vectors

from . import _core  # type: ignore[import-not-found]

jax.ffi.register_ffi_target(
    "mul_bootstrap",
    _core.mul_bootstrap(),
    platform="CUDA",
)


def _ffi_call(
    ct_init: jax.Array,
    bsk_reshaped: jax.Array,
    all_rot_fft: jax.Array,           # NEW: The whole table, without batch
    all_rotations_indices: jax.Array, # NEW: The batched indices
    log_beta: int,
    log_q: int,
    l: int,
) -> jax.Array:
    """Raw FFI call.
    ct_init: (2,N) int64,
    bsk: (n_iter,a,b,c,m,d),
    all_rot_fft: (2*degree,m,d),
    all_rotations_indices: (*batch,n_iter).
    """
    call = jax.ffi.ffi_call(
        "mul_bootstrap",
        jax.ShapeDtypeStruct(
            (bsk_reshaped.shape[1], ct_init.shape[-1]), # FIX: we remove shape[:-1]
            jnp.int64
        ),
        vmap_method="expand_dims",
    )
    result = call(
        ct_init,
        bsk_reshaped,
        all_rot_fft,             # Passing the global table
        all_rotations_indices,   # Passing the indices
        log2_beta=log_beta,
        log2_q=log_q,
        l=l,
        degree=ct_init.shape[-1],
    )
    #breakpoint()
    return result


@partial(jax.jit, static_argnames=["log_q", "log_beta", "l", "degree", "collapse","n_lut"])
def multiply_seq_monomial(
    ciphertext,
    public_key: jax.Array,
    BSK,
    log_q: int,
    log_beta: int,
    l: int,
    degree: int,
    collapse: int,
    all_rot_fft: jax.Array,
    n_lut:int
) -> jax.Array:
    n = public_key.shape[-1]
    n_iter = n // collapse
    ct_init = jnp.stack(ciphertext) if isinstance(ciphertext, tuple) else ciphertext
    kronecker_matrix = all_binary_vectors(collapse)
    pk_reshaped = public_key.reshape(*public_key.shape[:-1], n_iter, collapse)
    all_rotations_val = jnp.dot(pk_reshaped, kronecker_matrix.T)
    q = 2.0**log_q
    
    # The indices keep their batch dimension thanks to vmap
    #all_rotations_indices = (jnp.round(jnp.round(2 * degree * all_rotations_val / q)/n_lut)*n_lut).astype(
    all_rotations_indices = (jnp.round(2 * degree * all_rotations_val / (q*n_lut))*n_lut).astype(
        jnp.int32
    ) % (2 * degree)
    
    bsk_array = BSK
    shape = list(bsk_array.shape)
    shape[3] = n_iter
    shape.insert(4, 2**collapse)
    bsk_reshaped = bsk_array.reshape(shape)
    bsk_reshaped = jnp.permute_dims(bsk_reshaped, (3, 0, 2, 1, 4, 5))

    # REMOVED: rot_fft = all_rot_fft[all_rotations_indices] -> Goodbye OOM!
    
    ct_final = _ffi_call(ct_init.astype(jnp.int64), bsk_reshaped, all_rot_fft, all_rotations_indices, log_beta, log_q, l)
    #breakpoint()
    return ct_final.astype(jnp.float64)
