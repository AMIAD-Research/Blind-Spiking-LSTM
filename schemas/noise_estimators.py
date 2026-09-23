import jax.numpy as jnp
from schemas.polynomial_jax import centered_mod
from schemas.format import Ciphertext, Plaintext



def get_noise_lwe(ciphertext:Ciphertext,sk:Plaintext,dict_params:dict,message:Plaintext):
    a,b = ciphertext[0],ciphertext[1]
    product = jnp.dot(sk,a)
    c = centered_mod(b-product-message,dict_params["q"])
    return c
