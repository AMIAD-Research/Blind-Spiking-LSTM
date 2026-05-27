from typing import Tuple,List
import jax.numpy as jnp
from jax import Array

Ciphertext = List[Array]
Plaintext = Array
RGSW = List[Ciphertext]