import numpy as np
from jax import vmap
import jax.numpy as jnp
from schemas.polynomial_jax import coef_rotation, jax_fourier

q = float(2**64)
Q = q
degree_lut = 2**11
t = degree_lut
degree = 2**11
n_keyswitch_bootstrapping = 840
sigma = 2**22.6
sigma_lut = 2**22.6
sigma_lwe = 2**48.6

###packing monomial
beta_ks_monomial = 2**2
l_ks_monomial = 20


###packing out cell
beta_ks_default = 2**8
l_ks_default = 4

###ks before BS
beta_ks_lwe = 2*1
l_ks_lwe = 20

##BS
beta_bs = 2**13
l_bs = 2
collapse = 3

dict_params = {
        "t" : t,
        "degree" : degree,
        "q" : q,
        "beta_bs" : beta_bs,
        "l_bs" : l_bs,
        "beta_ks" : beta_ks_default,
        "l_ks" : l_ks_default,
        "sigma" : sigma,
}



dict_params_ks_LWE = {
        "t" : t,
        "degree" : n_keyswitch_bootstrapping,
        "q" : q,
        "beta_bs" : beta_bs,
        "l_bs" : l_bs,
        "beta_ks" : beta_ks_lwe,
        "l_ks" : l_ks_lwe,
        "sigma" : sigma_lwe,
}


dict_params_packing = {
        "t" : t,
        "degree" : degree,
        "q" : q,
        "beta_bs" : beta_bs,
        "l_bs" : l_bs,
        "beta_ks" : beta_ks_monomial,
        "l_ks" : l_ks_monomial,
        "sigma" : sigma,
}


dict_params_lut = {
        "t" : degree_lut,
        "degree" : degree_lut,
        "q" : q,
        "beta_bs" : beta_bs,
        "l_bs" : l_bs,
        "beta_ks" : beta_ks_default,
        "l_ks" : l_ks_default,
        "collapse" : collapse,
        "sigma" : sigma_lut,
}

beta_x = float(2**56)
beta_w = q/(beta_x*4)

all_rot_possible_index = jnp.arange(-degree_lut, degree_lut)
all_rot_possible = np.zeros((2*degree_lut, degree_lut))
all_rot_possible[:,0] = 1
all_rot_possible = vmap(coef_rotation,(0,0))(all_rot_possible,all_rot_possible_index)
all_rot_possible = jnp.roll(all_rot_possible,-degree_lut,axis=0)
all_rot_possible_fourier = vmap(jax_fourier,0)(all_rot_possible)