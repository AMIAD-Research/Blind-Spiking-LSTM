import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from schemas.RLWE_jax import sample_sk, encrypt, get_delta, decrypt, key_switch, bootstrapping, cuboot,cuboot_merge, decrypt_quantization, BR
from schemas.LWE_jax import decrypt_LWE,get_keyswitch_sk_LWE,key_switch_LWE, decrypt_LWE_quantization
from nn_tfhe.activation import prepare_bootstrapping
import jax.numpy as jnp
from schemas.polynomial_jax import jax_fourier, coef_rotation, jax_inversefourier, decomposition, centered_mod, fft_polynomial_multiply, GLEV_polynomial
from schemas.text_jax import sample_extract, all_binary_vectors, packing,rotate_ciphertext,multiply_ciphertext_RGSW, gadget_product, multiply_plaintext_ciphertext, sum_ciphertext_ciphertext, lwe_to_RLWE, multiply_ciphertext_ciphertext
from schemas.noise_estimators import get_noise_lwe
from jax import vmap
import numpy as np
from typing import Tuple
import jax
import time
jax.config.update('jax_enable_x64', True)
from colorama import init, Fore, Back, Style
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    #'--xla_gpu_enable_latency_hiding_scheduler=true '
)
os.environ["JAX_ENABLE_PGLE"] = 'True'
init(autoreset=True)

t = 2**9
degree=2**11
B = 1000
L = 1
n_keyswitch_bootstrapping = 801
beta = 2
l=64
q = float(beta**l)
beta_ks = 2**2
l_ks= 10
l_bs = 2
beta_bs = 2**13
sigma = 2**21
sigma_lut = 2**21
sigma_lwe = 2**48.8
seed = 1
collapse = 3
key = jax.random.PRNGKey(seed)
key_LUT = jax.random.PRNGKey(seed+1)
dict_params = {
        "t" : t,
        "degree" : degree,
        "q" : q,
        "beta_bs" : beta_bs,
        "l_bs" : l_bs,
        "beta_ks" : beta_ks,
        "l_ks" : l_ks,
        "sigma" : sigma,
}


dict_params_ks_LWE = {
        "t" : t,
        "degree" : n_keyswitch_bootstrapping,
        "q" : q,
        "beta_bs" : beta_bs,
        "l_bs" : l_bs,
        "beta_ks" : beta_ks,
        "l_ks" : l_ks,
        "sigma" : sigma_lwe,
}
degree_lut = 2**11
dict_params_packing = {
        "t" : t,
        "degree" : degree_lut,
        "q" : q,
        "beta_bs" : beta_bs,
        "l_bs" : l_bs,
        "beta_ks" : beta_ks,
        "l_ks" : l_ks,
        "sigma" : sigma_lut,
}


dict_params_lut = {
        "t" : t,
        "degree" : degree_lut,
        "q" : q,
        "beta_bs" : beta_bs,
        "l_bs" : l_bs,
        "beta_ks" : beta_ks,
        "l_ks" : l_ks,
        "key" : key,
        "sigma" : sigma_lut,
}
###Secret key
np.random.seed(0)
sk1 = sample_sk(dict_params)
sk_bsk = np.random.choice([0.,1.],size=(n_keyswitch_bootstrapping))


delta = get_delta(dict_params)
key= jax.random.split(key)[0]
print("Encryption/Decryption")
m1 = jax.random.randint(key,(B,degree),-t//4+1,t//4)


key= jax.random.split(key,(B))
start = time.time()
c1 = vmap(encrypt,(0,None,None,0))(m1*delta,sk1,dict_params,key)
key= jax.random.split(key[0])[0]





###Cofficients Extraction
print("Extracting all coefficients")
##Getting all LWEs
start = time.time()
c_lwe = vmap(sample_extract,(0,None))(c1,0)
c_lwe[0].block_until_ready()
m_1 = vmap(decrypt_LWE,(0,None,None))(c_lwe,sk1,dict_params)
if (m_1.flatten()==m1[:,0].flatten()).all():
    print(Fore.GREEN+"Coefficient Extraction all OK")
else:
    print(Fore.RED+"Coefficient Extraction all Not OK")
extract_time = time.time() - start
print(Fore.YELLOW +f"Coefficients extraction all time : {extract_time}")
print()

print("Preparing bootstrapping with collapse")
sk_lut = np.random.choice([0.,1.],(dict_params_lut["degree"],))
beta_x = 2**60
#beta_x = delta
def identity(x,degree):
        return jnp.ones(degree)

_, _, bsk_ordered, key_switching_key_bs, encrypted_LUT = prepare_bootstrapping(key, sk1, sk_lut, sk_bsk,
                                                                                                              dict_params, dict_params_lut, dict_params_ks_LWE,
                                                                                                              identity, beta_x, collapse)

#lut_fn = lambda x:np.where(x>0,x,0)
lut_fn = lambda x:x


encrypted_LUT = [
    jnp.zeros((B,degree_lut)),
    jnp.concat([beta_x*jnp.ones((B,degree_lut//2)), jnp.zeros((B,degree_lut//2))],axis=-1)
]


all_rot_possible_index = jnp.arange(-degree_lut, degree_lut)
all_rot_possible = np.zeros((2*degree_lut, degree_lut))
all_rot_possible[:,0] = 1
all_rot_possible = vmap(coef_rotation,(0,0))(all_rot_possible,all_rot_possible_index)
all_rot_possible = jnp.roll(all_rot_possible,-degree_lut,axis=0)
all_rot_possible_fourier = vmap(jax_fourier,0)(all_rot_possible)


### Boostrapping all_coef
print("Boostrapping all with collapse")
##Getting all LWE coef


c_lwe_ks = vmap(key_switch_LWE,(None,0,None))(key_switching_key_bs,c_lwe,dict_params_ks_LWE)
# #Bootstrapping

print("JAX")
#with jax.profiler.trace("jax-trace/boot",create_perfetto_link=True):
#jax.profiler.start_trace("jax-trace/boot")
start = time.time()
boot = vmap(bootstrapping,in_axes=(0,0,None,None,None,None,None,None,None))(c_lwe_ks, encrypted_LUT , bsk_ordered, dict_params_lut["q"],dict_params_lut["beta_bs"],dict_params_lut["l_bs"]
                                                                                                                    ,dict_params_lut["degree"],collapse, all_rot_possible_fourier)
boot[1].block_until_ready()
boot[0].block_until_ready()
boostrapping_time = time.time() - start
# breakpoint()


f_m1 = (vmap(decrypt_LWE_quantization,(0,None,None,None))(boot,sk_lut,dict_params_lut, beta_x)).flatten()

#breakpoint()


#a = lut_fn(m_1[:,0].flatten())
a = jnp.where(m1[:,0]>=0 , 1, 0)
b = f_m1.flatten()
if jnp.abs(a - b.flatten()).mean() <=0.5:
    print(Fore.GREEN + "Boostratpping with collapse OK")
    print(jnp.abs(a - b.flatten()).mean())
else:
    print(Fore.RED + "Boostratpping with collapse Not OK")
    print(Fore.RED + f"Error {jnp.abs(a - b.flatten()).mean()}")

print(Fore.YELLOW + f"Bootstrapping with collapse compute time : {boostrapping_time}")
print(Fore.YELLOW + f"Bootstrapping with collapse compute time amortized: {boostrapping_time/B}")
print()










print("Cuda")
boot = cuboot_merge(c_lwe_ks, encrypted_LUT , bsk_ordered, dict_params_lut["q"],dict_params_lut["beta_bs"],dict_params_lut["l_bs"]
            ,dict_params_lut["degree"],collapse, all_rot_possible_fourier,1)
boot[1].block_until_ready()
boot[0].block_until_ready()
boostrapping_time = 0

for _ in range(100):
    start = time.time()
    boot = cuboot_merge(c_lwe_ks, encrypted_LUT , bsk_ordered, dict_params_lut["q"],dict_params_lut["beta_bs"],dict_params_lut["l_bs"]
            ,dict_params_lut["degree"],collapse, all_rot_possible_fourier,1)
    boot[1].block_until_ready()
    boot[0].block_until_ready()

    boostrapping_time += time.time() - start

f_m1 = (vmap(decrypt_LWE_quantization,(0,None,None,None))(boot,sk_lut,dict_params_lut, beta_x)).flatten()




a = jnp.where(m1[:,0]>=0 , 1, 0)
b = jnp.round(f_m1.flatten())
if jnp.abs(a - b.flatten()).mean() <=0.5:
    print(Fore.GREEN + "Boostratpping with collapse OK")
    print(jnp.abs(a - b.flatten()).mean())
else:
    print(Fore.RED + "Boostratpping with collapse Not OK")
    print(Fore.RED + f"Error {jnp.abs(a - b.flatten()).mean()}")

print(Fore.YELLOW + f"Bootstrapping with collapse compute time : {boostrapping_time/100}")
print(Fore.YELLOW + f"Bootstrapping with collapse compute time amortized: {boostrapping_time/(B*100)}")
print()




