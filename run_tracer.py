import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    #'--xla_gpu_enable_latency_hiding_scheduler=true '
)


# jax.config.update("jax_optimization_level", "O3")
# jax.config.update("jax_debug_nans", False)
# jax.config.update("jax_debug_infs", False)
# jax.config.update("jax_enable_x64", True)

# jax.config.update("jax_compilation_cache_dir", "./jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update(
#     "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
# )

import jax.numpy as jnp

import numpy as np
from flax import nnx


from tqdm import tqdm
import torch
from schemas.RLWE_jax import encrypt
from schemas.LWE_jax import decrypt_LWE_quantization
from schemas.text_jax import rotate_ciphertext
import json
import time
from nn_jax.HF_ds import get_embedding_model, get_datasets
from nn_tfhe.activation import HeavysideBoostrapper, ReluBoostrapper
from nn_tfhe.Spike_LSTM import CipherSpikeLSTM, CipherMLP
from nn_tfhe.utils_spike import *
import jax
jax.config.update("jax_enable_x64", True)


jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)
seed = 0
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


from setup_crypto import *

with open("conf_sst2.json") as f:
        config = json.load(f)


embeddings_model = get_embedding_model(config)
hidden_dim = config["hidden_dim"]
config["max_length"] = 128
task = config["task"]
input_dim = config["input_dim"]
n_lut = config["n_lut"]
model = SpikeLSTMModel(input_dim, hidden_dim,task,0.3,n_lut,nnx.Rngs(1))
datasets = get_datasets(config)
valset = datasets["validation"].with_format("jax")


###crypto setup
sk, ksk_packing, galois_key, bsk, key_switching_key_bs = sample_crypto_keys(seed)
#plain_model = load_model(f"ckpts_lstm_spike/checkpoints_{config["name"]}_{hidden_dim}_seed_{seed}", model)
plain_model = model
_, _, sf, si, so, s_head1, s_out = get_boundaries(plain_model,valset.iter(batch_size=1024),task,embeddings_model)
W_f, W_i, W_o, W_head1, W_out = get_linears(plain_model,sf,si,so,s_head1,s_out)
#breakpoint()
lut_f, lut_i, lut_o = get_lut(plain_model,sf,si,so)
heavyBS, reluBS = get_tfhe_activation(bsk,key_switching_key_bs,ksk_packing,galois_key,s_head1,n_lut)
key = jax.random.PRNGKey(seed)
cipher_head = CipherMLP(
                                dict_params = dict_params,
                                dict_params_lut = dict_params_lut,
                                hidden_dim = hidden_dim,
                                linear1 = W_head1,
                                linear2 = W_out,
                                reluBS = reluBS
)

    
cipher_lstm = CipherSpikeLSTM(
                                dict_params = dict_params,
                                dict_params_lut = dict_params_lut,
                                input_dim=input_dim,
                                hidden_dim = hidden_dim,
                                max_len = config["max_length"],
                                linear_f = W_f,
                                linear_i = W_i,
                                linear_o = W_o,
                                heavyBS = heavyBS,
                                n_lut = n_lut,
                                lut_f = lut_f,
                                lut_i = lut_i,
                                lut_o = lut_o,
                                beta_x = beta_x
)
max_len = config["max_length"]
batch_size = 1
n_test = 5
runtime_tot = 0
from jax.profiler import ProfileOptions
opts = jax.profiler.ProfileOptions()

opts.host_tracer_level = 2
opts.device_tracer_level = 1
opts.python_tracer_level = 0


for i in tqdm(range(2)):
        
    seq_len = np.ones(batch_size).astype(int)*max_len
    x = jnp.array(np.random.uniform(-1,1,(batch_size,max_len,input_dim))*1.)



    ###Encryption
    x_emb_poly = jnp.concat([x,jnp.zeros((x.shape[0],max_len,degree-input_dim))],axis=-1)
    X_emb_poly = jnp.round(beta_x*x_emb_poly)
    key = jax.random.split(key,(x.shape[0],max_len))
    X_cipher = vmap(vmap(encrypt,(0,None,None,0)),(0,None,None,0))(X_emb_poly, sk, dict_params, key)
    key = jax.random.split(key[0,0])[0]

    start = time.time()
    ((H_t, C_t), (out_H_new, out_C_new)) = vmap(cipher_lstm,(0,0))(X_cipher, seq_len)
    H_t = vmap(vmap(rotate_ciphertext,(0,None)),(0,None))(out_H_new,-input_dim)
    H_t = H_t[0][jnp.arange(x.shape[0]), seq_len-1], H_t[1][jnp.arange(x.shape[0]), seq_len-1]

    out = vmap(cipher_head,0)(H_t) 
    y_c = (vmap(vmap(decrypt_LWE_quantization,(0,None,None,None)),(0,None,None,None))(out,sk,dict_params,beta_x*beta_w)*s_out).flatten()
    y_c.block_until_ready()



with jax.profiler.trace("jax_trace",profiler_options=opts):
    for i in tqdm(range(1)):
        ((H_t, C_t), (out_H_new, out_C_new)) = vmap(cipher_lstm,(0,0))(X_cipher, seq_len)
        H_t = vmap(vmap(rotate_ciphertext,(0,None)),(0,None))(out_H_new,-input_dim)
        H_t = H_t[0][jnp.arange(x.shape[0]), seq_len-1], H_t[1][jnp.arange(x.shape[0]), seq_len-1]

        out = vmap(cipher_head,0)(H_t) 
        y_c = (vmap(vmap(decrypt_LWE_quantization,(0,None,None,None)),(0,None,None,None))(out,sk,dict_params,beta_x*beta_w)*s_out).flatten()
        y_c.block_until_ready()



