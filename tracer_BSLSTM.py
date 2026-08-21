"""
Benchmark d'un pas de temps du CipherSpikeLSTM, avec dimension de batch.

Principe : on reconstruit a la main le corps de `scan_step`, on capture tous les
intermediaires reels (bonnes shapes, bon niveau de bruit), puis on chronometre
chaque sous-operation jittee separement.

Toutes les sous-ops sont vmappees sur un axe de batch de tete (les LUT et les
poids restent partages, in_axes=None).

On compare ensuite :
    somme(sous-ops)  vs  1 pas complet  vs  scan complet / max_len
et, avec --batch-sweep, la scalabilite du temps par echantillon.

Usage:
    python bench_timestep.py --batch 4
    python bench_timestep.py --batch-sweep 1,2,4,8 --skip-scan
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "

import argparse
import json
import time
from functools import partial

import numpy as np
import torch
import jax
import jax.numpy as jnp
from jax import vmap
from flax import nnx

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

from schemas.RLWE_jax import encrypt, sample_extract
from schemas.text_jax import (
    rotate_ciphertext,
    sum_ciphertext_ciphertext,
    centered_mod_ciphertext,
)
from schemas.polynomial_jax import centered_mod
from nn_jax.HF_ds import get_embedding_model, get_datasets
from nn_tfhe.Spike_LSTM import CipherSpikeLSTM, CipherMLP
from nn_tfhe.utils_spike import *
from setup_crypto import *


# --------------------------------------------------------------------------- #
#  Chronometrage
# --------------------------------------------------------------------------- #
def bench(fn, *args, reps=20, warmup=3):
    
    jax.block_until_ready(fn(*args))
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))

    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return float(np.median(ts))



# --------------------------------------------------------------------------- #
#  Setup
# --------------------------------------------------------------------------- #
def build(conf_path, seed=0):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    with open(conf_path) as f:
        config = json.load(f)
    config["max_length"] = 128

    embeddings_model = get_embedding_model(config)
    hidden_dim = config["hidden_dim"]
    task = config["task"]
    input_dim = config["input_dim"]
    n_lut = config["n_lut"]

    model = SpikeLSTMModel(input_dim, hidden_dim, task, 0.3, n_lut, nnx.Rngs(1))
    datasets = get_datasets(config)
    valset = datasets["validation"].with_format("jax")

    sk, ksk_packing, galois_key, bsk, key_switching_key_bs = sample_crypto_keys(seed)
    plain_model = model
    _, _, sf, si, so, s_head1, s_out = get_boundaries(
        plain_model, valset.iter(batch_size=1024), task, embeddings_model
    )
    W_f, W_i, W_o, W_head1, W_out = get_linears(plain_model, sf, si, so, s_head1, s_out)
    lut_f, lut_i, lut_o = get_lut(plain_model, sf, si, so)
    heavyBS, reluBS = get_tfhe_activation(
        bsk, key_switching_key_bs, ksk_packing, galois_key, s_head1, n_lut
    )

    cipher_lstm = CipherSpikeLSTM(
        dict_params=dict_params,
        dict_params_lut=dict_params_lut,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        max_len=config["max_length"],
        linear_f=W_f,
        linear_i=W_i,
        linear_o=W_o,
        heavyBS=heavyBS,
        n_lut=n_lut,
        lut_f=lut_f,
        lut_i=lut_i,
        lut_o=lut_o,
        beta_x=beta_x,
    )
    cipher_head = CipherMLP(
        dict_params=dict_params,
        dict_params_lut=dict_params_lut,
        hidden_dim=hidden_dim,
        linear1=W_head1,
        linear2=W_out,
        reluBS=reluBS,
    )
    return config, sk, cipher_lstm, cipher_head



def make_input(config, sk, seed=0):
    """Chiffre `batch` sequences aleatoires -> X_cipher de shape (B, L, ...)."""
    L, input_dim = config["max_length"], config["input_dim"]
    n =  L
    key = jax.random.PRNGKey(seed)
    x = jnp.array(np.random.uniform(-1, 1, (n, input_dim)))
    x_poly = jnp.concat([x, jnp.zeros((n, degree - input_dim))], axis=-1)
    keys = jax.random.split(key, n)
    flat = vmap(encrypt, (0, None, None, 0))(jnp.round(beta_x * x_poly),
                                             sk, dict_params, keys)
    return flat



def warm_carry(lstm, X_cipher, config, step):
    """Etat (H, C) batche apres `step` pas reels : bruit representatif."""
    seq_len = np.ones(config["max_length"]).astype(int)
    run = jax.jit(lambda X: lstm(X, seq_len))
    (_, (H_hist, C_hist)) = run(X_cipher)
    jax.block_until_ready((H_hist, C_hist))
    return ((H_hist[0][step], H_hist[1][ step]),
            (C_hist[0][step], C_hist[1][ step]))



@partial(jax.jit,static_argnames=["model"])
def LUT(model,C_lwe_reshape):
    return vmap(vmap(model.heavyBS.prepare_LUT, (0, 0)), (0, 0))(C_lwe_reshape, model.lut_f)

@partial(jax.jit,static_argnames=["model"])
def PBS(model,input_boot, lut_boot):
    return model.heavyBS.cuboot(input_boot, lut_boot)

@partial(jax.jit,static_argnames=["model"])
def packing(model,input_packing):
    return model.heavyBS.packing(input_packing)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", default="conf_sst2.json")

    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--warm-step", type=int, default=0,
                    help="indice du pas dont on reprend l'etat (bruit realiste)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config, sk, model, head = build(args.conf, args.seed)


    print("== setup BSLSTM ==")
    print(f"  hidden_dim={model.hidden_dim}  n_lut={model.n_lut}  ")




    
    
    X_cipher = make_input(config, sk,  args.seed)
    x_t = jax.tree.map(lambda a: a[0], X_cipher)  
    H, C = warm_carry(model, X_cipher, config, args.warm_step)
    dim = jnp.arange(model.hidden_dim)
    ### Micro benchmark
    #
    # breakpoint()
    C_lwe = vmap(sample_extract,(None,0))(C, dim)
    
    C_lwe_reshape = (
        C_lwe[0].reshape(model.n_lut, model.hidden_dim // model.n_lut, model.degree_lut), 
        C_lwe[1].reshape(model.n_lut, model.hidden_dim // model.n_lut, 1)
    )

    ### first forward
    #Linear
    HX = sum_ciphertext_ciphertext(x_t, H, dict_params["q"])
    F_t = model.W_f(HX)
    lut_C_t_1 = vmap(vmap(model.heavyBS.prepare_LUT, (0, 0)), (0, 0))(C_lwe_reshape, model.lut_f)
    
    lut_C_t_1 = centered_mod_ciphertext(
        (jnp.sum(lut_C_t_1[0], axis=0), jnp.sum(lut_C_t_1[1], axis=0)), dict_params["q"]
    )


    #Forget
    LUT_F_time = bench(LUT,model,C_lwe_reshape,reps=args.reps,warmup=args.warmup)
    LUT1_obj = LUT(model,C_lwe_reshape)
    LUT1_obj = centered_mod_ciphertext(
                (jnp.sum(LUT1_obj[0], axis=0), jnp.sum(LUT1_obj[1], axis=0)), model.dict_params["q"]
            )

    PBS_f_time = bench(PBS,model,F_t,LUT1_obj,reps=args.reps,warmup=args.warmup)
    F = PBS(model,F_t,LUT1_obj)
    F = (
            F[0].reshape(model.hidden_dim, model.degree_lut), 
            F[1].reshape(model.hidden_dim ,1)
        )

    #Input
    PBS_i_time = bench(PBS,model,F_t,model.lut_i,reps=args.reps,warmup=args.warmup)
    I = PBS(model,F_t,LUT1_obj)
    C_out_lwe = (
                    centered_mod(I[0].reshape(model.hidden_dim, model.degree_lut)+F[0], dict_params["q"]), 
                    centered_mod(I[1].reshape(model.hidden_dim, 1)+F[1], dict_params["q"])
                )
    
    
    C_out_lwe_reshape = (
            C_out_lwe[0].reshape(model.n_lut, model.hidden_dim // model.n_lut, model.degree_lut), 
            C_out_lwe[1].reshape(model.n_lut, model.hidden_dim // model.n_lut, 1)
        )
    I = (
            I[0].reshape(model.hidden_dim, model.degree_lut), 
            I[1].reshape(model.hidden_dim ,1)
        )

    #Output
    LUT_O_time = bench(LUT,model,C_out_lwe_reshape,reps=args.reps,warmup=args.warmup)
    

    PBS_o_time = bench(PBS,model,F_t,LUT1_obj,reps=args.reps,warmup=args.warmup)
    
    #packing output
    packing_time = 0
    packing_time += bench(packing,model,F,reps=args.reps,warmup=args.warmup)
    packing_time += bench(packing,model,I,reps=args.reps,warmup=args.warmup)

    PBS_tot_time = PBS_f_time + PBS_i_time + PBS_o_time
    LUT_tot_time = LUT_F_time + LUT_O_time
    total_time_per_timestep =  PBS_tot_time + LUT_tot_time + packing_time
    print(f"Bootstrapping pt : {jnp.round(100*PBS_tot_time/total_time_per_timestep)}%")
    print(f"Setuping LUT pt : {jnp.round(100*LUT_tot_time/total_time_per_timestep)}%")
    print(f"Output packing pt : {jnp.round(100*packing_time/total_time_per_timestep)}%")
    print(f"Total runtime for single timestep : {jnp.round(1000*total_time_per_timestep)}ms")
    print(f"Cout d'un PBS : {1000*PBS_f_time*model.n_lut/model.hidden_dim}ms")







    
    

if __name__ == "__main__":
    main()