import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
import pandas as pd
import jax
import argparse
jax.config.update("jax_enable_x64", True)

jax.config.update("jax_optimization_level", "O3")
jax.config.update("jax_debug_nans", False)
jax.config.update("jax_debug_infs", False)

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_autotune_level=4'
)

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

import jax.numpy as jnp
import numpy as np
from nn_jax.utils import process_batch
from transformers import  AutoModel
from math import sqrt
from transformers import logging
logging.set_verbosity_error()
import torch
from tqdm import tqdm
from nn_tfhe.utils_spike import *
from nn_jax.utils import  process_batch, get_emb
from nn_jax.HF_ds import get_datasets, get_embedding_model

import json
from flax import nnx


seed = 0
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)




from parameters_crypto import *

def main(config):

    
    ###Plain model init
    embeddings_model = get_embedding_model(config)
    
    hidden_dim = config["hidden_dim"]
    task = config["task"]
    max_len = config["max_length"]
    n_lut = config["n_lut"]

    ##Datasets definition
    datasets = get_datasets(config)
    valset = datasets["validation"].with_format("jax")
    testset = datasets["test"].with_format("jax")
    test_size = testset.shape[0]
    n_test = 10
    batch_size = 1024

    li_logits_qplain = []
    li_logits_qplain_noise_input = []
    li_logits_qplain_all = []
    li_logits_plain = []
    print("Starting inference")
    config["seed"] = seed
    plain_model = get_spike_plain_model(config)
    y_plain = []
    ###plaintext inference
    for idx,batch in enumerate(testset.iter(batch_size=1024)):
        x, seq_len, _ = process_batch(batch, config["task"])
        x = get_emb(embeddings_model,x)
        _,_,h_all,_ = jax.vmap(plain_model.lstm,in_axes=0)(x)
        hf = h_all[jnp.arange(x.shape[0]),seq_len-1]
        if task == "Single Sentence":
            y_hat = plain_model.head(hf).flatten()
        else:
            size = x.shape[0]//2
            first_sentence = hf[:size]
            second_sentence = hf[size:]
            h = jnp.concat([first_sentence,second_sentence],axis=-1)
            y_hat = plain_model.head(h).flatten()
        y_plain.append(y_hat)
        plain_logits = jnp.where(y_hat>0,1,0)
        li_logits_plain.append(plain_logits)
    y_plain = jnp.concat(y_plain) 



    _, _, sf, si, so, s_head1, s_out = get_boundaries(plain_model,valset.iter(batch_size=1024),task,embeddings_model)
    
    plain_model.lstm.set_s(sf,si,so)
    plain_model.set_s(s_head1,s_out)
    plain_model.lstm.set_qgamma(sf,si,so, degree_lut//n_lut)
    std_ksk =  sqrt((degree_lut * l_ks_lwe * sigma_lwe**2 * beta_ks_lwe**2) /12 + n_keyswitch_bootstrapping * q**2 / (12*beta_ks_lwe**(2*l_ks_lwe)))/q
    noise_ms_lstm = sqrt(n_keyswitch_bootstrapping * 0.5/(12*collapse*4*(degree_lut//n_lut)**2))
    noise_ms_mlp = sqrt(n_keyswitch_bootstrapping * 0.5/(12*collapse*4*(degree_lut//n_lut)**2))

    std_ksk_f = sqrt(std_ksk**2+noise_ms_lstm**2)*sf*4
    std_ksk_i = sqrt(std_ksk**2+noise_ms_lstm**2)*si*4
    std_ksk_o = sqrt(std_ksk**2+noise_ms_lstm**2)*so*4
    std_ksk_all = jnp.stack([std_ksk_f ,std_ksk_i, std_ksk_o])
    std_ksk_mlp = sqrt(std_ksk**2+noise_ms_mlp**2)*s_head1*4
    
    packing_noise = sqrt(degree_lut * hidden_dim* l_ks_default * beta_ks_default ** 2 * sigma_lut ** 2 / 12 + 0.5 * degree_lut * q**2 / (12*beta_ks_default**(2*l_ks_default)))/beta_x
    
    std_bs = sqrt(2 * l_bs * degree_lut * n_keyswitch_bootstrapping * beta_bs**2 *sigma_lut**2 * 2**collapse /(collapse*12) + degree_lut * q**2 * n_keyswitch_bootstrapping/(collapse*beta_bs**(2*l_bs)*24)) / beta_x
    std_bs_lstm = jnp.stack([
        std_bs + packing_noise*jnp.sqrt(degree_lut), std_bs, std_bs +packing_noise*jnp.sqrt(degree_lut)
    ])
    
    std_bs_lstm = jnp.repeat(std_bs_lstm.reshape(-1,1),hidden_dim,-1)

    ###quantization only
    for _,batch in enumerate(tqdm(testset.iter(batch_size=batch_size))):
        x, seq_len, _ = process_batch(batch, task)
        x = get_emb(embeddings_model,x)

        ##Noise
        noise_lstm_input = np.random.normal(0,0,(x.shape[0],max_len,3,hidden_dim//n_lut))
        noise_lstm_output = np.random.normal(0,0,(x.shape[0],max_len,3,hidden_dim))
        _,_,qh_all,_  = jax.vmap(plain_model.lstm.qforward,in_axes=(0,None,None,0,0))(x, beta_x, q, noise_lstm_input, noise_lstm_output)
        qhf = qh_all[jnp.arange(x.shape[0]),seq_len-1]
        if task == "Single Sentence":
            noise_mlp_input = np.random.normal(0,0,(x.shape[0], hidden_dim))
            noise_mlp_output = np.random.normal(0,0,(x.shape[0],hidden_dim))
            qy_hat = plain_model.qhead(qhf, beta_x, q, noise_mlp_input, noise_mlp_output).flatten()
        else:
            current_size = x.shape[0]//2
            noise_mlp_input = np.random.normal(0,0,(x.shape[0]//2,hidden_dim))
            noise_mlp_output = np.random.normal(0,0,(x.shape[0]//2,hidden_dim))
            first_sentence = qhf[:current_size]
            second_sentence = qhf[current_size:]
            qh = jnp.concat([first_sentence,second_sentence],axis=-1)
            qy_hat = plain_model.qhead(qh, beta_x, q, noise_mlp_input, noise_mlp_output).flatten()
        
        qplain_logits = jnp.where(qy_hat>0,1,0)
        li_logits_qplain.append(qplain_logits)
    plain_logits = jnp.concat(li_logits_plain)
    qplain_logits = jnp.concat(li_logits_qplain)
    error_quantization_only = jnp.sum(plain_logits == qplain_logits)/plain_logits.shape[0]
    print(f"Fidelity prediction quantization only {config["name"]} : {error_quantization_only}")

    ###quantization + noise_input
    for _ in range(n_test):
        for _,batch in enumerate(tqdm(testset.iter(batch_size=batch_size))):
            x, seq_len, _ = process_batch(batch, task)
            x = get_emb(embeddings_model,x)

            ##Noise
            noise_lstm_input = np.random.normal(0,std_ksk_all,(x.shape[0],max_len,3,hidden_dim//n_lut))
            noise_lstm_output = np.random.normal(0,0,(x.shape[0],max_len,3,hidden_dim))
            _,_,qh_all,_  = jax.vmap(plain_model.lstm.qforward,in_axes=(0,None,None,0,0))(x, beta_x, q, noise_lstm_input, noise_lstm_output)
            qhf = qh_all[jnp.arange(x.shape[0]),seq_len-1]
            if task == "Single Sentence":
                noise_mlp_input = np.random.normal(0,std_ksk_mlp,(x.shape[0], hidden_dim))
                noise_mlp_output = np.random.normal(0,0,(x.shape[0],hidden_dim))
                qy_hat = plain_model.qhead(qhf, beta_x, q, noise_mlp_input, noise_mlp_output).flatten()
            else:
                current_size = x.shape[0]//2
                noise_mlp_input = np.random.normal(0,std_ksk_mlp,(x.shape[0]//2,hidden_dim))
                noise_mlp_output = np.random.normal(0,0,(x.shape[0]//2,hidden_dim))
                first_sentence = qhf[:current_size]
                second_sentence = qhf[current_size:]
                qh = jnp.concat([first_sentence,second_sentence],axis=-1)
                qy_hat = plain_model.qhead(qh, beta_x, q, noise_mlp_input, noise_mlp_output).flatten()
            
            qplain_logits = jnp.where(qy_hat>0,1,0)
            li_logits_qplain_noise_input.append(qplain_logits)
    

    qplain_noise_input_logits= jnp.concat(li_logits_qplain_noise_input).reshape(-1,test_size)
    error_quantization_noise_input = jnp.sum(plain_logits[None,:] == qplain_noise_input_logits)/(test_size * n_test)
    print(f"Fidelity prediction quantization + noise input {config["name"]} : {error_quantization_noise_input}")
     
    ###quantization + noise_input + noise_output
    for _ in range(n_test):
        for _,batch in enumerate(tqdm(testset.iter(batch_size=batch_size))):
            x, seq_len, _ = process_batch(batch, task)
            x = get_emb(embeddings_model,x)

            ##Noise
            noise_lstm_input = np.random.normal(0,std_ksk_all,(x.shape[0],max_len,3,hidden_dim//n_lut))
            noise_lstm_output = np.random.normal(0,std_bs_lstm,(x.shape[0],max_len,3,hidden_dim))
            _,_,qh_all,_  = jax.vmap(plain_model.lstm.qforward,in_axes=(0,None,None,0,0))(x, beta_x, q, noise_lstm_input, noise_lstm_output)
            qhf = qh_all[jnp.arange(x.shape[0]),seq_len-1]
            if task == "Single Sentence":
                noise_mlp_input = np.random.normal(0,std_ksk_mlp,(x.shape[0], hidden_dim))
                noise_mlp_output = np.random.normal(0,std_bs,(x.shape[0],hidden_dim))
                qy_hat = plain_model.qhead(qhf, beta_x, q, noise_mlp_input, noise_mlp_output).flatten()
            else:
                current_size = x.shape[0]//2
                noise_mlp_input = np.random.normal(0,std_ksk_mlp,(x.shape[0]//2,hidden_dim))
                noise_mlp_output = np.random.normal(0,std_bs,(x.shape[0]//2,hidden_dim))
                first_sentence = qhf[:current_size]
                second_sentence = qhf[current_size:]
                qh = jnp.concat([first_sentence,second_sentence],axis=-1)
                qy_hat = plain_model.qhead(qh, beta_x, q, noise_mlp_input, noise_mlp_output).flatten()
            
            qplain_logits = jnp.where(qy_hat>0,1,0)
            li_logits_qplain_all.append(qplain_logits)
    

    qplain_noise_all_logits = jnp.concat(li_logits_qplain_all).reshape(-1,test_size)
    error_quantization_noise_all = jnp.sum(plain_logits[None,:] == qplain_noise_all_logits)/(test_size * n_test)
    print(f"Fidelity prediction quantization + noise input + noise_output {config["name"]} : {error_quantization_noise_all}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plaintext training")
    parser.add_argument("--config",type=str,default="conf_sst2.json")
    args = parser.parse_args()
    config_path = args.config
    with open(config_path) as f:
        config = json.load(f)
    main(config)

