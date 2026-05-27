import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_deterministic_ops=true '
)
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from nn_jax.utils import save_model, process_batch
from flax import nnx
import jax
import torch
import importlib
import random
import argparse
import optax
import json
import pandas as pd
from setup_crypto import *

from nn_jax.Spike_LSTM import SpikeLSTMModel
from nn_jax.utils import  train_step, process_batch,  get_emb, save_model, load_model
from nn_jax.HF_ds import get_datasets, get_embedding_model

from nn_tfhe.utils_spike import get_boundaries, get_spike_plain_model
from nn_tfhe.Spike_LSTM import CipherMLP, CipherSpikeLSTM

from schemas.RLWE_jax import encrypt, decrypt_quantization
from schemas.LWE_jax import decrypt_LWE_quantization
from schemas.text_jax import rotate_ciphertext, sum_ciphertext_ciphertext









def main(config):
    ##Model definition
    embeddings_model = get_embedding_model(config)
    seed = config["seed"]
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    input_dim = config["input_dim"]
    hidden_dim = config["hidden_dim"]
    task = config["task"]
    n_lut = config["n_lut"]
    model = SpikeLSTMModel(input_dim, hidden_dim,task,config["dropout"],n_lut, nnx.Rngs(config["seed"]))
    

    max_gradient_norm = 1
    tx = optax.chain(
        optax.clip_by_global_norm(max_gradient_norm),
        optax.adamw(config["lr"])
    )
    optimizer = nnx.Optimizer(
    model, tx, wrt=nnx.Param
    )
    module = importlib.import_module("nn_jax.utils")
    metric_fn = getattr(module,config["metric"])
    ##Datasets definition
    datasets = get_datasets(config)
    trainset = datasets["train"].with_format("jax")
    valset = datasets["validation"].with_format("jax")
    testset = datasets["test"].with_format("jax")
    batch_size = config["batch_size"]

    ###Training
    loss_fn = getattr(module,config["loss"])
    train_metric = []
    train_loss = []
    val_metric = []
    val_loss = []
    max_metric = 0
    n_epoch = config["n_epoch"]
    for i in range(n_epoch):
        print(f"epoch : {i}")
        metric = 0
        loss_epoch = 0
        model.train()
        for idx, batch in enumerate(trainset.iter(batch_size=batch_size)):
            x, seq_len, y = process_batch(batch,task)
            x = get_emb(embeddings_model,x)
            loss, y_hat = train_step(model, optimizer, loss_fn, (x,seq_len, y))
            metric += metric_fn(y_hat, y)
            loss_epoch += loss
        train_metric.append(metric/(idx+1))
        train_loss.append(loss_epoch/(idx+1))
        loss_tot = 0
        metric_tot = 0
        model.eval()
        for idx,batch in enumerate(valset.iter(batch_size=512)):
            x, seq_len, y = process_batch(batch,task)
            x = get_emb(embeddings_model,x)
            loss, y_hat = loss_fn(model, (x, seq_len, y))
            loss_tot += loss
            metric_tot += metric_fn(y_hat,y)
        loss_tot = loss_tot/(idx+1)
        metric_tot = metric_tot/(idx+1)
        val_loss.append(loss_tot)
        val_metric.append(metric_tot)
        print(f"Validation metric : {metric_tot}")
        if metric_tot>max_metric:
            save_model(f"ckpts_lstm_spike/checkpoints_{config["name"]}_{hidden_dim}_seed_{seed}",model)
            max_metric = metric_tot
        

    ###crypto setup
    sk, ksk_packing, galois_key, bsk, key_switching_key_bs = sample_crypto_keys(seed)
    plain_model = get_spike_plain_model(config)
    ###plaintext inference
    li_logits_plain = []
    li_y_plain = []
    for idx,batch in enumerate(testset.iter(batch_size=1024)):
        x, seq_len, _ = process_batch(batch, task)
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
        plain_logits = jnp.where(y_hat>0,1,0)
        li_logits_plain.append(plain_logits)
        li_y_plain.append(y_hat)
        
    logits_p = jnp.concat(li_logits_plain).flatten()
    y_plain = jnp.concat(li_y_plain)
    index = np.arange(len(testset))
    df = pd.DataFrame(
    {"index":index,"prediction":logits_p}
    )
    df.to_csv(f"submission_plain_spike/{config["submission_name"]}_seed_{config["seed"]}.tsv",sep='\t',index=False)
    
    _, _, sf, si, so, s_head1, s_out = get_boundaries(plain_model,valset.iter(batch_size=1024),task,embeddings_model)
    
    W_f, W_i, W_o, W_head1, W_out = get_linears(plain_model,sf,si,so,s_head1,s_out)
    lut_f, lut_i, lut_o = get_lut(plain_model,sf,si,so)
    heavyBS, reluBS = get_tfhe_activation(bsk,key_switching_key_bs,ksk_packing,galois_key,s_head1,n_lut)

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
                                    input_dim = input_dim,
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

    ###cipher inference
    if task == "Single Sentence":
        batch_size = 64
    else:
        batch_size = 32
    max_len = config["max_length"]
    li_logits_c = []
    
    key = jax.random.PRNGKey(seed)
    key = jax.random.split(key)[0]
    #breakpoint()
    for idx,batch in enumerate(testset.iter(batch_size=batch_size)):
        x, seq_len, _ = process_batch(batch, task)
        x = get_emb(embeddings_model,x)        
        # _,_,h_all,_ = jax.vmap(plain_model.lstm,in_axes=0)(x)
        # hf = h_all[jnp.arange(x.shape[0]),seq_len-1]

        ###cipher
        x_emb_poly = jnp.concat([x,jnp.zeros((x.shape[0],max_len,degree-input_dim))],axis=-1)
        X_emb_poly = jnp.round(beta_x*x_emb_poly)
        key = jax.random.split(key,(x.shape[0],max_len))
        X_cipher = vmap(vmap(encrypt,(0,None,None,0)),(0,None,None,0))(X_emb_poly, sk, dict_params, key)
        key = jax.random.split(key[0,0])[0]
        ((_, _), (out_H_new, _)) = vmap(cipher_lstm,(0,0))(X_cipher, seq_len)
        if task == "Single Sentence":
            H_t = vmap(vmap(rotate_ciphertext,(0,None)),(0,None))(out_H_new,-input_dim)
            H_t = H_t[0][jnp.arange(x.shape[0]), seq_len-1], H_t[1][jnp.arange(x.shape[0]), seq_len-1]
            h_t = vmap(decrypt_quantization,(0,None,None,None))(H_t,sk, dict_params, beta_x)[:,:hidden_dim]
            out = vmap(cipher_head,0)(H_t)
        else:
            # size = x.shape[0]//2
            # first_sentence = hf[:size]
            # second_sentence = hf[size:]
            # h_final = jnp.concat([first_sentence,second_sentence],axis=-1)
            # y_hat = plain_model.head(h_final).flatten()



            n_sample = x.shape[0]//2
            first_sentence_cipher = [out_H_new[0][:n_sample], out_H_new[1][:n_sample]]
            second_sentence_cipher = [out_H_new[0][n_sample:], out_H_new[1][n_sample:]]
            first_sentence_cipher = [
                                    first_sentence_cipher[0][jnp.arange(n_sample),seq_len[:n_sample]-1], first_sentence_cipher[1][jnp.arange(n_sample),seq_len[:n_sample]-1]
            ]
            second_sentence_cipher = [
                                    second_sentence_cipher[0][jnp.arange(n_sample),seq_len[n_sample:]-1], second_sentence_cipher[1][jnp.arange(n_sample),seq_len[n_sample:]-1]
            ]
            first_sentence_cipher = vmap(rotate_ciphertext,(0,None))(first_sentence_cipher,-input_dim)
            second_sentence_cipher = vmap(rotate_ciphertext,(0,None))(second_sentence_cipher,-input_dim + hidden_dim)
            sentences = vmap(sum_ciphertext_ciphertext,(0,0,None))(first_sentence_cipher,second_sentence_cipher,q)
            sentence_dec = vmap(decrypt_quantization,(0,None,None,None))(sentences, sk, dict_params,beta_x)[:,:2*hidden_dim]
            out = vmap(cipher_head,0)(sentences)
        y_c = (vmap(vmap(decrypt_LWE_quantization,(0,None,None,None)),(0,None,None,None))(out,sk,dict_params,beta_x*beta_w)*s_out).flatten()
        #breakpoint()
        logits_c = jnp.where(y_c>0,1,0)
        li_logits_c.append(logits_c)

        
    

    logits_c = jnp.concat(li_logits_c).flatten()
    df = pd.DataFrame(
    {"index":index,"prediction":logits_c}
    )
    df.to_csv(f"submission_cipher_spike/{config["submission_name"]}_seed_{config["seed"]}.tsv",sep='\t',index=False)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plaintext training")
    parser.add_argument("--config",type=str,default="conf_sst2.json")
    parser.add_argument("--seed",type=int,default=0)
    args = parser.parse_args()
    config_path = args.config
    seed = args.seed
    with open(config_path) as f:
        config = json.load(f)
    config["seed"] = seed
    main(config)

