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
from nn_jax.utils import  process_batch
from flax import nnx
import random
import jax
import torch
import importlib
import argparse
import optax
import json
import pandas as pd

from nn_jax.Spike_LSTM_NML import SpikeLSTMModel_NML
from nn_jax.Spike_LSTM import SpikeLSTMModel
from nn_jax.utils import  train_step, process_batch,  get_emb, save_model, load_model
from nn_jax.HF_ds import get_datasets, get_embedding_model














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
    mlp_factor = config["mlp_factor"]
    model = SpikeLSTMModel_NML(input_dim, hidden_dim,task,config["dropout"],config["n_lut"],mlp_factor, nnx.Rngs(config["seed"]))
    

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
            save_model(f"ckpts_lstm_spike_nml/checkpoints_{config["name"]}_{hidden_dim}_seed_{seed}",model)
            max_metric = metric_tot

    plain_model = load_model(f"ckpts_lstm_spike_nml/checkpoints_{config["name"]}_{hidden_dim}_seed_{seed}", model)
    plain_model.eval()
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
    df.to_csv(f"submission_NML/{config["submission_name"]}_seed_{config["seed"]}.tsv",sep='\t',index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plaintext training")
    parser.add_argument("--config",type=str,default="conf_sst2_nml.json")
    parser.add_argument("--seed",type=int,default=0)
    args = parser.parse_args()
    config_path = args.config
    seed = args.seed
    with open(config_path) as f:
        config = json.load(f)
    config["seed"] = seed
    main(config)