import jax.numpy as jnp
import jax
from nn_jax.Spike_LSTM import SpikeLSTMModel
from nn_jax.utils import load_model, process_batch, get_emb
from schemas.polynomial_jax import coef_rotation
from nn_tfhe.Spike_LSTM import Linear

def get_max_values(model,batch,task):
      x, seq_len, _ = batch

      lstm = model.lstm
      x_emb = x
      mask = jnp.ones_like(x_emb)
      #breakpoint()
      _,_,h_all,_ = jax.vmap(lstm,0)(x_emb)
      #breakpoint()
      x_emb = jnp.concat([x_emb[:,1:],x_emb[:,:1]],axis=1)
      
      all_possible_index = jnp.arange(h_all.shape[1])
      mask = all_possible_index < seq_len[:, jnp.newaxis]
      hx = jnp.concat([x_emb,h_all],axis=-1)[mask]
      
      hidden_dim = h_all.shape[-1]
      h = hx[...,-hidden_dim:]

      max_h = jnp.abs(hx).max()
      #breakpoint()
      f = lstm.linearf(hx)
      i = lstm.lineari(hx)
      o = lstm.linearo(hx)
      prop = 1.5
      frac = 99.5
      
      sf = jnp.percentile(jnp.abs(f),frac,axis=(0))*prop
      si = jnp.percentile(jnp.abs(i),frac,axis=(0))*prop
      so = jnp.percentile(jnp.abs(o),frac,axis=(0))*prop

      if task == "Single Sentence":
            #breakpoint()
            head1 = jnp.abs(model.head.layers[0](h))
            out = jnp.abs(model.head(h))
      else:
            n_sample = x_emb.shape[0]//2
            h_all = h_all[jnp.arange(x_emb.shape[0]),seq_len-1]
            head_in = jnp.concat([h_all[:n_sample],h_all[n_sample:]],axis=-1)
            head1 = model.head.layers[0](head_in)
            out = model.head(head_in)
      s_head1 = jnp.percentile(jnp.abs(head1), frac, axis=(0))*prop
      s_out = jnp.percentile(jnp.abs(out), frac, axis=(0))*prop
      return max_h, max_h, sf, si, so, s_head1, s_out


def get_spike_plain_model(config):
      hidden_dim = config["hidden_dim"]
      input_dim = config["input_dim"]
      task = config["task"]
      n_lut = config["n_lut"]
      seed = config["seed"]
      plain_model = SpikeLSTMModel(input_dim, hidden_dim, task, config["dropout"], n_lut)
      plain_model = load_model(f"ckpts_lstm_spike/checkpoints_{config["name"]}_{hidden_dim}_seed_{seed}", plain_model)
      plain_model.eval()
      return plain_model

def get_boundaries(plain_model, dataset, task, embeddings_model):
      #breakpoint()
      x,seq_len,_ = process_batch(next(dataset),task)
      x = get_emb(embeddings_model, x)
      #breakpoint()
      return get_max_values(plain_model, (x, seq_len,_),task)


def get_plain_LUT(degree_lut:int,
                  n_lut:int,
                  index:int,
                  s:float,
                  gamma:float):
      ###sub_lut
      sub_degree = degree_lut//n_lut
      part_lut = jnp.linspace(-s, s, num=sub_degree, endpoint=False)
      values = jnp.where(part_lut>=gamma,1.,0.)
      
      ###Complete_LUT
      complete_LUT = jnp.zeros(degree_lut)
      base_range = jnp.arange(sub_degree)
      indices = base_range * n_lut + index
      complete_LUT = complete_LUT.at[indices].set(values)
      return coef_rotation(complete_LUT,-degree_lut//2)
