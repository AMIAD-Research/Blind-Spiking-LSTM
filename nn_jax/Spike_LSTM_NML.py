from flax import nnx
import jax.numpy as jnp
import jax
import numpy as np
from nn_jax.Linear import *
from nn_jax.Spike_LSTM import LSTMCell





class SpikeLSTMModel_NML(nnx.Module):
    def __init__(self,input_dim:int, hidden_dim:int, task:str="Single Sentence",dropout:float=0.5,n_lut=1, mlp_factor=4, rngs: nnx.Rngs=nnx.Rngs(0)):
        super().__init__()
        self.task = task
        self.lstm = LSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, n_lut=n_lut, rngs=rngs)
        self.dropout_out = nnx.Dropout(dropout,rngs=rngs)
        self.hidden_dim = hidden_dim
        if self.task == "Single Sentence":
            self.head = nnx.Sequential(
                LinearS1(hidden_dim,int(hidden_dim*mlp_factor),rngs=rngs),
                nnx.relu,
                LinearS1(int(hidden_dim*mlp_factor),1,rngs=rngs)
            )
        elif self.task == "Similarity and Paraphrase":
            self.head =nnx.Sequential(
                LinearS1(2*hidden_dim,int(hidden_dim*mlp_factor),rngs=rngs),
                nnx.relu,
                LinearS1(int(hidden_dim*mlp_factor),1,rngs=rngs),
            )
        
        
    def __call__(self,x): 
        _,_,h_all,_  = jax.vmap(self.lstm,in_axes=0)(x)
        y = self.dropout_out(h_all)
        if self.task == "Single Sentence":
            return self.head(y)
        elif self.task == "Similarity and Paraphrase":
            return y
    


    
