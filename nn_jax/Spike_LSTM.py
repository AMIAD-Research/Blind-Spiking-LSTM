from flax import nnx
import jax.numpy as jnp
import jax
import numpy as np
from nn_jax.Linear import *


 
class LSTMCell(nnx.Module):
    def __init__(self, input_dim:int, hidden_dim:int, n_lut:int=4, rngs: nnx.Rngs=nnx.Rngs(0)):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_len = 128
        self.n_lut = n_lut
        self.linearf = LinearS1(input_dim + hidden_dim,hidden_dim//n_lut,rngs=rngs)
        self.lineari = LinearS1(input_dim + hidden_dim,hidden_dim//n_lut,rngs=rngs)
        self.linearo = LinearS1(input_dim + hidden_dim,hidden_dim//n_lut,rngs=rngs)
        #self.eta = nnx.Param(jnp.ones(hidden_dim)/self.max_len)
        # self.eta = jnp.ones(hidden_dim)/self.max_len
        # self.eta = nnx.Param(jnp.ones((n_lut, hidden_dim//n_lut))/self.max_len)
        self.eta = nnx.Param(jnp.array(
            np.random.normal(0,1,(n_lut, hidden_dim//n_lut))/hidden_dim))
        
        norm = 1.

        self.gammaf = init_gamma_uniform(n_lut, hidden_dim//n_lut,norm)

        self.gammai = init_gamma_uniform(n_lut, hidden_dim//n_lut,norm)

        self.gammao = init_gamma_uniform(n_lut, hidden_dim//n_lut,norm)
        
        
        self.alpha = 2.5
    
    def spike(self, x, low=0.0, high=1.0,theta=0):

        return jax.vmap(bwn_quant_op, in_axes=(0, None, None, 0, None))(
                x, low, high, theta, self.alpha
            )



    def __call__(self, x):



        # 1. Pre-allocation of the output tensor (essential with fori_loop)
        out_h = jnp.zeros((self.max_len, self.hidden_dim))
        out_c = jnp.zeros((self.max_len, self.hidden_dim))


        # 3. Internal function with the expected signature (index, state)
        def body_fun(i, val):
            h, c, out_h, out_c = val
            x_t = x[i]  # Explicit extraction of the time step
            
            hx = jnp.concat([x_t, h], axis=-1)
            
            f_t = self.linearf(hx)
            i_t = self.lineari(hx)
            o_t = self.linearo(hx)

            f_t = jnp.concat([
                self.spike(f_t,0.,1.,self.gammaf[k]) for k in range(self.n_lut)
            ])
            i_t = jnp.concat([
                self.spike(i_t,0.,1.,self.gammai[k])*self.eta[k] for k in range(self.n_lut)
            ])
            o_t = jnp.concat([
                self.spike(o_t,0.,1.,self.gammao[k]) for k in range(self.n_lut)
            ])
                
            # Computation of the next states (to adjust according to your exact spiking logic)
            c_next = f_t * c + i_t 
            h_next = o_t * c_next

            # 4. In-place storage via dynamic_update_slice (compiled in-place by XLA)
            out_h = out_h.at[i].set(h_next)
            out_c = out_c.at[i].set(c_next)
            return h_next, c_next, out_h, out_c

        # 5. Lancement de la boucle JAX
        init_val = (jnp.zeros(self.hidden_dim),jnp.zeros(self.hidden_dim), out_h, out_c)
        h_final, c_final, h_all, c_all = jax.lax.fori_loop(0, self.max_len, body_fun, init_val)

        return h_final, c_final, h_all, c_all
    
    def set_s(self,sf,si,so):
        self.sf = sf
        self.si = si
        self.so = so
    
    def qforward(self, x, beta_x, q, noise_input, noise_output):
        beta_w = q / (4*beta_x)
        out_h = jnp.zeros((self.max_len, self.hidden_dim))
        out_c = jnp.zeros((self.max_len, self.hidden_dim))


        # 3. Internal function with the expected signature (index, state)
        def body_fun(i, val):
            h, c, out_h, out_c = val
            x_t = x[i]  # Explicit extraction of the time step

            hx = jnp.concat([x_t, h], axis=-1)
            HX = jnp.round(hx*beta_x)
            f_t = self.linearf.qforward(HX,beta_w,q,self.sf)*4
            i_t = self.lineari.qforward(HX,beta_w,q,self.si)*4
            o_t = self.linearo.qforward(HX,beta_w,q,self.so)*4
            #if not self.dropout.deterministic:
            
            index_f_sup = jnp.concat([jnp.where(f_t > self.sf,1.,0.) for _ in range(self.n_lut)])
            index_f_inf = jnp.concat([jnp.where(f_t < -self.sf,1.,0.) for _ in range(self.n_lut)])
            index_i_sup = jnp.concat([jnp.where(i_t > self.si,1.,0.) for _ in range(self.n_lut)])
            index_i_inf = jnp.concat([jnp.where(i_t < -self.si,1.,0.) for _ in range(self.n_lut)])
            index_o_sup = jnp.concat([jnp.where(o_t > self.so,1.,0.) for _ in range(self.n_lut)])
            index_o_inf = jnp.concat([jnp.where(o_t < -self.so,1.,0.) for _ in range(self.n_lut)])
            f_t = f_t + noise_input[i,0]
            i_t = i_t + noise_input[i,1]
            o_t = o_t + noise_input[i,2]
            f_t = jnp.concat([
                self.spike(f_t,0.,1.,self.gammaf[i]) for i in range(self.n_lut)
            ])
            i_t = jnp.concat([
                self.spike(i_t,0.,1.,self.gammai[i])*self.eta[i] for i in range(self.n_lut)
            ])
            o_t = jnp.concat([
                self.spike(o_t,0.,1.,self.gammao[i]) for i in range(self.n_lut)
            ])
            #breakpoint()
            f_t = f_t * (1-index_f_sup - index_f_inf) - 0 * index_f_sup - 1 * index_f_inf
            i_t = i_t * (1-index_i_sup - index_i_inf) - 0 * index_i_sup - jnp.concat(self.eta) * index_i_inf
            o_t = o_t * (1-index_o_sup - index_o_inf) - 0 * index_o_sup - 1 * index_o_inf
            # Computation of the next states (to adjust according to your exact spiking logic)
            c_next = f_t * c + noise_output[i,0] + i_t + noise_output[i,1]
            h_next = o_t * c_next + noise_output[i,2]

            # 4. In-place storage via dynamic_update_slice (compiled in-place by XLA)
            out_h = out_h.at[i].set(h_next)
            out_c = out_c.at[i].set(c_next)
            return h_next, c_next, out_h, out_c

        # 5. Lancement de la boucle JAX
        init_val = (jnp.zeros(self.hidden_dim),jnp.zeros(self.hidden_dim), out_h, out_c)
        h_final, c_final, h_all, c_all = jax.lax.fori_loop(0, self.max_len, body_fun, init_val)

        return h_final, c_final, h_all, c_all

    def set_qgamma(self,sf,si,so, sub_degree):
        for i in range(self.n_lut):
            new_gammaf = jnp.max(jnp.stack([self.gammaf[i],2*sf*jnp.ceil(self.gammaf[i]*sub_degree/(2*sf))/sub_degree]),axis=0)
            new_gammai = jnp.max(jnp.stack([self.gammai[i],2*si*jnp.ceil(self.gammai[i]*sub_degree/(2*si))/sub_degree]),axis=0)
            new_gammao = jnp.max(jnp.stack([self.gammao[i],2*so*jnp.ceil(self.gammao[i]*sub_degree/(2*so))/sub_degree]),axis=0)
            self.gammaf = self.gammaf.at[i].set(new_gammaf)
            self.gammai = self.gammai.at[i].set(new_gammai)
            self.gammao = self.gammao.at[i].set(new_gammao)





class SpikeLSTMModel(nnx.Module):
    def __init__(self,input_dim:int, hidden_dim:int, task:str="Single Sentence",dropout:float=0.5,n_lut=1, rngs_forward: nnx.Rngs=nnx.Rngs(0)):
        super().__init__()
        self.task = task
        self.lstm = LSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, n_lut=n_lut, rngs=rngs_forward)
        self.n_lut = n_lut
        self.dropout_out = nnx.Dropout(dropout,rngs=rngs_forward)
        self.hidden_dim = hidden_dim
        if self.task == "Single Sentence":
            self.head = nnx.Sequential(
                LinearS1(hidden_dim,hidden_dim,rngs=rngs_forward),
                nnx.relu,
                LinearS1(hidden_dim,1,rngs=rngs_forward)
            )
        elif self.task == "Similarity and Paraphrase":
            self.head =nnx.Sequential(
                LinearS1(2*hidden_dim,hidden_dim,rngs=rngs_forward),
                nnx.relu,
                LinearS1(hidden_dim,1,rngs=rngs_forward),
            )
        
        
    def __call__(self,x): 
        _,_,h_all,_  = jax.vmap(self.lstm,in_axes=0)(x)
        y = self.dropout_out(h_all)
        if self.task == "Single Sentence":
            return self.head(y)
        elif self.task == "Similarity and Paraphrase":
            return y
    

    def qforward(self,x, beta,q, noise_lstm_input, noise_lstm_output, noise_mlp):
        _,_,h_all,_  = jax.vmap(self.lstm.qforward,in_axes=(0,None,None,0,0))(x, beta, q, noise_lstm_input,  noise_lstm_output)
        y = h_all
        if self.task == "Single Sentence":
            y = self.qhead(y, beta, q, noise_mlp)
            return y
        elif self.task == "Similarity and Paraphrase":
            return y

    def set_s(self,shead1,sout):
        self.shead1 = shead1
        self.sout = sout

    def qhead(self,x, beta,q, noise_mlp_input, noise_mlp_output):
        beta_w = q / (4*beta)
        y = jnp.round(x*beta)
        y = self.head.layers[0].qforward(y,beta_w,q,self.shead1)*4
        index_y_sup = jnp.where(y>self.shead1, 1.,0.)
        index_y_inf = jnp.where(y<-self.shead1, 1.,0.)
        y_relu = nnx.relu(y+noise_mlp_input) 
        #y = y_relu + noise_mlp_output#- index_y_inf*y_relu -
        y = y_relu * (1 - index_y_sup - index_y_inf) + 0*index_y_sup - self.shead1.flatten()*index_y_inf + noise_mlp_output
        y = jnp.round(y*beta)
        y = self.head.layers[2].qforward(y,beta_w,q, self.sout)*4
        return y
    
