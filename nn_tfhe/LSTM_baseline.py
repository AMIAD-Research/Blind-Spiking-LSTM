from flax import nnx
import jax.numpy as jnp
import numpy as np
import jax
from jax import vmap
from nn_tfhe.activation import HeavysideBoostrapper
from nn_tfhe.Spike_LSTM import Linear
from schemas.format import Ciphertext, Plaintext
from schemas.text_jax import multiply_plaintext_ciphertext, rotate_ciphertext, sum_ciphertext_ciphertext,centered_mod_ciphertext, sum_plaintext_to_ciphertext
from schemas.RLWE_jax import sample_extract, get_delta, cuboot_merge
from schemas.polynomial_jax import weights_to_polynomial, centered_mod
from schemas.LWE_jax import key_switch_LWE




gamma_sigmoid = 4
gamma_tanh = 4




class CipherLSTM:
    def __init__(self,
                dict_params:dict, dict_params_lut:dict,
                input_dim:int,
                hidden_dim:int, max_len:int, 
                linear_f:Linear,
                linear_c:Linear,
                linear_i:Linear,
                linear_o:Linear,
                heavyBS:HeavysideBoostrapper,
                n_lut:int,
                beta_x:float=2**53,
                
                  ):
        super().__init__()
        self.dict_params = dict_params
        self.dict_params_lut = dict_params_lut
        self.degree_lut = dict_params_lut["degree"]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.n_lut = n_lut
        x = jnp.linspace(-5,5,self.degree_lut)
        x_square = 0.25*x**2
        self.LUT_sigmoid = [jnp.zeros((self.hidden_dim//self.n_lut, self.degree_lut)), beta_x * jnp.stack([jnp.round(gamma_sigmoid * jax.nn.sigmoid(x)) for _ in range(self.hidden_dim//self.n_lut)])]
        self.LUT_tanh = [jnp.zeros((self.hidden_dim//self.n_lut, self.degree_lut)), beta_x * jnp.stack([jnp.round(gamma_tanh * jax.nn.tanh(x)) for _ in range(self.hidden_dim//self.n_lut)])]
        self.LUT_square = [jnp.zeros((self.hidden_dim, self.degree_lut)), beta_x * jnp.stack([jnp.round(beta_x * x_square) for _ in range(self.hidden_dim)])]
        self.heavyBS = heavyBS
        self.beta_x = beta_x


        self.W_f = linear_f
        self.W_i = linear_i
        self.W_c = linear_c
        self.W_o = linear_o



        self.bootstrapping_key = self.heavyBS.bootstrapping_key
        self.key_switching_key_bs = self.heavyBS.key_switching_key_bs
        self.dict_params_ks_LWE = self.heavyBS.dict_params_ks_LWE
        self.collapse = self.heavyBS.collapse
        self.all_rot_possible_fourier = self.heavyBS.all_rot_possible_fourier
        self.ksk_packing = self.heavyBS.ksk_packing

    def multiply(self,x,y):
        plus =  [
                    centered_mod(x[0] + y[0],self.dict_params["q"]),
                    centered_mod(x[1] + y[1],self.dict_params["q"])
                ]
        plus_ks = vmap(key_switch_LWE,(None,0,None))(self.key_switching_key_bs, plus, self.dict_params_ks_LWE )
        minus = [
                        centered_mod(x[0] - y[0],self.dict_params["q"]),
                        centered_mod(x[1] - y[1],self.dict_params["q"])
                ]
        minus_ks = vmap(key_switch_LWE,(None,0,None))(self.key_switching_key_bs, minus, self.dict_params_ks_LWE )
        
        boot_plus = cuboot_merge(plus_ks, self.LUT_square, self.bootstrapping_key,
                                                                    self.dict_params_lut["q"],
                                                                    self.dict_params_lut["beta_bs"],
                                                                    self.dict_params_lut["l_bs"],
                                                                    self.dict_params_lut["degree"],
                                                                    self.collapse,
                                                                    self.all_rot_possible_fourier,
                                                                    1)
        
        boot_minus = cuboot_merge(minus_ks, self.LUT_square, self.bootstrapping_key,
                                                                    self.dict_params_lut["q"],
                                                                    self.dict_params_lut["beta_bs"],
                                                                    self.dict_params_lut["l_bs"],
                                                                    self.dict_params_lut["degree"],
                                                                    self.collapse,
                                                                    self.all_rot_possible_fourier,
                                                                    1)
        output = [
                    centered_mod(boot_plus[0][:,0] + boot_minus[0][:,0],self.dict_params["q"]),
                    centered_mod(boot_plus[1][:,0] + boot_minus[1][:,0],self.dict_params["q"])
                ]

    
        return output

    def bootstrapping(self, input_bs, LUT):
        input_ks = vmap(key_switch_LWE,(None,0,None))(self.key_switching_key_bs, input_bs, self.dict_params_ks_LWE)
        boot = cuboot_merge(input_ks, LUT, self.bootstrapping_key,
                                                            self.dict_params_lut["q"],
                                                            self.dict_params_lut["beta_bs"],
                                                            self.dict_params_lut["l_bs"],
                                                            self.dict_params_lut["degree"],
                                                            self.collapse,
                                                            self.all_rot_possible_fourier,
                                                            1)
        output = [boot[0][:,0], boot[1][:,0]]
        return output
    def __call__(self, x,seq_len):
        # Initialization of the hidden states (the initial "carry")
        H_0 = (jnp.zeros(self.dict_params["degree"]), jnp.zeros(self.dict_params["degree"]))
        C_0 = (jnp.zeros(self.dict_params["degree"]), jnp.zeros(self.dict_params["degree"]))
        init_carry = (H_0, C_0)
        # Definition of the function that will be executed at each time step
        def scan_step(carry, x_t):
            H, C = carry

            # --- Start of your TFHE computation step ---
            dim = jnp.arange(self.hidden_dim)
            C_lwe = vmap(sample_extract, (None, 0))(C, dim)

            
            ## Concat shape (degree)
            HX = sum_ciphertext_ciphertext(x_t, H, self.dict_params["q"])

            ## Linear, output's shape [(hidden_size,degree),(hidden_size,1)]
            F_t = self.W_f(HX)
            I_t = self.W_i(HX)
            C_t = self.W_c(HX)
            O_t = self.W_o(HX)

            ## Sigmoids + tanh
            # input_sig = [jnp.concat([F_t[0], I_t[0], C_t[0], O_t[0]], axis=0), jnp.concat([F_t[1], I_t[1], C_t[1], O_t[1]], axis=0)]
            # lut_sig = [jnp.concat([self.LUT_sigmoid[0],self.LUT_sigmoid[0],self.LUT_tanh[0],self.LUT_sigmoid[0]], axis=0), jnp.concat([self.LUT_sigmoid[1],self.LUT_sigmoid[1],self.LUT_tanh[1],self.LUT_sigmoid[1]], axis=0)]
            # input_sig_ks = vmap(key_switch_LWE,(None,0,None))(self.key_switching_key_bs, input_sig, self.dict_params_ks_LWE)
            # #breakpoint()
            # boot = cuboot_merge(input_sig_ks, lut_sig, self.bootstrapping_key,
            #                                                 self.dict_params_lut["q"],
            #                                                 self.dict_params_lut["beta_bs"],
            #                                                 self.dict_params_lut["l_bs"],
            #                                                 self.dict_params_lut["degree"],
            #                                                 self.collapse,
            #                                                 self.all_rot_possible_fourier,
            #                                                 1)
            
            # sig_f = [jnp.concat([boot[0][:self.hidden_dim//self.n_lut,0] for _ in range(self.n_lut)],axis=0), 
            #         jnp.concat([boot[1][:self.hidden_dim//self.n_lut,0] for _ in range(self.n_lut)],axis=0)]
            # sig_i = [jnp.concat([boot[0][self.hidden_dim//self.n_lut:2*self.hidden_dim//self.n_lut,0] for _ in range(self.n_lut)],axis=0), 
            #         jnp.concat([boot[1][self.hidden_dim//self.n_lut:2*self.hidden_dim//self.n_lut,0] for _ in range(self.n_lut)],axis=0)]
            # tanh_c = [jnp.concat([boot[0][2*self.hidden_dim//self.n_lut:3*self.hidden_dim//self.n_lut,0] for _ in range(self.n_lut)],axis=0), 
            #         jnp.concat([boot[1][2*self.hidden_dim//self.n_lut:3*self.hidden_dim//self.n_lut,0] for _ in range(self.n_lut)],axis=0)]
            # sig_o = [jnp.concat([boot[0][3*self.hidden_dim//self.n_lut:,0] for _ in range(self.n_lut)],axis=0),
            #         jnp.concat([boot[1][3*self.hidden_dim//self.n_lut:,0] for _ in range(self.n_lut)],axis=0)]
            boot = self.bootstrapping(F_t, self.LUT_sigmoid)
            sig_f = [jnp.concat([boot[0] for _ in range(self.n_lut)],axis=0), 
                    jnp.concat([boot[1] for _ in range(self.n_lut)],axis=0)]
            
            boot = self.bootstrapping(I_t, self.LUT_sigmoid)
            sig_i = [jnp.concat([boot[0] for _ in range(self.n_lut)],axis=0), 
                    jnp.concat([boot[1] for _ in range(self.n_lut)],axis=0)]
            
            boot = self.bootstrapping(C_t, self.LUT_tanh)
            tanh_c = [jnp.concat([boot[0] for _ in range(self.n_lut)],axis=0), 
                    jnp.concat([boot[1] for _ in range(self.n_lut)],axis=0)]
            
            boot = self.bootstrapping(O_t, self.LUT_sigmoid)
            sig_o = [jnp.concat([boot[0] for _ in range(self.n_lut)],axis=0), 
                    jnp.concat([boot[1] for _ in range(self.n_lut)],axis=0)]


            
            forget_update = self.multiply(sig_f,C_lwe)
            input_update = self.multiply(sig_i,tanh_c)

            out_C = [
                    centered_mod(forget_update[0] + input_update[0],self.dict_params["q"]),
                    centered_mod(forget_update[1]+ input_update[1],self.dict_params["q"])
                ]
            out_C_ks = vmap(key_switch_LWE,(None,0,None))(self.key_switching_key_bs, out_C, self.dict_params_ks_LWE)
            LUT_tanh_C = [jnp.zeros((self.hidden_dim,self.degree_lut)), self.beta_x * jnp.stack([jnp.round(gamma_tanh * jnp.linspace(-5,5,self.degree_lut)) for _ in range(self.hidden_dim)])]
            boot = cuboot_merge(out_C_ks, LUT_tanh_C, self.bootstrapping_key,
                                                            self.dict_params_lut["q"],
                                                            self.dict_params_lut["beta_bs"],
                                                            self.dict_params_lut["l_bs"],
                                                            self.dict_params_lut["degree"],
                                                            self.collapse,
                                                            self.all_rot_possible_fourier,
                                                            1)
            tanh_out_c = boot[0][:,0], boot[1][:,0]
            out_H = self.multiply(tanh_out_c, sig_o)
            #breakpoint()
            ## Post treatment
            C_t = self.heavyBS.packing(out_C)
            H_t = self.heavyBS.packing(out_H)
            #breakpoint()
            H_t = rotate_ciphertext(H_t, self.input_dim)
            #breakpoint()
            # --- End of your computation step --
            new_carry = (H_t, C_t)

            # We return the new state AND what we want to stack for the history
            return new_carry, (H_t, C_t)

        # Execution of the XLA-optimized loop
        # jax.lax.scan will automatically iterate over the first dimension of 'x' (time)
        final_carry, final_outputs = jax.lax.scan(scan_step, init_carry, x)
        
        return final_carry, final_outputs


class CipherMLP(nnx.Module):
    def __init__(self,
                dict_params:dict, dict_params_lut:dict, 
                hidden_dim:int,
                linear1:Linear,
                linear2:Linear,
                reluBS):
        
        self.dict_params = dict_params
        self.dict_params_lut = dict_params_lut
        self.hidden_dim = hidden_dim
        self.linear1 = linear1
        self.linear2 = linear2
        self.reluBS = reluBS
    
    def __call__(self, H_cipher):
        H1 = self.linear1(H_cipher)
        Y1 = self.reluBS(H1)
        H2 = self.reluBS.packing(Y1)
        Y2 = self.linear2(H2)
        return Y2