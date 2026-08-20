from flax import nnx
import jax.numpy as jnp
import numpy as np
import jax
from jax import vmap
from nn_tfhe.activation import HeavysideBoostrapper
from schemas.format import Ciphertext, Plaintext
from schemas.text_jax import multiply_plaintext_ciphertext, rotate_ciphertext, sum_ciphertext_ciphertext,centered_mod_ciphertext, sum_plaintext_to_ciphertext
from schemas.RLWE_jax import sample_extract, get_delta
from schemas.polynomial_jax import weights_to_polynomial




class Linear(nnx.Module):
    def __init__(self, output_dim:int, dict_params:dict, factor_W:int):
        super().__init__()
        self.dict_params = dict_params
        self.delta = get_delta(dict_params)
        self.factor = factor_W
        self.degree = dict_params["degree"]
        self.kernel_polynomial = jnp.zeros((output_dim, self.degree))

        self.bias = jnp.zeros(output_dim)
        
    
    def set_weights(self, W:jax.Array,B:jax.Array):# B:jax.Array):
        """This function takes the weights of a linear layer operating on cleartext data and sets up the weights for the
        linear layer

        Args:
            W (jax.Array): Weights
        """

        weight_poly1 = (vmap(weights_to_polynomial, in_axes=0)(jnp.round(self.factor[:,None]*W.transpose())))
        weights_poly1 = np.zeros((W.transpose().shape[0],self.degree))
        weights_poly1[:,-W.transpose().shape[1]+1:] = weight_poly1[:,:-1]
        weights_poly1[:,0] = weight_poly1[:,-1]
        self.kernel_polynomial = weights_poly1

        self.bias = jnp.round(self.factor*B)

    

    
    def __call__(self, x:Ciphertext):
        """Function that applies a linear layer on homomorphic data

        Args:
            x (Ciphertext): Encrypted data of shape [(degree),(degree)]
        Returns an LWE [(hidden_size,degree),(hidden_size)]
        """
        ##Multiplication of the polynomials
        h1_cipher = vmap(multiply_plaintext_ciphertext,in_axes=(0,None,None,None))(self.kernel_polynomial, x, self.dict_params["degree"], False)

        ##Extraction of the first coefficient corresponding to the hidden layer
        h1_cipher_extract = vmap(sample_extract,in_axes=(0,None))(h1_cipher,0)
        #breakpoint()
        h1_cipher_extract = vmap(sum_plaintext_to_ciphertext,(0,0))(self.bias,h1_cipher_extract)

        return vmap(centered_mod_ciphertext,(0,None))(h1_cipher_extract,self.dict_params["q"])



class CipherSpikeLSTM:
    def __init__(self,
                dict_params:dict, dict_params_lut:dict,
                input_dim:int,
                hidden_dim:int, max_len:int, 
                linear_f:Linear,
                linear_i:Linear,
                linear_o:Linear,
                heavyBS:HeavysideBoostrapper,
                n_lut:int,
                lut_f:jnp.array,
                lut_i:Plaintext,
                lut_o:jnp.array,
                beta_x:float=2**53,
                
                  ):
        super().__init__()
        self.dict_params = dict_params
        self.degree_lut = dict_params_lut["degree"]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.beta_x = beta_x


        self.W_f = linear_f
        self.W_i = linear_i
        self.W_o = linear_o

        self.heavyBS = heavyBS
        self.n_lut = n_lut
        self.lut_f = lut_f
        self.lut_i = lut_i
        self.lut_o = lut_o



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
            C_lwe_reshape = (
                C_lwe[0].reshape(self.n_lut, self.hidden_dim // self.n_lut, self.degree_lut), 
                C_lwe[1].reshape(self.n_lut, self.hidden_dim // self.n_lut, 1)
            )
            
            ## Concat shape (degree)
            HX = sum_ciphertext_ciphertext(x_t, H, self.dict_params["q"])

            ## Linear, output's shape [(hidden_size,degree),(hidden_size,1)]
            F_t = self.W_f(HX)
            I_t = self.W_i(HX)
            O_t = self.W_o(HX)

            ## First BS
            lut_C_t_1 = vmap(vmap(self.heavyBS.prepare_LUT, (0, 0)), (0, 0))(C_lwe_reshape, self.lut_f)
            lut_C_t_1 = centered_mod_ciphertext(
                (jnp.sum(lut_C_t_1[0], axis=0), jnp.sum(lut_C_t_1[1], axis=0)), self.dict_params["q"]
            )
            #breakpoint()
            input_boot = [jnp.concat([F_t[0], I_t[0]], axis=0), jnp.concat([F_t[1], I_t[1]], axis=0)]
            lut_boot = [jnp.concat([lut_C_t_1[0], self.lut_i[0]], axis=0), jnp.concat([lut_C_t_1[1], self.lut_i[1]], axis=0)]
            #breakpoint()
            boot = self.heavyBS.cuboot(input_boot, lut_boot)
            
            # Applying your reshaping
            spike_I = [
                jnp.permute_dims(boot[0][self.hidden_dim // self.n_lut:], (1, 0, 2)).reshape(-1, self.degree_lut),
                jnp.permute_dims(boot[1][self.hidden_dim // self.n_lut:], (1, 0, 2)).reshape(-1, 1)
            ]
            spike_F_C = [
                jnp.permute_dims(boot[0][:self.hidden_dim // self.n_lut], (1, 0, 2)).reshape(-1, self.degree_lut),
                jnp.permute_dims(boot[1][:self.hidden_dim // self.n_lut], (1, 0, 2)).reshape(-1, 1)
            ]

            ## Sum
            C_t = vmap(sum_ciphertext_ciphertext, (0, 0, None))(spike_F_C, spike_I, self.dict_params["q"])
            C_t_reshape = (
                C_t[0].reshape(self.n_lut, self.hidden_dim // self.n_lut, self.degree_lut), 
                C_t[1].reshape(self.n_lut, self.hidden_dim // self.n_lut, 1)
            )
            
            ## Third BS
            lut_H = vmap(vmap(self.heavyBS.prepare_LUT, (0, 0)), (0, 0))(C_t_reshape, self.lut_o)
            lut_H = centered_mod_ciphertext(
                (jnp.sum(lut_H[0], axis=0), jnp.sum(lut_H[1], axis=0)), self.dict_params["q"]
            )

            H_t = self.heavyBS.cuboot(O_t, lut_H)
            H_t = [
                jnp.permute_dims(H_t[0], (1, 0, 2)).reshape(-1, self.degree_lut),
                jnp.permute_dims(H_t[1], (1, 0, 2)).reshape(-1, 1)
            ]

            ## Post treatment
            C_t = self.heavyBS.packing(C_t)
            H_t = self.heavyBS.packing(H_t)
            H_t = rotate_ciphertext(H_t, self.input_dim)
            # --- End of your computation step ---

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