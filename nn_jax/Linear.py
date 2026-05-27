from flax import nnx
import jax.numpy as jnp
import jax


class LinearS1(nnx.Module):
    def __init__(self,input_dim, output_dim, rngs):
        super().__init__()
        self.linear = nnx.Linear(input_dim,output_dim,rngs=rngs)
        self.linear.bias = jnp.zeros(output_dim)
    def __call__(self, x):
        #kernel = self.linear.kernel.value
        kernel = self.get_norm_kernel()
        return (x@kernel) #+ self.linear.bias

    
    def get_norm_kernel(self):
        kernel = self.linear.kernel.value
        norm = jnp.max(jnp.abs(kernel),axis=0,keepdims=True) + 1e-3
        kernel = kernel/norm
        return kernel
    
    def qforward(self,x,beta_w,q,s=None):
        if s is None:
            s = 1
            breakpoint()   
        kernel_norm = self.get_norm_kernel()
        kernel = jnp.round(kernel_norm * beta_w/s)
        return (x@kernel)*s/q


@jax.custom_vjp
def bwn_quant_op(x, low, high, theta, alpha=2.0):
    # Forward: high si x >= theta, sinon low
    return jnp.where(x >= theta, high, low)

def bwn_quant_fwd(x, low, high, theta, alpha):
    res = bwn_quant_op(x, low, high, theta, alpha)
    return res, (x, low, high, theta, alpha)

def bwn_quant_bwd(res, g):
    x, low, high, theta, alpha = res
    
    # 1. On centre le surrogate Cauchy sur le seuil theta
    diff = x - theta
    surrogate = 1.0 / (1.0 + jnp.square(alpha * diff))
    
    # 2. Gradient par rapport à x
    # Astuce de stabilité : on découple l'amplitude du saut pour éviter 
    # l'explosion des gradients dans le LSTM (on garde g * surrogate pur)
    grad_x = g * surrogate
    
    # 3. Gradients par rapport aux bornes (low et high)
    # Le gradient est routé uniquement vers la borne qui a été activée
    mask_high = jnp.where(x >= theta, 1.0, 0.0)
    mask_low = 1.0 - mask_high
    
    grad_low = g * mask_low
    grad_high = g * mask_high
    
    # 4. Gradient par rapport à theta (seuil)
    #breakpoint()
    grad_theta = -grad_x
    
    return grad_x, grad_low, grad_high, grad_theta, None

bwn_quant_op.defvjp(bwn_quant_fwd, bwn_quant_bwd)




def init_gamma_uniform(n_lut, dim, max_val=1):
    # Crée n_lut seuils espacés uniformément entre -max_val et max_val
    # Shape: (n_lut,)
    thresholds = jnp.linspace(-max_val, max_val, n_lut)
    # Broadcast sur la dimension de la couche pour obtenir (n_lut, dim)
    return jnp.broadcast_to(thresholds[:, None], (n_lut, dim))