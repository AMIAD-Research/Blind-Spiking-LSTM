from schemas.polynomial_jax import centered_mod,  GLEV_polynomial, decomposition
from schemas.text_jax import gadget_product,sum_ciphertext_ciphertext, modulus_switch,blind_rotate, sample_extract
from schemas.format import Ciphertext,Plaintext,RGSW
import jax.numpy as jnp
from functools import partial
import numpy as np
from typing import Tuple
import jax
from jax import vmap


def encrypt_LWE(m:Plaintext, sk:jnp.ndarray,dict_params:dict,key:jnp.ndarray)->Ciphertext:
    """Function to compute the LWE encryption of a plaintext m using the private key sk

    Args:
        m (Plaintext): Message to encrypt
        sk (Plaintext): Private key
        dict_params (dict): Dictionary grouping the encryption's meta-parameters

    Returns:
        Ciphertext: Encrypted message
    """
    N = dict_params["degree"]
    q = dict_params["q"]
    sigma = dict_params["sigma"]
    public_key = jax.random.randint(key,N,-q/2,q/2)*1.
    e = jnp.round(jax.random.normal(key,1))*sigma
    #We compute the ciphertext B, but we don't want to get the gradient of this operation
    product = jnp.dot(sk,public_key)
    b = centered_mod(product + jax.lax.stop_gradient(m) + e,q)
    return public_key,b

def decrypt_LWE(ciphertext:Ciphertext,sk:Plaintext, dict_params:dict )->Plaintext:
    """This function decrypts a ciphertext

    Args:
        c (ciphertext): ciphertext to decrypt
        sk (polynomial, optional): the private key to use for decryption.
        dict_params (dict): Dictionary grouping the encryption's meta-parameters

    Returns:
        plaintext: Cleartext message
    """
    t = dict_params["t"]
    delta = dict_params["q"]/t
    public_key, b  = ciphertext[0],ciphertext[1]
        
    product = jnp.dot(public_key,sk)
    decrypted_plaintext = centered_mod(
                                    jnp.round((b-product)/delta),t)
    return decrypted_plaintext



def decrypt_LWE_quantization(ciphertext:Ciphertext, sk:Plaintext, dict_params:dict, scale:float=None) ->Plaintext:
    
    t = dict_params["t"]
    q = dict_params["q"]
    if scale is None:
        scale = q/t
    public_key, b  = ciphertext[0],ciphertext[1]
        
    product = jnp.dot(public_key,sk)
    decrypted_plaintext = centered_mod(b-product,q)/scale
    return decrypted_plaintext


@partial(jax.jit, static_argnames=['q'])
def gadget_product_LWE(glev:Ciphertext, decomp:Plaintext,q:int)->Ciphertext:
    """This function applies the product related to the gadget decomposition.

    Args:
        glev (Ciphertext): Ciphertext in GLEV
        decomp (Plaintext): Plaintext decomposed following a base beta
        q (int): Modulus of the ciphertext space
        degree (int): Degree of the glev polynomials

    Returns:
        Ciphertext: Result of the product (close to a scalar product)
    """
    glev_public_key, glev_b = glev[0],glev[1]
    public_key_sum = centered_mod(jnp.tensordot(decomp,glev_public_key),q)
    
    b_sum = centered_mod(jnp.tensordot(decomp,glev_b),q)
    return public_key_sum,b_sum



def get_keyswitch_sk_LWE(new_sk:Plaintext, old_sk:Plaintext, dict_params:dict,key:jnp.ndarray):
    beta = dict_params["beta_ks"]
    l = dict_params["l_ks"]
    q = dict_params["q"]
    glev_old_sk = GLEV_polynomial(old_sk, beta,l,q)
    key_glev = jax.random.split(key,(l,old_sk.shape[-1]))
    glev_sk = vmap(vmap(encrypt_LWE,in_axes=(0,None,None,0)),in_axes=(0,None,None,0))(-glev_old_sk,new_sk,dict_params,key_glev)
    return glev_sk



def key_switch_LWE(key_switching_key:Ciphertext, ciphertext:Ciphertext,dict_params:dict )->Ciphertext:
    """This function switches keys for an encrypted message

    Args:
        new_sk (Plaintext): New private key to use
        ciphertext (Ciphertext): Message encrypted with the old private key
        old_sk (Plaintext): Old private key
        dict_params (dict): Dictionary grouping the encryption's meta-parameters

    Returns:
        _type_: _description_
    """
    beta = dict_params["beta_ks"]
    l = dict_params["l_ks"]
    q = dict_params["q"]
    public_key = ciphertext[0]
    b = ciphertext[1]

    decomposition_public_key = decomposition(public_key,beta,l,q)
    ciphertext2 = gadget_product_LWE(key_switching_key,decomposition_public_key,q)
    return centered_mod(ciphertext2[0],q),centered_mod(b+ciphertext2[1],q)