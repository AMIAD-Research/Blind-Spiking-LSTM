import numpy as np
import jax.numpy as jnp
from schemas.polynomial_jax import fft_polynomial_multiply, centered_mod
from schemas.format import Ciphertext,Plaintext,RGSW



def key_switch_noise(var_input:float,dict_params:dict,mean_sk:float,var_sk:float,var_ksk:float):
    """This function compute the estimator of the variance of the noise for the key switch operation

    Args:
        var_input (float): Variance of the input of the key switch
        dict_params (dict): dictionary gathering all the relevants informations of the scheme
        mean_sk (float): expectation of the secret key
        var_sk (float): variance of the secret key
        var_ksk (float): variance of the key switching key

    Returns:
        float: Esimt of the variance of the output of the key switch
    """
    q = dict_params["q"]
    beta = dict_params["beta_ks"]
    l = dict_params["l_ks"]
    n = dict_params["degree"]
    var = var_input + n*(q**2)*(var_sk+mean_sk**2)/(12*(beta**(2*l))) + n*l*var_ksk*(beta**2+2)/12 + n**2*l*beta**2*(2**11*q/2**64)**2/12
    return var


def mod_switch_noise(var_input:float,dict_params:dict,mean_sk:float,var_sk:float,new_modulus:float):
    """Returns the estimator of the variance of the noise for the modulus switch, in the vanilla case

    Args:
        var_input (float): variance of the input
        dict_params (dict): dictionary gathering all the relevants informations of the scheme
        mean_sk (float): expectation of the secret key
        var_sk (float): variance of the secret key
        new_modulus (float): New modulus 

    Returns:
        _type_: _description_
    """
    n = dict_params["degree"]
    q = dict_params["q"]
    return var_input*((new_modulus/q)**2)+n*(var_sk+mean_sk**2)/12


def mod_switch_collapse_noise(collapse:float, degree_lut:float, degree_lwe:float):
    """Returns the estimator of the variance of the noise for the modulus switch, in case of the collapse

    Args:
        collapse (int): Collapse
        degree_lut (int): Degree of the LUT
        degree_lwe (int): Degree of the input

    Returns:
        _type_: _description_
    """
    return (1-0.5**collapse)*degree_lwe/(12*collapse*(2*degree_lut)**2)



def blind_rotate_noise(degree_lwe:float, degree_lut:float, m:float, beta:float, l:float, q:float, var_bsk:float):
    """This function computes the variance of the noise after the blind rotate

    Args:
        degree_lwe (float): degree of the lwe ciphertext
        degree_lut (float): degree of the LUT
        m (float): Collapse
        beta (float): beta
        l (float): l
        q (float): modulus of the ciphertext
        var_bsk (float): Variance f the noise of the bootstrapping key

    Returns:
        _type_: _description_
    """
    return (degree_lwe/m)*(1+degree_lut/2)*q**2*beta**(-2*l)/12 + 2*degree_lut*(degree_lwe/m)*l*2**m*(beta**2+2)*var_bsk/12 #+ (degree_lwe/m)*degree_lut**2*l*beta**2*(2**11*q/2**64)**2/12


def bootstrapping_noise(var_input, dict_params, dict_params_ks_lwe, dict_params_lut, mean_sk, var_sk,
                        var_bsk, var_ksk, collapse):
    """This function computes the noise of the bootstrapping

    Args:
        var_input (_type_): _description_
        dict_params (_type_): _description_
        dict_params_ks_lwe (_type_): _description_
        dict_params_lut (_type_): _description_
        mean_sk (_type_): _description_
        var_sk (_type_): _description_
        var_bsk (_type_): _description_
        var_ksk (_type_): _description_
        collapse (_type_): _description_

    Returns:
        _type_: _description_
    """

    degree_lut = dict_params_lut["degree"]
    degree_lwe = dict_params_ks_lwe["degree"]
    q = dict_params["q"]
    beta_bs = dict_params_lut["beta"]
    l_bs = dict_params_lut["l"]
    ks_noise = key_switch_noise(var_input,dict_params_ks_lwe, mean_sk, var_sk,var_ksk) - var_input
    br_noise = blind_rotate_noise(degree_lwe, degree_lut, collapse, beta_bs, l_bs, q, var_bsk)
    mod_switch_noise = mod_switch_collapse_noise(collapse, degree_lut,degree_lwe)
    return var_input + ks_noise + br_noise + mod_switch_noise


def get_noise(ciphertext:Ciphertext,sk:Plaintext,dict_params:dict,message:Plaintext):
    n = dict_params["degree"]
    a,b = ciphertext[0],ciphertext[1]
    product = fft_polynomial_multiply(a,sk)
    c = centered_mod(b-product-message,dict_params["q"])
    return c

def get_noise_lwe(ciphertext:Ciphertext,sk:Plaintext,dict_params:dict,message:Plaintext):
    a,b = ciphertext[0],ciphertext[1]
    product = jnp.dot(sk,a)
    c = centered_mod(b-product-message,dict_params["q"])
    return c

def decrypt_no_delta(ciphertext,sk,dict_params):
    n = dict_params["degree"]
    a,b = ciphertext[0],ciphertext[1]
    product = fft_polynomial_multiply(a,sk,n)
    c = centered_mod(b-product,dict_params["q"])
    return c