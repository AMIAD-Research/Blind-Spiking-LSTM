from schemas.polynomial_jax import centered_mod, exact_polynomial_multiply, fft_polynomial_multiply, GLEV_polynomial, decomposition, apply_automorphism
from schemas.text_jax import gadget_product,sum_ciphertext_ciphertext, modulus_switch,blind_rotate, sample_extract, all_binary_vectors, rotate_ciphertext
from schemas.format import Ciphertext,Plaintext,RGSW
import jax.numpy as jnp
from functools import partial
import numpy as np
from typing import Tuple
import jax
from jax import vmap
import math
from cuTFHE.multiply import multiply_seq_monomial as cuTFHE_multiply_seq_monomial
import time
jax.config.update('jax_enable_x64', True)


    

    
def get_delta(dict_params):
    q = dict_params["q"]
    t = dict_params["t"]
    return jnp.round(q/t)



def sample_sk(dict_params):
    degree = dict_params["degree"]
    
    #sk = np.random.choice([-1.,0.,1.],(degree,))
    sk = np.random.choice([0.,1.],(degree,))
    return jnp.array(sk)






def encrypt(m:Plaintext, sk:Plaintext,dict_params:dict,key:jnp.ndarray)->Ciphertext:
    """Cette méthode permet d'encrypter un message en clair

    Args:
        m (plaintext): plaintext de shape (K,1) avec K un entier arbitraire

    Returns:
        torch.tensor: tenseur de taille (K,n+1). La première feature est la clé publique, la seconde est b
    """
    
    degree = dict_params["degree"]
    q = dict_params["q"]
    sigma = dict_params["sigma"]
    public_key = jax.random.randint(key,degree,-q/2,q/2)*1.
    e = jnp.round(jax.random.normal(key,(degree,))*sigma)
    #We compute the ciphertext B, but we don't want to get the gradient of this operation
    product = jnp.round(fft_polynomial_multiply(public_key,sk))
    b = centered_mod(product + m + e,q)
    return public_key,b
    
def decrypt(ciphertext:Ciphertext,sk:Plaintext, dict_params:dict )->Plaintext:
    """_summary_

    Args:
        c (ciphertext): ciphertext à déchiffrer
        sk (polynomial, optional): la clé privée à utiliser pour déchiffrer. Si égal à None, on utilise la clé contenue dans le schéma. Defaults to None.
        divide_delta (bool, optional): Indique s'il on doit diviser par delta lors du déchiffrage. Defaults to True.

    Returns:
        plaintext: Message clair
    """
    t = dict_params["t"]
    degree = dict_params["degree"]
    delta = dict_params["q"]/t
    public_key, b  = ciphertext[0],ciphertext[1]
        
    product = fft_polynomial_multiply(public_key,sk)
    decrypted_plaintext = centered_mod(
                                    jnp.round((b-product)/delta),t)

    return decrypted_plaintext

def decrypt_quantization(ciphertext:Ciphertext, sk:Plaintext, dict_params:dict, scale:float=None) ->Plaintext:
    
    t = dict_params["t"]
    degree = dict_params["degree"]
    q = dict_params["q"]
    if scale is None:
        scale = q/t
    public_key, b  = ciphertext[0],ciphertext[1]
        
    product = fft_polynomial_multiply(public_key,sk)
    decrypted_plaintext = centered_mod(b-product,q)/scale
    return decrypted_plaintext

def get_keyswitch_sk(new_sk:Plaintext, old_sk:Plaintext, dict_params:dict,key:jnp.ndarray):
    beta = dict_params["beta_ks"]
    l = dict_params["l_ks"]
    q = dict_params["q"]
    glev_old_sk = GLEV_polynomial(old_sk, beta,l,q)
    key_glev = jax.random.split(key,(l))
    glev_sk = vmap(encrypt,in_axes=(0,None,None,0))(-glev_old_sk,new_sk,dict_params,key_glev)
    return glev_sk

def key_switch(key_switching_key:Ciphertext, ciphertext:Ciphertext,dict_params:dict )->Ciphertext:
    """Cette fonction permet de changer de clé pour un message chiffré

    Args:
        new_sk (Plaintext): Nouvelle clé privée à utiliser
        ciphertext (Ciphertext): Message chiffré par l'ancienne clé privée
        old_sk (Plaintext): Ancienne clé privée
        dict_params (dict): Dictionnaire regroupant les méta-paramètres du chiffrement

    Returns:
        _type_: _description_
    """
    beta = dict_params["beta_ks"]
    l = dict_params["l_ks"]
    q = dict_params["q"]
    degree = dict_params["degree"]
    public_key = ciphertext[0]
    b = ciphertext[1]

    decomposition_public_key = decomposition(public_key,beta,l,q)
    new_public_key = jnp.zeros_like(public_key,dtype=jnp.float64)
    ciphertext1 = new_public_key,b
    ciphertext2 = gadget_product(key_switching_key,decomposition_public_key,degree)
    return sum_ciphertext_ciphertext(ciphertext1,ciphertext2,q)


 
def get_RGSW(sk:jnp.ndarray, polynomial:Plaintext, dict_params:dict, key:jnp.ndarray)->RGSW:
    """Cette fonction permet de calculer le RGSW d'un plaintex

    Args:
        sk (jnp.ndarray): Clé privée
        polynomial (Plaintext): Plaintext à transformer en RGSW
        dict_params (dict): Dictionnaire regroupant les méta-paramètres du chiffrement

    Returns:
        RGSW: _description_
    """
    q = dict_params["q"]
    beta = dict_params["beta_rgsw"]
    l = dict_params["l_rgsw"]
    ##Right part of RGSW
    right_key = jax.random.split(key,(l))
    poly_glev = GLEV_polynomial(polynomial,beta,l,q)
    encrypted_poly_glev = vmap(encrypt,in_axes=(0,None,None,0))(poly_glev,sk,dict_params,right_key)
    ##Left part of RGSW
    left_key = jax.random.split(key,(l))
    sk_times_poly = fft_polynomial_multiply(-sk,polynomial)
    sk_times_poly_glev = centered_mod(GLEV_polynomial(sk_times_poly,beta,l,q),q)
    encrypted_sk_times_poly_glev= vmap(encrypt,in_axes=(0,None,None,0))(sk_times_poly_glev,sk,dict_params, left_key)
    return encrypted_sk_times_poly_glev, encrypted_poly_glev


def get_boostrapping_key(sk_lut:jnp.ndarray, sk_lwe:jnp.ndarray, dict_params:dict,key:jnp.ndarray)->RGSW:
    """Cette fonction calcule la clés de bootstrappint

    Args:
        sk_lut (jnp.ndarray): Clé privée de la LUT
        sk_lwe (jnp.ndarray): Clé privée du chiffré en entrée du bootstrapping
        dict_params (dict): Ensemble des paramètres permettant de calculer les chiffrements
        key (jnp.ndarray): randomness

    Returns:
        RGSW: _description_
    """
    q = dict_params["q"]
    beta = dict_params["beta_bs"]
    l = dict_params["l_bs"]
    
    ##Left part of RGSW
    matrix = jnp.matmul(
                        jnp.expand_dims(sk_lwe,1),jnp.expand_dims(-sk_lut,0))
    
    left_key = jax.random.split(key,(sk_lwe.shape[-1],l))
    left_part_rgsw_encrypted = centered_mod(
                    vmap(GLEV_polynomial,in_axes=(0,None,None,None))(matrix,beta,l,q),q
    )
    left_part_rgsw = vmap(vmap(encrypt,in_axes=(0,None,None,0)),in_axes=(0,None,None,0))(left_part_rgsw_encrypted,sk_lut,dict_params,left_key)
    ##Right part of RGSW
    poly = jnp.zeros((sk_lwe.shape[-1],sk_lut.shape[-1]))
    poly = poly.at[:,0].set(sk_lwe)
    right_key = jax.random.split(left_key[0,0],(sk_lwe.shape[-1],l))
    right_part_rgsw_encrypted = centered_mod(
                    vmap(GLEV_polynomial,in_axes=(0,None,None,None))(poly,beta,l,q),q
    )
    right_part_rgsw = vmap(vmap(encrypt,in_axes=(0,None,None,0)),in_axes=(0,None,None,0))(right_part_rgsw_encrypted,sk_lut,dict_params,right_key)
    return left_part_rgsw,right_part_rgsw

def get_boostrapping_key_collapse(sk_poly:jnp.ndarray, sk_lwe:jnp.ndarray, dict_params:dict,key:jnp.ndarray,collapse:int)->RGSW:
    """Cette fonction permet de calculer la clé de bootstrapping avec le collapse
    Args:
        sk (jnp.ndarray): Clé privée
        polynomial (Plaintext): Plaintext à transformer en RGSW
        dict_params (dict): Dictionnaire regroupant les méta-paramètres du chiffrement

    Returns:
        RGSW: _description_
    """
    q = dict_params["q"]
    beta = dict_params["beta_bs"]
    l = dict_params["l_bs"]
    kronecker_matrix = all_binary_vectors(collapse)
    sk_lwe_compare = jnp.reshape(sk_lwe,(-1,collapse))
    eq = (sk_lwe_compare[:, None, :] == kronecker_matrix[None, :, :])
    all_eq = jnp.all(eq, axis=-1)*1
    all_eq = all_eq.flatten()
    ##Left part of RGSW
    matrix = jnp.matmul(
                        jnp.expand_dims(all_eq,1),jnp.expand_dims(-sk_poly,0))
    
    left_key = jax.random.split(key,(all_eq.shape[-1],l))
    left_part_rgsw_encrypted = centered_mod(
                    vmap(GLEV_polynomial,in_axes=(0,None,None,None))(matrix,beta,l,q),q
    )
    left_part_rgsw = vmap(vmap(encrypt,in_axes=(0,None,None,0)),in_axes=(0,None,None,0))(left_part_rgsw_encrypted,sk_poly,dict_params,left_key)
    ##Right part of RGSW
    poly = jnp.zeros((all_eq.shape[-1],sk_poly.shape[-1]))
    poly = poly.at[:,0].set(all_eq)
    right_key = jax.random.split(left_key[0,0],(all_eq.shape[-1],l))
    right_part_rgsw_encrypted = centered_mod(
                    vmap(GLEV_polynomial,in_axes=(0,None,None,None))(poly,beta,l,q),q
    )
    right_part_rgsw = vmap(vmap(encrypt,in_axes=(0,None,None,0)),in_axes=(0,None,None,0))(right_part_rgsw_encrypted,sk_poly,dict_params,right_key)
    return left_part_rgsw,right_part_rgsw







def get_packing_KSK(lwe_sk:Plaintext,rlwe_sk:Plaintext,dict_params:dict,
                    q:int,beta:int,l:int,key):
    degree_lwe = lwe_sk.shape[0]
    degree_RLWE = rlwe_sk.shape[0]
    lwe_sk_polynomial = jnp.zeros((degree_lwe,degree_RLWE))
    lwe_sk_polynomial = lwe_sk_polynomial.at[:,0].set(lwe_sk)
    lwe_sk_polynomial_glev = vmap(GLEV_polynomial,in_axes=(0,None,None,None))(lwe_sk_polynomial,beta,l,q)
    key = jax.random.split(key,(degree_lwe,l))
    KSK = vmap(vmap(encrypt,in_axes=(0,None,None,0)),in_axes=(0,None,None,0))(lwe_sk_polynomial_glev,rlwe_sk,dict_params,key)
    return KSK



def get_galois_key(sk,dict_params,key):
    degree = dict_params["degree"]
    n_steps = jnp.log2(degree).astype(int)
    k = jnp.array([2**(n_steps-i)+1 for i in range(n_steps)])
    sk_auto = vmap(apply_automorphism,(None,0))(sk,k)
    galois_key = vmap(get_keyswitch_sk,(None,0,None,0))(sk,sk_auto,dict_params,key)
    galois_key = jnp.stack(galois_key,axis=2)
    return galois_key






@partial(jax.jit,static_argnames=['q','beta','l','degree','collapse','n_lut'])
def bootstrapping(lwe_ciphertext:Ciphertext,LUT:Ciphertext,BSK:RGSW,
                    q:int,beta:int,l:int,degree:int, collapse:int, all_rot_fft:jnp.array=None, n_lut:int=1):
    """Cette fonction permet de réaliser un boostrapping fonctionnel sur une LWE ciphertext

    Args:
        lwe_ciphertext (Ciphertext): LWE ciphertext 
        LUT (Ciphertext): Look-Up Table. Polynôme chiffré de la fonction de loop-up table
        BSK (RGSW): Clé de bootstrapping
        q (int): modulo d'entré du message chiffré lwe
        degree (int): Degré du polynôme de la LUT
        beta (int): Base de décomposition
        l (int): PUissance, on a la relation beta**l=q si on veut une décomposition parfaite

    Returns:
        lwe_ciphertext(Ciphertext): Message chiffrant l'image par la fonction de la LUT du message lwe d'entré 
    """

    ctMS = lwe_ciphertext
    rotation = blind_rotate(ctMS, LUT, BSK, q, beta, l, degree, collapse, all_rot_fft,n_lut)
    if n_lut ==1:
        return sample_extract(rotation,0)
    else:
        index = jnp.arange(n_lut)
        return vmap(sample_extract,(None,0))(rotation,index)



def get_ksk(sk_poly:jnp.array, sk_lwe:jnp.ndarray,dict_params,key:jnp.ndarray):
    l = dict_params["l"]
    beta = dict_params["beta_ks"]
    q = dict_params["q_ks"]
    poly = jnp.zeros((sk_lwe.shape[-1],sk_poly.shape[-1]))
    poly = poly.at[:,0].set(sk_lwe)
    right_key = jax.random.split(key,(sk_lwe.shape[-1],l))
    right_part_rgsw_encrypted = centered_mod(
                    vmap(GLEV_polynomial,in_axes=(0,None,None,None,None))(poly,beta,l,q),q
    )
    glev = vmap(vmap(encrypt,in_axes=(0,None,None,0)),in_axes=(0,None,None,0))(right_part_rgsw_encrypted,sk_poly,dict_params,right_key)
    return glev



cutfhe_multiply = jax.jit(
        vmap(
            cuTFHE_multiply_seq_monomial,
            in_axes=(None, 0, None, None, None, None, None, None, None),
        ),
        static_argnames=["log_q", "log_beta", "l", "degree", "collapse"],
    )

cutfhe_multiply_merge = jax.jit(vmap(
            cuTFHE_multiply_seq_monomial,
            in_axes=(0, 0, None, None, None, None, None, None, None,None),
        ),
        static_argnames=["log_q", "log_beta", "l", "degree", "collapse","n_lut"],
    )
    
def cuboot(c_lwe_ks, encrypted_LUT, bsk,  q, beta ,l,
        degree, collapse, all_rot_possible_fourier):
    pk, b = c_lwe_ks
    
    rotated_lut = cutfhe_multiply(jnp.stack(encrypted_LUT),pk,bsk, int(jnp.log2(q)), int(jnp.log2(beta)+1e-6),l, degree, collapse,all_rot_possible_fourier )
    #breakpoint()
    b_ms = jnp.round(2*degree*b/q)
    #breakpoint()
    lut_rotated = vmap(rotate_ciphertext,(0,0))(rotated_lut,-b_ms)
    #breakpoint()
    return vmap(sample_extract,(0,None))(lut_rotated,0)

def cuboot_merge(c_lwe_ks, encrypted_LUT, bsk,  q, beta ,l,
        degree, collapse, all_rot_possible_fourier, n_lut):
    pk, b = c_lwe_ks
    #breakpoint()
    rotated_lut = cutfhe_multiply_merge(jnp.stack(encrypted_LUT,axis=1),pk, bsk, int(math.log2(q)), int(math.log2(beta) + 1e-6), l, degree, collapse,all_rot_possible_fourier, n_lut )
    #breakpoint()
    #b_ms = jnp.round(jnp.round(2*degree*b/q)/n_lut)*n_lut
    b_ms = jnp.round(2*degree*b/(q*n_lut))*n_lut
    #breakpoint()
    lut_rotated = vmap(rotate_ciphertext,(0,0))(rotated_lut,-b_ms)
    #breakpoint()
    if n_lut is None:
        return vmap(sample_extract,(0,None))(lut_rotated,0)
    else:
        index = jnp.arange(n_lut)
        return vmap(vmap(sample_extract,(None,0)),(0,None))(lut_rotated,index)

def BR(c_lwe_ks, encrypted_LUT, bsk,  q, beta ,l,
        degree, collapse, all_rot_possible_fourier, n_lut):
    pk, b = c_lwe_ks
    #breakpoint()
    rotated_lut = cutfhe_multiply_merge(jnp.stack(encrypted_LUT,axis=1),pk, bsk, int(math.log2(q)), int(math.log2(beta) + 1e-6), l, degree, collapse,all_rot_possible_fourier, n_lut )
    #breakpoint()
    #b_ms = jnp.round(jnp.round(2*degree*b/q)/n_lut)*n_lut
    b_ms = jnp.round(2*degree*b/(q*n_lut))*n_lut
    #breakpoint()
    lut_rotated = vmap(rotate_ciphertext,(0,0))(rotated_lut,-b_ms)
    return lut_rotated
