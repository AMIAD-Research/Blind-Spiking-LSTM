import jax.numpy as jnp
import jax
from typing import Callable
from jax import vmap
from flax import nnx
import copy
from schemas.format import Ciphertext,RGSW
from schemas.polynomial_jax import jax_fourier
from schemas.text_jax import  packing, multiply_plaintext_ciphertext, packing, pack_lwe_to_rlwe
from schemas.RLWE_jax import bootstrapping, cuboot_merge, get_boostrapping_key_collapse, get_packing_KSK
from schemas.LWE_jax import key_switch_LWE, get_keyswitch_sk_LWE



def prepare_bootstrapping(key,
                          sk:jax.Array, sk_lut:jax.Array, sk_bsk:jax.Array,
                          dict_params:dict, dict_params_lut:dict, dict_params_ks_LWE:dict, 
                          function_lut:Callable, factor:int, collapse:int=None):
    """Cette fonction calcule les différents clés nécessaires au bootstrapping

    Args:
        key (jax.random): Rngs de jax afin de faire des tirage aléatoire pour l'encryption
        sk (jax.Array): Clé privée du message m
        sk_lut (jax.Array): Clé privée de la LUT
        sk_bsk (jax.Array): Clé privée pour le bootstrapping, plus petite que sk afin d'accélérer le bootstrapping
        dict_params (dict): Paramètres de chiffrement avec sk
        dict_params_lut (dict): Paramètres de chiffrement de la LUT
        dict_params_ks_LWE (dict): Paramètres de chiffement des LWE
        function_lut (Callable): Fonction renvoie le polynôme de la LUT
        factor (int): Factor par lequel on multiplie la LUT

    """
    
    t = dict_params["t"]
    q = dict_params["q"]
    degree_lut = dict_params_lut["degree"]
    key_switching_key_bs = get_keyswitch_sk_LWE(sk_bsk,sk,dict_params_ks_LWE,key)
    key = jax.random.split(key)[0]
    LUT = function_lut(t,degree_lut)
    encrypted_LUT = [jnp.zeros(degree_lut),LUT*factor]
    key = jax.random.split(key)[0]

    ##Bootstrapping key
    if collapse is None:
        BSK = get_boostrapping_key(sk_lut,sk_bsk,dict_params_lut,key)
    else:
        BSK = get_boostrapping_key_collapse(sk_lut,sk_bsk,dict_params_lut,key,collapse)
    key= jax.random.split(key)[0]
    bsk_array = jnp.array(BSK)
    bsk_array_nf = bsk_array 
    bsk_array_nf = jnp.permute_dims(bsk_array_nf,[2,0,1,3,4])
    bsk_array_fourier = jax_fourier(bsk_array)
    bsk_ordered = jnp.permute_dims(bsk_array_fourier,[1,0,3,2,4])

    ###Les paramètres de key switching sont définis par le dictionnaire de la lut
    dict_params_copy = copy.deepcopy(dict_params)
    dict_params_copy["beta_ks"] = dict_params_lut["beta_ks"]
    dict_params_copy["l_ks"] = dict_params_lut["l_ks"]
    ##Key switch for packing
    ksk_packing = get_packing_KSK(sk,sk, dict_params_copy, q, dict_params_copy["beta_ks"], dict_params_copy["l_ks"], key)
    key = jax.random.split(key)[0]

    ##Key switch from sk_lut to sk1
    
    key_switching_key_back = get_keyswitch_sk_LWE(sk,sk_lut,dict_params_copy,key)

    return key_switching_key_back, ksk_packing, bsk_ordered, key_switching_key_bs, encrypted_LUT


class HeavysideBoostrapper:
    """Cette classe permet de réaliser un bootstrapping, càd une fonction d'activation
    """
    def __init__(self,
                bootstrapping_key:RGSW,
                key_switching_key_bs:RGSW,
                ksk_packing:Ciphertext,
                galois_key:Ciphertext,
                dict_params:dict,
                dict_params_lut:dict,
                dict_params_ks_LWE:dict,
                dict_params_packing:dict,
                collapse:int,
                all_rot_possible_fourier:jnp.array,
                n_lut:int
                ):
        

        self.bootstrapping_key = bootstrapping_key
        self.key_switching_key_bs = key_switching_key_bs
        self.ksk_packing = ksk_packing
        self.galois_key = galois_key
        self.dict_params = dict_params
        self.dict_params_lut = dict_params_lut
        self.dict_params_ks_LWE = dict_params_ks_LWE
        self.dict_params_packing = dict_params_packing
        self.collapse = collapse
        self.all_rot_possible_fourier = all_rot_possible_fourier
        self.log2q = int(jnp.log2(self.dict_params_lut["q"]))
        self.log2beta = int(jnp.log2(self.dict_params_lut["beta_bs"])+1e-6)
        self.n_lut = n_lut
    
    
    def prepare_LUT(self, c_lwe, input_lut):
        c_rlwe = pack_lwe_to_rlwe(c_lwe, self.galois_key, self.dict_params_packing["beta_ks"],  self.dict_params_packing["l_ks"], self.dict_params_packing["q"],self.dict_params_packing["degree"])
        prod = multiply_plaintext_ciphertext(input_lut, c_rlwe, self.dict_params_packing["degree"])
        return prod
    

    def packing(self,x):
        return packing(x,
                    self.ksk_packing,
                    self.dict_params["q"],
                    self.dict_params["beta_ks"],
                    self.dict_params["l_ks"],
                    self.dict_params["degree"])



    def __call__(self, h:Ciphertext, encrypted_LUT:Ciphertext):
        """Cette fonction permet d'appliquer la fonction du boostrapping. En règle générale, c'est une fonction Relu, mais peut aussi-être une sigmoid ou autre.

        Args:
            h (Ciphertext): Hidden state, correspondant à une shape [( hidden_dim,degree), (hidden_dim,1)]
        """


  
        ###On change la taille de la clé secrète du chiffré pour diminuer le nombre d'itération dans la for loop du bootstrapping
        c_lwe_ks = key_switch_LWE(self.key_switching_key_bs, h, self.dict_params_ks_LWE)



        ###On réalise le bootstapping fonctionnel
        y1_boot = bootstrapping(c_lwe_ks,
                                                                                    encrypted_LUT,
                                                                                    self.bootstrapping_key,
                                                                                    self.dict_params_lut["q"],
                                                                                    self.dict_params_lut["beta_bs"],
                                                                                    self.dict_params_lut["l_bs"],
                                                                                    self.dict_params_lut["degree"],
                                                                                    self.collapse,
                                                                                    self.all_rot_possible_fourier,
                                                                                    )
        
        return y1_boot
    
    def cuboot(self,h,encrypted_LUT):
        c_lwe_ks = vmap(key_switch_LWE,(None,0,None))(self.key_switching_key_bs, h, self.dict_params_ks_LWE)
        y1_boot = cuboot_merge(c_lwe_ks, encrypted_LUT, self.bootstrapping_key,
                                                            self.dict_params_lut["q"],
                                                            self.dict_params_lut["beta_bs"],
                                                            self.dict_params_lut["l_bs"],
                                                            self.dict_params_lut["degree"],
                                                            self.collapse,
                                                            self.all_rot_possible_fourier,
                                                            self.n_lut)
        return y1_boot



class ReluBoostrapper:
    """Cette classe permet de réaliser un bootstrapping, càd une fonction d'activation
    """
    def __init__(self,
                bootstrapping_key:RGSW,
                key_switching_key_bs:RGSW,
                ksk_packing:Ciphertext,
                dict_params:dict,
                dict_params_lut:dict,
                dict_params_ks_LWE:dict,
                dict_params_packing:dict,
                collapse:int,
                all_rot_possible_fourier:jnp.array,
                beta_x:float,
                s_head:jnp.array
                ):
        

        self.bootstrapping_key = bootstrapping_key
        self.key_switching_key_bs = key_switching_key_bs
        self.ksk_packing = ksk_packing
        self.dict_params = dict_params
        self.dict_params_lut = dict_params_lut
        self.dict_params_ks_LWE = dict_params_ks_LWE
        self.dict_params_packing = dict_params_packing
        self.collapse = collapse
        self.all_rot_possible_fourier = all_rot_possible_fourier
        self.beta_x = beta_x
        self.hidden_dim = s_head.shape[0]
        self.LUT = [
            jnp.zeros((self.hidden_dim, self.dict_params_lut["degree"])),
            jnp.stack([
                jnp.concat(
                    [jnp.arange(0,s_head[i],s_head[i]*2/dict_params_lut["degree"]) * beta_x, jnp.zeros(dict_params_lut["degree"]//2)])
                for i in range(self.hidden_dim)])
        ]

    def packing(self,x):
        return packing(x,
                    self.ksk_packing,
                    self.dict_params["q"],
                    self.dict_params["beta_ks"],
                    self.dict_params["l_ks"],
                    self.dict_params["degree"])


    def __call__(self, h:Ciphertext):
        """Cette fonction permet d'appliquer la fonction du boostrapping. En règle générale, c'est une fonction Relu, mais peut aussi-être une sigmoid ou autre.

        Args:
            h (Ciphertext): Hidden state, correspondant à une shape [( hidden_dim,degree), (hidden_dim,1)]
        """


  
        ###On change la taille de la clé secrète du chiffré pour diminuer le nombre d'itération dans la for loop du bootstrapping
        c_lwe_ks = vmap(key_switch_LWE,(None,0,None))(self.key_switching_key_bs, h, self.dict_params_ks_LWE)



        ###On réalise le bootstapping fonctionnel
        y1_boot = vmap(bootstrapping,(0,0, None,None,None,None,None,None,None))(c_lwe_ks,
                                                                                self.LUT,
                                                                                self.bootstrapping_key,
                                                                                self.dict_params_lut["q"],
                                                                                self.dict_params_lut["beta_bs"],
                                                                                self.dict_params_lut["l_bs"],
                                                                                self.dict_params_lut["degree"],
                                                                                self.collapse,
                                                                                self.all_rot_possible_fourier,
                                                                                )
        
        return y1_boot