import numpy as np
from jax import vmap
import jax
import jax.numpy as jnp
from parameters_crypto import *
from schemas.RLWE_jax import  get_packing_KSK, get_galois_key
from nn_tfhe.Spike_LSTM import  Linear
from nn_tfhe.activation import prepare_bootstrapping
from nn_tfhe.activation import HeavysideBoostrapper, ReluBoostrapper
from nn_tfhe.utils_spike import get_plain_LUT



def identity(x,degree):
        return jnp.ones(degree)

def sample_crypto_keys(seed):
        key = jax.random.PRNGKey(seed)
        sk = np.random.choice([0.,1],degree)
        sk_bsk = np.random.choice([0.,1.],(n_keyswitch_bootstrapping))
        sk_lut = sk

        
        # ###Génération des clés de packing
        ksk_packing= get_packing_KSK(sk_lut, sk_lut, dict_params,
                                        q, dict_params["beta_ks"],dict_params["l_ks"], key )

        key = jax.random.split(key,int(jnp.log2(degree)))
        galois_key = get_galois_key(sk, dict_params_packing, key)


        key = jax.random.split(key[0])[0]
        _, _, bsk, key_switching_key_bs, _ = prepare_bootstrapping(key, sk, sk_lut, sk_bsk,
                                                        dict_params, dict_params_lut, dict_params_ks_LWE,
                                                        identity, beta_x, collapse)

        return sk, ksk_packing, galois_key, bsk, key_switching_key_bs

def get_linears(plain_model,sf,si,so,s_head1,s_out):
        hidden_dim = plain_model.hidden_dim

        w_f = plain_model.lstm.linearf.get_norm_kernel()
        b_f = plain_model.lstm.linearf.linear.bias
        W_f = Linear(hidden_dim,dict_params,beta_w/sf)
        W_f.set_weights(w_f,b_f)

        w_i = plain_model.lstm.lineari.get_norm_kernel()
        b_i = plain_model.lstm.lineari.linear.bias
        W_i = Linear(hidden_dim,dict_params,beta_w/si)
        W_i.set_weights(w_i,b_i)

        w_o = plain_model.lstm.linearo.get_norm_kernel()
        b_o = plain_model.lstm.linearo.linear.bias
        W_o = Linear(hidden_dim,dict_params,beta_w/so)
        W_o.set_weights(w_o,b_o)

        w_head1 = plain_model.head.layers[0].get_norm_kernel()
        b_head1 = plain_model.head.layers[0].linear.bias
        W_head1 = Linear(hidden_dim,dict_params,beta_w/s_head1)
        W_head1.set_weights(w_head1,b_head1)

        w_out = plain_model.head.layers[2].get_norm_kernel()
        b_out = plain_model.head.layers[2].linear.bias
        W_out = Linear(hidden_dim,dict_params,beta_w/s_out)
        W_out.set_weights(w_out,b_out)

        return W_f, W_i, W_o, W_head1, W_out

def get_lut(plain_model,sf,si,so):
        n_lut = plain_model.lstm.n_lut
        hidden_dim = plain_model.hidden_dim
        gammaf = plain_model.lstm.gammaf
        gammai = plain_model.lstm.gammai
        gammao = plain_model.lstm.gammao
        eta = plain_model.lstm.eta
        indexes = jnp.arange(n_lut)
        lut_f = vmap(vmap(get_plain_LUT,(None,None,None,0,0)),(None,None,0,None,0))(degree_lut, n_lut, indexes,sf,gammaf)
        lut_i = vmap(vmap(get_plain_LUT,(None,None,None,0,0)),(None,None,0,None,0))(degree_lut, n_lut, indexes,si,gammai)
        lut_i = jnp.round(jnp.sum(lut_i*eta[...,None],axis=0)*beta_x)
        lut_i = [jnp.zeros((hidden_dim//n_lut,degree_lut)), lut_i]
        lut_o = vmap(vmap(get_plain_LUT,(None,None,None,0,0)),(None,None,0,None,0))(degree_lut, n_lut, indexes,so,gammao)
        return lut_f, lut_i, lut_o



def get_tfhe_activation(bsk,key_switching_key_bs,ksk_packing,galois_key,s_head1,n_lut):
        heavyBS = HeavysideBoostrapper(
                                    bootstrapping_key = bsk,
                                    key_switching_key_bs = key_switching_key_bs,
                                    ksk_packing = ksk_packing,
                                    galois_key = galois_key,
                                    dict_params = dict_params,
                                    dict_params_lut = dict_params_lut,
                                    dict_params_ks_LWE = dict_params_ks_LWE,
                                    dict_params_packing = dict_params_packing,
                                    collapse = collapse,
                                    all_rot_possible_fourier = all_rot_possible_fourier,
                                    n_lut = n_lut
        )

        reluBS = ReluBoostrapper(
                                        bootstrapping_key = bsk,
                                        key_switching_key_bs = key_switching_key_bs,
                                        ksk_packing = ksk_packing,
                                        dict_params = dict_params,
                                        dict_params_lut = dict_params_lut,
                                        dict_params_ks_LWE = dict_params_ks_LWE,
                                        dict_params_packing = dict_params_packing,
                                        collapse = collapse,
                                        all_rot_possible_fourier = all_rot_possible_fourier,
                                        beta_x = beta_x,
                                        s_head = s_head1
        )
        return heavyBS, reluBS

