import jax
from jax import vmap
import jax.numpy as jnp
from typing import Tuple
from schemas.format import Ciphertext,Plaintext,RGSW
from schemas.polynomial_jax import fft_polynomial_multiply, decomposition, GLEV_polynomial, centered_mod, coef_rotation,fft_product_twice, partial_fft_product,jax_inversefourier, jax_fourier, apply_automorphism
from functools import partial
import math
import numpy as np
jax.config.update('jax_enable_x64', True)

@jax.jit
def sum_plaintext_to_ciphertext(plaintext:Plaintext,ciphertext:Ciphertext):
    """Cette fonction permet d'additionner un message clair par un message chiffré

    Args:
        plaintext (Plaintext): Message clair
        ciphertext (Ciphertext): Message chiffré

    Returns:
        Ciphertext: Message chiffré
    """
    return ciphertext[0], ciphertext[1]+plaintext



@partial(jax.jit, static_argnames=['q','modulo'])
def sum_ciphertext_ciphertext(ciphertext1:Ciphertext, ciphertext2:Ciphertext,q:int,modulo=True)->Ciphertext:
    """Cette fonction calcule la somme de deux ciphertexts.

    Args:
        ciphertext1 (Ciphertext): Ciphertext1
        ciphertext2 (Ciphertext): Ciphertext2
        q (int): Modulo de l'espace des chiffrés

    Returns:
        Ciphertext: Ciphertext1+ciphertext2
    """
    if modulo:
        return centered_mod(ciphertext1[0]+ciphertext2[0],q),centered_mod(ciphertext1[1]+ciphertext2[1],q)
    else:
        return ciphertext1[0]+ciphertext2[0], ciphertext1[1]+ciphertext2[1]


@partial(jax.jit, static_argnames=['q'])
def diff_ciphertext_ciphertext(ciphertext1:Ciphertext, ciphertext2:Ciphertext,q:int)->Ciphertext:
    """Cette fonction calcule la différence entre 2 ciphertexts.

    Args:
        ciphertext1 (Ciphertext): Ciphertext1
        ciphertext2 (Ciphertext): Ciphertext2
        q (int): Modulo de l'espace des chiffrés

    Returns:
        Ciphertext: ciphertext1 - ciphertext2
    """
    return centered_mod(ciphertext1[0]-ciphertext2[0],q),centered_mod(ciphertext1[1]-ciphertext2[1],q)

@partial(jax.jit,static_argnames=["q"])
def centered_mod_ciphertext(ciphertext:Ciphertext,q:int):
    """Cette fonction renvoie le modulo centré de q d'un message chiffré

    Args:
        ciphertext (Ciphertext): Message chiffré
        q (int): modulus

    Returns:
        _type_: _description_
    """
    return centered_mod(ciphertext[0],q),centered_mod(ciphertext[1],q)


@partial(jax.jit, static_argnames=['stay_fft'])
def multiply_plaintext_plaintext(plaintext1:Plaintext, plaintext2:Plaintext,  stay_fft=False)->Plaintext:
    """Multiplication entre deux messages clairs.

    Args:
        plaintext1 (Plaintext): _description_
        plaintext2 (Plaintext): _description_
        degree (int): _description_

    Returns:
        Plaintext: _description_
    """
    return fft_polynomial_multiply(plaintext1,plaintext2,stay_fft)

@partial(jax.jit, static_argnames=['degree','stay_fft','ciphertext_fft'])
def multiply_plaintext_ciphertext(plaintext:Plaintext,ciphertext:Ciphertext,degree:int, stay_fft:bool=False, ciphertext_fft:bool=False)->Ciphertext:
    """Multiplication d'un message ciphertext par un message plaintext.

    Args:
        plaintext (Plaintext): Message clair
        ciphertext (Ciphertext): Message chiffré
        degree (int): Degré polynôme

    Returns:
        Ciphertext: Message chiffré
    """
    public_key, b = ciphertext[0],ciphertext[1]
    if degree == 1:
        new_public_key = plaintext * public_key
        new_b = plaintext * b
    else:
        if ciphertext_fft:
            new_public_key, new_b = partial_fft_product(plaintext, public_key, b, stay_fft)
        else:
            new_public_key, new_b = fft_product_twice(plaintext, public_key, b, stay_fft)
    return new_public_key,new_b


#@partial(jax.jit,static_argnames=['q','beta','l','degree','rgsw_fft'])
def multiply_ciphertext_RGSW(ciphertext:Ciphertext, RGSW:RGSW,
                                q:int,beta:int,l:int,degree:int, rgsw_fft:bool=False)->Ciphertext:
    """Cette fonction multiplie un ciphertext avec un RGSW et renvoie un chiffré RLWE.

    Args:
        ciphertext (Ciphertext): Chiffré en RLWE, RLWE(m1)
        RGSW (RGSW): Chiffré en RGSW, RGSW(m2)
        q (int): Modulo de l'espace des chiffrés
        beta (int): Base de décomposition
        l (int): Puissance maximale de décomposition
        degree (int): Degré des polynômes

    Returns:
        Ciphertext: Chiffré RLWE(m1*m2)
    """
    cipher_stacked = jnp.stack(ciphertext)
    dec = vmap(decomposition,(0,None,None,None))(cipher_stacked, beta, l, q)

    # rgsw_1 = jnp.stack([RGSW[0][0],RGSW[0][1]])
    # rgsw_2 = jnp.stack([RGSW[1][0],RGSW[1][1]])
    # rgsw_stacked = jnp.stack([rgsw_1,rgsw_2])
    # rgsw_stacked_perm2 = jnp.permute_dims(rgsw_stacked,(0,2,1,3))
    # prod = vmap(gadget_product,(0,0,None,None,None))(rgsw_stacked_perm2, dec, q, degree, rgsw_fft)
    # public_key_times_GLEV_minus_square = prod[0][0],prod[1][0]
    # b_times_GLEV = prod[0][1],prod[1][1]
    
    decomp_public_key, decomp_b = dec
    encrypted_s_time_m_glev, encrypted_glev = RGSW
    public_key_times_GLEV_minus_square = gadget_product(encrypted_s_time_m_glev,decomp_public_key,degree,rgsw_fft)
    b_times_GLEV = gadget_product(encrypted_glev,decomp_b,degree,rgsw_fft)

    sum_ciphers = sum_ciphertext_ciphertext(b_times_GLEV,public_key_times_GLEV_minus_square,q)
    return sum_ciphers



@partial(jax.jit,static_argnames=['q','beta','l','degree'])
def multiply_ciphertext_ciphertext(ciphertext1:Ciphertext, ciphertext2:Ciphertext,glev_s_quare:Ciphertext,
                                   degree:int,q:int,beta:int,l:int,delta:int)->Ciphertext:
    """Cette fonction permet de multiplier deux chiffrés entre eux.

    Args:
        ciphertext1 (Ciphertext): Ciphertext1, RLWE(delta*m1)
        ciphertext2 (Ciphertext): Ciphertext2, RLWE(delta*m2)
        RGSW_sk (RGSW): Clé secrète chiffré en RGSW
        degree (int): Degré des polynômes
        q (int): Modulo de l'espace des chiffrés
        beta (int): Base de décomposition
        l (int): Puissance maximale de décomposition

    Returns:
        Ciphertext: RLWE(delta*m1*m2)
    """
    public_key_1,b_1 = ciphertext1[0], ciphertext1[1]
    public_key_2,b_2 = ciphertext2[0], ciphertext2[1]
    b_1_b_2 = jnp.round(fft_polynomial_multiply(b_1, b_2)/delta)
    a_1_b_2 = jnp.round(fft_polynomial_multiply(public_key_1, b_2)/delta)
    b_1_a_2 = jnp.round(fft_polynomial_multiply(b_1, public_key_2)/delta)

    a_1_a_2 = jnp.round(fft_polynomial_multiply(public_key_1, public_key_2)/delta)
    decomp_a_1_a_2 = decomposition(a_1_a_2, beta, l,q)
    second_term = gadget_product(glev_s_quare,decomp_a_1_a_2,degree)
    new_public_key = a_1_b_2 + b_1_a_2 + second_term[0]
    new_b = b_1_b_2 + second_term[1]
    return centered_mod_ciphertext((new_public_key, new_b),q)



@partial(jax.jit, static_argnames=['degree','glev_fft'])
def gadget_product(glev:Ciphertext, decomp:Plaintext,  degree:int, glev_fft:bool=False)->Ciphertext:
    """Cette fonction fonction applique le produit lié à la gadget décomposition.

    Args:
        glev (Ciphertext): Chiffré en GLEV
        decomp (Plaintext): Plaintext décomposé suivant une base beta
        q (int): Modulo de l'espace des chiffrés
        degree (int): Degré des polynômes de glev

    Returns:
        Ciphertext: Résultat du produit (proche d'un produit scalaire)
    """
    product = jax.vmap(multiply_plaintext_ciphertext,in_axes=(0,0,None,None,None))(decomp, glev, degree, True, glev_fft)
    product = jnp.stack(product)
    product = jnp.sum(product,axis=1)
    public_key_sum, b_sum = vmap(jax_inversefourier,0)(product)
    return public_key_sum,b_sum



@partial(jax.jit, static_argnames=['beta','l','q'])
def get_GLEV_ciphertext(c:Ciphertext,beta:int,l:int, q:int)->Ciphertext:
    """Cette fonction prend un chiffré et renvoie son GLEV associé.

    Args:
        c (Ciphertext): Chiffré, par défaut en RLWE
        beta (int): Base de décomposition
        l (int): Puissance maximale de décomposition
        q (int):Modulo du chiffré
        degree (int): Degré du chiffré

    Returns:
        Ciphertext: GLEV associé au chiffré c
    """
    public_key = c[0]
    b = c[1]
    glev_public_key = GLEV_polynomial(public_key,beta,l,q)
    glev_b = GLEV_polynomial(b,beta,l,q)
    return glev_public_key,glev_b

@partial(jax.jit, static_argnames=['current_modulus','new_modulus'])
def modulus_switch(c:Ciphertext,current_modulus:int,new_modulus:int):
    """Cette fonction permet de passer d'un modulo à un autre.

    Args:
        ciphertext (Ciphertext): Chiffré sur lequel nous souhaitons appliquer le changement de modulo
        current_modulus (int): Le modulo du chiffré en entré
        new_modulus (int): Le modulo que l'on souhaite avoir pour le chiffré de sortie

    Returns:
        Ciphertext: Chiffré avec le nouveau modulo
    """
    pk, b = c
    new_pk = jnp.round(pk*new_modulus/current_modulus)
    new_b = jnp.round(b*new_modulus/current_modulus)
    return new_pk, new_b



jax.jit
def sample_extract(ciphertext:Ciphertext,alpha):
    """Cette fonction permet d'extraire un coefficient d'un RLWE et renvoie le LWE associé

    Args:
        ciphertext (Ciphertext): Polynôme P chiffré en RLWE
        alpha (int): Index du coefficient que l'on souhaite extraire du polynôme

    Returns:
        Ciphertext: LWE correspondant au chiffré de P[index]
    """
    public_key,b = ciphertext[0],ciphertext[1]
    degree = public_key.shape[-1]
    j = jnp.arange(degree)
    src = jnp.where(j <= alpha, alpha - j, alpha - j + degree)
    sign = jnp.where(j <= alpha, 1, -1)
    extracted = jnp.take_along_axis(public_key,src,axis=-1)*sign
    return extracted,jnp.expand_dims(b[alpha],axis=-1)


@partial(jax.jit,static_argnames=['q','beta','l','degree'])
def packing(ciphertexts:Ciphertext,KSP:Ciphertext,
            q:int, beta:int, l:int, degree:int):
    """Cette fonction permet de packer un ensemble de LWE dans un RLWE

    Args:
        ciphertexts (Ciphertext): L'ensemble des ciphertextes
        KSP (Ciphertext): Clé de key_switch  pour le packing 
        q (int): modulus de l'espace des chiffrés
        beta (int): base de décomposition de entiers
        l (int): exposant maximal pour notre décomposition
        degree (int): Degré du polynôme souhaité à l'issue du packing

    Returns:
        Ciphertext: Ciphertext en RLWE. Ses coefficients sont de gauche à droite ceux de la liste des LWE
    """
    public_keys,bs = ciphertexts[0],ciphertexts[1]
    n_lwes = bs.shape[0]
    packed_b = jnp.zeros((degree,))
    packed_b = packed_b.at[:n_lwes].set(jnp.squeeze(bs,axis=-1))
    public_key_polynomial = jnp.transpose(public_keys)
    if n_lwes < degree:##Si le nombre de LWE n'est pas égal au degré du polynôme, on complète par des zeros pour avoir des polynômes de la bonne taille
        completion_polynomial = jnp.zeros((degree,degree-n_lwes))
        public_key_polynomial = jnp.concat([public_key_polynomial,completion_polynomial], axis=-1)
    public_key_polynomial_decomp = vmap(decomposition,in_axes=(0,None,None,None))(public_key_polynomial, beta, l,q)
    second_term = vmap(gadget_product,in_axes = (0,0,None)) (KSP, public_key_polynomial_decomp, degree)
    result_public_key = - jnp.sum(second_term[0],axis=0)
    result_b = packed_b - jnp.sum(second_term[1],axis=0)
    return centered_mod(result_public_key,q),centered_mod(result_b,q)

@partial(jax.jit, static_argnames=['q', 'beta', 'l', 'degree'])
def packing_optimized(ciphertexts: tuple, KSP: jnp.ndarray, 
                      q: int, beta: int, l: int, degree: int):
    """
    Version optimisée en mémoire du packing LWE -> RLWE.
    """
    public_keys, bs = ciphertexts[0], ciphertexts[1]
    n_lwes = bs.shape[0]
    n_lwe_dim = public_keys.shape[1]

    # 1. Utilisation de jnp.pad au lieu de .at[].set() ou jnp.zeros()
    bs_squeezed = jnp.squeeze(bs, axis=-1)
    packed_b = jnp.pad(bs_squeezed, (0, degree - n_lwes))

    # 2. Transposition et padding idiomatique
    public_key_polynomial = jnp.transpose(public_keys)
    if n_lwes < degree:
        # Pad uniquement la dimension 'degree' (axe 1)
        public_key_polynomial = jnp.pad(public_key_polynomial, ((0, 0), (0, degree - n_lwes)))

    # 3. Remplacement de vmap par jax.lax.scan pour accumuler à la volée
    # Cela évite de matérialiser le tenseur (n_lwe_dim, degree, l) en mémoire
    def scan_body(carry, elements):
        acc_a, acc_b = carry
        ksp_i, pk_i = elements
        
        # Décomposition à la volée pour un seul LWE (taille: degree)
        pk_i_decomp = decomposition(pk_i, beta, l, q)
        
        # Produit de gadget pour ce composant
        res_a, res_b = gadget_product(ksp_i, pk_i_decomp, degree)
        
        # Accumulation
        return (acc_a + res_a, acc_b + res_b), None

    # Initialisation des accumulateurs (polynômes RLWE de taille 'degree')
    init_carry = (
        jnp.zeros(degree, dtype=public_keys.dtype),
        jnp.zeros(degree, dtype=public_keys.dtype)
    )

    # Exécution du scan le long de la dimension LWE
    final_carry, _ = jax.lax.scan(scan_body, init_carry, (KSP, public_key_polynomial))

    # 4. Finalisation
    result_public_key = -final_carry[0]
    result_b = packed_b - final_carry[1]

    return centered_mod(result_public_key, q), centered_mod(result_b, q)


@partial(jax.jit,static_argnames=['q','beta','l','degree'])
def lwe_to_RLWE(lwe:Ciphertext,KSP:Ciphertext,
            q:int, beta:int, l:int, degree:int):
    """This function packs a single LWE into a RLWE, putting the LWE coefficient on the constant value of the polynomial. This function has the advantages of not using too much memory compared to the packing one.

    Args:
        lwe (Ciphertext): _description_
        KSP (Ciphertext): _description_
        q (int): _description_
        beta (int): _description_
        l (int): _description_
        degree (int): _description_

    Returns:
        _type_: _description_
    """
    public_key,b = lwe
    first_term = jnp.zeros(degree)
    first_term = first_term.at[0].set(b[0])
    public_keys = jnp.zeros((degree,degree))
    public_keys = public_keys.at[:,0].set(public_key)
    dec = decomposition(public_key, beta, l, q).T
    second_term = vmap(vmap(multiply_plaintext_ciphertext,(0,0,None,None)),(0,0,None,None))(dec,KSP,1,False)
    result_public_key = - jnp.sum(second_term[0],axis=(0,1))
    result_b = first_term - jnp.sum(second_term[1],axis=(0,1))
    return centered_mod(result_public_key,q),centered_mod(result_b,q)

@partial(jax.jit, static_argnames=['degree'])
def trivial_embedding(lwe:Ciphertext, degree: int) -> Ciphertext:
    """
    Étape 1 : Mapping gratuit du LWE vers le RLWE.
    Exploite l'arithmétique modulo (X^N + 1).
    """
    a,b = lwe
    # B(X) = b sur le coefficient 0
    B = jnp.zeros(degree)
    B = B.at[0].set(b.squeeze())
    
    # A(X) = a_0 - \sum_{i=1}^{N-1} a_{N-i} X^i
    # Utilisation de jnp.flip pour optimiser le reverseing des indices en XLA
    A = jnp.zeros(degree, dtype=a.dtype)
    A = A.at[0].set(a[0])
    A = A.at[1:].set(-jnp.flip(a[1:]))
    #breakpoint()
    return A, B

@partial(jax.jit, static_argnames=['k','beta','l','q','degree'])
def apply_automorphism_and_keyswitch(rlwe:Ciphertext, KSK: jax.Array, k: int,
                                    beta:int, l:int, q:int, degree:int ) -> Ciphertext:
    """
    Applique l'automorphisme X -> X^k et effectue le Key Switch.
    Dans une implémentation complète, ceci se ferait idéalement dans le domaine NTT.
    """
    rlwe = jnp.stack(rlwe)
    #A,B = rlwe

    
    # Note : Il faut gérer les changements de signes liés au X^N = -1
    # Pour simplifier le squelette, on illustre la permutation pure.
    rlwe_auto = vmap(apply_automorphism,(0,None))(rlwe,k)
    
    A_perm, B_perm = rlwe_auto[0], rlwe_auto[1]
    
    # --- Décomposition en base (Gadget Decomposition) ---
    # C'est ici que le 'Hoisting' entre en jeu : on décompose A_perm une seule fois.
    A_dec = decomposition(A_perm,beta,l,q) 
    
    # --- Produit tensoriel avec la clé d'évaluation (GEMM optimisé CUDA/TPU) ---
    A_ks, B_ks = gadget_product(KSK, A_dec, degree)
    
    return sum_ciphertext_ciphertext([jnp.zeros_like(B_perm),B_perm],[A_ks,B_ks],q)

@partial(jax.jit, static_argnames=['beta','l','q','degree'])
def pack_lwe_to_rlwe(lwe:Ciphertext, galois_keys: list[Ciphertext],
                    beta:int, l:int, q:int, degree:int) -> Ciphertext:
    """
    Étape 2 : L'arbre de Trace en O(log N).
    """
    # 1. Embedding initial
    lwe = modulus_switch(lwe,q,q/degree)
    rlwe = trivial_embedding(lwe, degree)
    
    # 2. Arbre d'automorphismes
    # XLA va dérouler cette boucle car N est statique, générant un graphe CUDA linéaire et très rapide.
    num_steps = int(math.log2(degree))

    for i in range(num_steps):
        # L'exposant de l'automorphisme pour la trace à l'étape i
        k = 2**(num_steps - i) + 1
        # Application de l'automorphisme sur le chiffré courant
        #rlwe = modulus_switch(rlwe, q, q/2)
        rlwe_auto = apply_automorphism_and_keyswitch(rlwe, galois_keys[i], k, beta, l, q, degree)
        
        
        # Addition homomorphique (Trace)
        A,B = sum_ciphertext_ciphertext(rlwe,rlwe_auto,q,modulo=False)
        #A = jnp.round(A/2)
        #B = jnp.round(B/2)
        rlwe = [A,B]
        
        #rlwe = centered_mod_ciphertext(rlwe,q)
        #breakpoint()
        
    return A, B


@jax.jit
def rotate_ciphertext(ciphertext:Ciphertext,exponent:jnp.ndarray):
    """Cette fonction applique une rotation au chiffré.

    Args:
        ciphertext (Ciphertext): Chiffré en RLWE auquel on applique un déphasage
        exponent (jnp.ndarray): Phase à appliquer au chiffré

    Returns:
        Ciphertext: Chiffré en RLWE déphasé de exponent
    """
    public_key, b = ciphertext[0],ciphertext[1]
    rotated_public_key = coef_rotation(public_key, exponent) ##On déphase la clé publique
    rotated_b = coef_rotation(b,exponent) ##On déphase le polynôme B
    return rotated_public_key, rotated_b


@partial(jax.jit,static_argnames=['q','beta','l','degree','collapse','n_lut'])
def blind_rotate(ciphertext_lwe:Ciphertext, LUT:Ciphertext, BSK:Tuple[RGSW],
                 q:int,beta:int,l:int,degree:int,collapse:int, all_rot_fft:jnp.array, n_lut:int):
    """Cette fonction permet de réaliser une rotation à l'aveugle, c'est à dire de changer la phase du polynôme sans connaître le dephasage que l'on souhaite appliqué.

    Args:
        ciphertext_lwe (Ciphertext): Message chiffré, correspondant à la phase chiffré en LWE
        LUT (Ciphertext): Look-Up Table chiffrée en RLWE
        BSK (RGSW): Clé de bootstrapping
        q (int): Modulo dans l'espace des chiffré
        beta (int): Base de décomposition
        l (int): Puissance maximale pour la décompositon
        degree (int): Degré de la Look-Up Table

    Returns:
        RLWE: Polynôme dephasé par me chiffré LWE 
    """
    public_key_lwe, b_lwe = ciphertext_lwe[0],ciphertext_lwe[1]
    b_lwe = jnp.round(2*degree*b_lwe/(q*n_lut))*n_lut
    ct_out = rotate_ciphertext(LUT,-b_lwe)
    ct_out = multiply_seq_monomial(ct_out,public_key_lwe,BSK,q,beta,l,degree,collapse, all_rot_fft,n_lut) ## On dephase de sum(a_i*s_i) sans connaître les s_i

    return ct_out
    

@partial(jax.jit,static_argnames=['q','beta','l','degree','collapse','n_lut'])
def multiply_seq_monomial(ciphertext:Ciphertext,public_key:jnp.array,BSK:Tuple[RGSW],
                          q:int,beta:int,l:int,degree:int,collapse:int, all_rot_fft:jnp.array,n_lut)->Ciphertext:
    """Cette fonction permet de réaliser la multiplication séquentielle dans l'espace chiffré de M par X^{sum(a_i*s_i)}.

    Args:
        ciphertext (Ciphertext): Ciphertext en RLWE
        public_key (jnp.array): Clé publique correspondant aux a_i
        BSK (Tuple[RGSW]): Clé de boostrapping correspondant aux messages chiffrés RGSW(s_i)
        q (int): Modulo de l'espace des chiffrés
        beta (int): Base de décomposition
        l (int): PUissance maximale de décomposition
        degree (int): Degré des polyômes des chiffrés

    Returns:
        Ciphertext: Message chiffré déphasé de sum(a_i*s_i)
    """
    n = public_key.shape[-1]

    n_iter = n//collapse
    ct_init = jnp.stack(ciphertext) if isinstance(ciphertext, tuple) else ciphertext
    kronecker_matrix = all_binary_vectors(collapse)
    pk_reshaped = public_key.reshape(n_iter, collapse)
    all_rotations_val = jnp.dot(pk_reshaped, kronecker_matrix.T)
    all_rotations_indices = (jnp.round(2 * degree * all_rotations_val / (q*n_lut))*n_lut).astype(jnp.int32)%(2*degree)
    bsk_array = BSK
    shape = list(bsk_array.shape)
    shape[3] = n_iter
    shape.insert(4, 2**collapse) # On split l'axe 3 en (n_iter, chunk_size)
    bsk_reshaped = bsk_array.reshape(shape)
    bsk_reshaped = jnp.permute_dims(bsk_reshaped,(3,0,2,1,4,5))

    def step(ct_in, scan_input):
    #     # carry is not used; set to None or any placeholder
        bsk_chunk, rot_indices = scan_input
        dec = decomposition(ct_in, beta, l, q)
        dec_fft = jax_fourier(dec)
        sum_rgsw = jnp.einsum('abcmd, md -> abcd', bsk_chunk, all_rot_fft[rot_indices])

        # # On applique le BSK
        prod = jnp.einsum('abcd, bcd -> ad', sum_rgsw, dec_fft)
        ct_next = centered_mod(jax_inversefourier(prod),q)
        return ct_next, None
    
    ct_final, _ = jax.lax.scan(step,ct_init,(bsk_reshaped, all_rotations_indices),n_iter)
    return ct_final
    


    
@partial(jax.jit,static_argnames=['q','beta','l','degree','collapse','n_lut'])
def many_LUT_multiply_seq_monomial(ciphertext:Ciphertext,public_key:jnp.array,BSK:Tuple[RGSW],
                          q:int,beta:int,l:int,degree:int,collapse:int, all_rot_fft:jnp.array, n_lut:int)->Ciphertext:
    """Cette fonction permet de réaliser la multiplication séquentielle dans l'espace chiffré de M par X^{sum(a_i*s_i)}.

    Args:
        ciphertext (Ciphertext): Ciphertext en RLWE
        public_key (jnp.array): Clé publique correspondant aux a_i
        BSK (Tuple[RGSW]): Clé de boostrapping correspondant aux messages chiffrés RGSW(s_i)
        q (int): Modulo de l'espace des chiffrés
        beta (int): Base de décomposition
        l (int): PUissance maximale de décomposition
        degree (int): Degré des polyômes des chiffrés

    Returns:
        Ciphertext: Message chiffré déphasé de sum(a_i*s_i)
    """
    n = public_key.shape[-1]

    n_iter = n//collapse
    ct_init = jnp.stack(ciphertext) if isinstance(ciphertext, tuple) else ciphertext
    kronecker_matrix = all_binary_vectors(collapse)
    pk_reshaped = public_key.reshape(n_iter, collapse)
    all_rotations_val = jnp.dot(pk_reshaped, kronecker_matrix.T)
    all_rotations_indices = (jnp.round(jnp.round(2 * degree * all_rotations_val / q)/n_lut)*n_lut).astype(jnp.int32)%(2*degree)
    bsk_array = BSK
    shape = list(bsk_array.shape)
    shape[3] = n_iter
    shape.insert(4, 2**collapse) # On split l'axe 3 en (n_iter, chunk_size)
    bsk_reshaped = bsk_array.reshape(shape)
    bsk_reshaped = jnp.permute_dims(bsk_reshaped,(3,0,2,1,4,5))

    def step(ct_in, scan_input):
    #     # carry is not used; set to None or any placeholder
        bsk_chunk, rot_indices = scan_input
        dec = decomposition(ct_in, beta, l, q)
        dec_fft = jax_fourier(dec)
        sum_rgsw = jnp.einsum('abcmd, md -> abcd', bsk_chunk, all_rot_fft[rot_indices])

        # # On applique le BSK
        prod = jnp.einsum('abcd, bcd -> ad', sum_rgsw, dec_fft)
        ct_next = centered_mod(jax_inversefourier(prod),q)
        return ct_next, None
    
    ct_final, _ = jax.lax.scan(step,ct_init,(bsk_reshaped, all_rotations_indices),n_iter)
    return ct_final


def all_binary_vectors(m: int):
    """
    Génère une matrice (3^m, m) contenant toutes les combinaisons 
    de  0, 1} pour un vecteur de taille m.
    """
    # Définition de l'espace de base
    values = jnp.array([ 0, 1], dtype=jnp.int32)
    
    # Création des grilles de coordonnées pour chaque dimension
    # indexing='ij' assure l'ordre cartésien correct
    grids = jnp.meshgrid(*[values] * m, indexing='ij')
    
    # Empilement sur le dernier axe et redimensionnement
    # On transforme la liste de m grilles (3, 3, ..., 3) en (3^m, m)
    return jnp.stack(grids, axis=-1).reshape(-1, m)
