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
    """This function adds a cleartext message to an encrypted message

    Args:
        plaintext (Plaintext): Cleartext message
        ciphertext (Ciphertext): Encrypted message

    Returns:
        Ciphertext: Encrypted message
    """
    return ciphertext[0], ciphertext[1]+plaintext



@partial(jax.jit, static_argnames=['q','modulo'])
def sum_ciphertext_ciphertext(ciphertext1:Ciphertext, ciphertext2:Ciphertext,q:int,modulo=True)->Ciphertext:
    """This function computes the sum of two ciphertexts.

    Args:
        ciphertext1 (Ciphertext): Ciphertext1
        ciphertext2 (Ciphertext): Ciphertext2
        q (int): Modulus of the ciphertext space

    Returns:
        Ciphertext: Ciphertext1+ciphertext2
    """
    if modulo:
        return centered_mod(ciphertext1[0]+ciphertext2[0],q),centered_mod(ciphertext1[1]+ciphertext2[1],q)
    else:
        return ciphertext1[0]+ciphertext2[0], ciphertext1[1]+ciphertext2[1]


@partial(jax.jit, static_argnames=['q'])
def diff_ciphertext_ciphertext(ciphertext1:Ciphertext, ciphertext2:Ciphertext,q:int)->Ciphertext:
    """This function computes the difference between 2 ciphertexts.

    Args:
        ciphertext1 (Ciphertext): Ciphertext1
        ciphertext2 (Ciphertext): Ciphertext2
        q (int): Modulus of the ciphertext space

    Returns:
        Ciphertext: ciphertext1 - ciphertext2
    """
    return centered_mod(ciphertext1[0]-ciphertext2[0],q),centered_mod(ciphertext1[1]-ciphertext2[1],q)

@partial(jax.jit,static_argnames=["q"])
def centered_mod_ciphertext(ciphertext:Ciphertext,q:int):
    """This function returns the centered modulo q of an encrypted message

    Args:
        ciphertext (Ciphertext): Encrypted message
        q (int): modulus

    Returns:
        _type_: _description_
    """
    return centered_mod(ciphertext[0],q),centered_mod(ciphertext[1],q)


@partial(jax.jit, static_argnames=['stay_fft'])
def multiply_plaintext_plaintext(plaintext1:Plaintext, plaintext2:Plaintext,  stay_fft=False)->Plaintext:
    """Multiplication between two cleartext messages.

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
    """Multiplication of a ciphertext message by a plaintext message.

    Args:
        plaintext (Plaintext): Cleartext message
        ciphertext (Ciphertext): Encrypted message
        degree (int): Polynomial degree

    Returns:
        Ciphertext: Encrypted message
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
    """This function multiplies a ciphertext with an RGSW and returns an RLWE ciphertext.

    Args:
        ciphertext (Ciphertext): Ciphertext in RLWE, RLWE(m1)
        RGSW (RGSW): Ciphertext in RGSW, RGSW(m2)
        q (int): Modulus of the ciphertext space
        beta (int): Decomposition base
        l (int): Maximum decomposition power
        degree (int): Degree of the polynomials

    Returns:
        Ciphertext: RLWE(m1*m2) ciphertext
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
    """This function multiplies two ciphertexts together.

    Args:
        ciphertext1 (Ciphertext): Ciphertext1, RLWE(delta*m1)
        ciphertext2 (Ciphertext): Ciphertext2, RLWE(delta*m2)
        RGSW_sk (RGSW): Secret key encrypted in RGSW
        degree (int): Degree of the polynomials
        q (int): Modulus of the ciphertext space
        beta (int): Decomposition base
        l (int): Maximum decomposition power

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
    """This function applies the product related to the gadget decomposition.

    Args:
        glev (Ciphertext): Ciphertext in GLEV
        decomp (Plaintext): Plaintext decomposed following a base beta
        q (int): Modulus of the ciphertext space
        degree (int): Degree of the glev polynomials

    Returns:
        Ciphertext: Result of the product (close to a scalar product)
    """
    product = jax.vmap(multiply_plaintext_ciphertext,in_axes=(0,0,None,None,None))(decomp, glev, degree, True, glev_fft)
    product = jnp.stack(product)
    product = jnp.sum(product,axis=1)
    public_key_sum, b_sum = vmap(jax_inversefourier,0)(product)
    return public_key_sum,b_sum



@partial(jax.jit, static_argnames=['beta','l','q'])
def get_GLEV_ciphertext(c:Ciphertext,beta:int,l:int, q:int)->Ciphertext:
    """This function takes a ciphertext and returns its associated GLEV.

    Args:
        c (Ciphertext): Ciphertext, by default in RLWE
        beta (int): Decomposition base
        l (int): Maximum decomposition power
        q (int): Modulus of the ciphertext
        degree (int): Degree of the ciphertext

    Returns:
        Ciphertext: GLEV associated with the ciphertext c
    """
    public_key = c[0]
    b = c[1]
    glev_public_key = GLEV_polynomial(public_key,beta,l,q)
    glev_b = GLEV_polynomial(b,beta,l,q)
    return glev_public_key,glev_b

@partial(jax.jit, static_argnames=['current_modulus','new_modulus'])
def modulus_switch(c:Ciphertext,current_modulus:int,new_modulus:int):
    """This function switches from one modulus to another.

    Args:
        ciphertext (Ciphertext): Ciphertext on which we want to apply the modulus change
        current_modulus (int): The modulus of the input ciphertext
        new_modulus (int): The modulus we want for the output ciphertext

    Returns:
        Ciphertext: Ciphertext with the new modulus
    """
    pk, b = c
    new_pk = jnp.round(pk*new_modulus/current_modulus)
    new_b = jnp.round(b*new_modulus/current_modulus)
    return new_pk, new_b



jax.jit
def sample_extract(ciphertext:Ciphertext,alpha):
    """This function extracts a coefficient from an RLWE and returns the associated LWE

    Args:
        ciphertext (Ciphertext): Polynomial P encrypted in RLWE
        alpha (int): Index of the coefficient we want to extract from the polynomial

    Returns:
        Ciphertext: LWE corresponding to the ciphertext of P[index]
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
    """This function packs a set of LWE into an RLWE

    Args:
        ciphertexts (Ciphertext): The set of ciphertexts
        KSP (Ciphertext): key_switch key for the packing
        q (int): modulus of the ciphertext space
        beta (int): integer decomposition base
        l (int): maximum exponent for our decomposition
        degree (int): Degree of the polynomial desired at the end of the packing

    Returns:
        Ciphertext: Ciphertext in RLWE. Its coefficients are, from left to right, those of the list of LWE
    """
    public_keys,bs = ciphertexts[0],ciphertexts[1]
    n_lwes = bs.shape[0]
    packed_b = jnp.zeros((degree,))
    packed_b = packed_b.at[:n_lwes].set(jnp.squeeze(bs,axis=-1))
    public_key_polynomial = jnp.transpose(public_keys)
    if n_lwes < degree:##If the number of LWE is not equal to the degree of the polynomial, we pad with zeros to get polynomials of the right size
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
    Memory-optimized version of LWE -> RLWE packing.
    """
    public_keys, bs = ciphertexts[0], ciphertexts[1]
    n_lwes = bs.shape[0]
    n_lwe_dim = public_keys.shape[1]

    # 1. Using jnp.pad instead of .at[].set() or jnp.zeros()
    bs_squeezed = jnp.squeeze(bs, axis=-1)
    packed_b = jnp.pad(bs_squeezed, (0, degree - n_lwes))

    # 2. Transposition and idiomatic padding
    public_key_polynomial = jnp.transpose(public_keys)
    if n_lwes < degree:
        # Pad only the 'degree' dimension (axis 1)
        public_key_polynomial = jnp.pad(public_key_polynomial, ((0, 0), (0, degree - n_lwes)))

    # 3. Replacing vmap with jax.lax.scan to accumulate on the fly
    # This avoids materializing the (n_lwe_dim, degree, l) tensor in memory
    def scan_body(carry, elements):
        acc_a, acc_b = carry
        ksp_i, pk_i = elements

        # On-the-fly decomposition for a single LWE (size: degree)
        pk_i_decomp = decomposition(pk_i, beta, l, q)

        # Gadget product for this component
        res_a, res_b = gadget_product(ksp_i, pk_i_decomp, degree)

        # Accumulation
        return (acc_a + res_a, acc_b + res_b), None

    # Initialization of the accumulators (RLWE polynomials of size 'degree')
    init_carry = (
        jnp.zeros(degree, dtype=public_keys.dtype),
        jnp.zeros(degree, dtype=public_keys.dtype)
    )

    # Execution of the scan along the LWE dimension
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
    Step 1: Free mapping from LWE to RLWE.
    Exploits modular arithmetic (X^N + 1).
    """
    a,b = lwe
    # B(X) = b on coefficient 0
    B = jnp.zeros(degree)
    B = B.at[0].set(b.squeeze())

    # A(X) = a_0 - \sum_{i=1}^{N-1} a_{N-i} X^i
    # Using jnp.flip to optimize index reversal in XLA
    A = jnp.zeros(degree, dtype=a.dtype)
    A = A.at[0].set(a[0])
    A = A.at[1:].set(-jnp.flip(a[1:]))
    #breakpoint()
    return A, B

@partial(jax.jit, static_argnames=['k','beta','l','q','degree'])
def apply_automorphism_and_keyswitch(rlwe:Ciphertext, KSK: jax.Array, k: int,
                                    beta:int, l:int, q:int, degree:int ) -> Ciphertext:
    """
    Applies the automorphism X -> X^k and performs the Key Switch.
    In a complete implementation, this would ideally be done in the NTT domain.
    """
    rlwe = jnp.stack(rlwe)
    #A,B = rlwe


    # Note: We need to handle the sign changes related to X^N = -1
    # To simplify the skeleton, we illustrate the pure permutation.
    rlwe_auto = vmap(apply_automorphism,(0,None))(rlwe,k)

    A_perm, B_perm = rlwe_auto[0], rlwe_auto[1]

    # --- Base decomposition (Gadget Decomposition) ---
    # This is where 'Hoisting' comes into play: we decompose A_perm only once.
    A_dec = decomposition(A_perm,beta,l,q)

    # --- Tensor product with the evaluation key (CUDA/TPU-optimized GEMM) ---
    A_ks, B_ks = gadget_product(KSK, A_dec, degree)
    
    return sum_ciphertext_ciphertext([jnp.zeros_like(B_perm),B_perm],[A_ks,B_ks],q)

@partial(jax.jit, static_argnames=['beta','l','q','degree'])
def pack_lwe_to_rlwe(lwe:Ciphertext, galois_keys: list[Ciphertext],
                    beta:int, l:int, q:int, degree:int) -> Ciphertext:
    """
    Step 2: The Trace tree in O(log N).
    """
    # 1. Initial embedding
    lwe = modulus_switch(lwe,q,q/degree)
    rlwe = trivial_embedding(lwe, degree)

    # 2. Tree of automorphisms
    # XLA will unroll this loop since N is static, generating a linear and very fast CUDA graph.
    num_steps = int(math.log2(degree))

    for i in range(num_steps):
        # The automorphism exponent for the trace at step i
        k = 2**(num_steps - i) + 1
        # Applying the automorphism on the current ciphertext
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
    """This function applies a rotation to the ciphertext.

    Args:
        ciphertext (Ciphertext): Ciphertext in RLWE to which we apply a phase shift
        exponent (jnp.ndarray): Phase to apply to the ciphertext

    Returns:
        Ciphertext: Ciphertext in RLWE shifted by exponent
    """
    public_key, b = ciphertext[0],ciphertext[1]
    rotated_public_key = coef_rotation(public_key, exponent) ##We shift the public key
    rotated_b = coef_rotation(b,exponent) ##We shift the polynomial B
    return rotated_public_key, rotated_b


@partial(jax.jit,static_argnames=['q','beta','l','degree','collapse','n_lut'])
def blind_rotate(ciphertext_lwe:Ciphertext, LUT:Ciphertext, BSK:Tuple[RGSW],
                 q:int,beta:int,l:int,degree:int,collapse:int, all_rot_fft:jnp.array, n_lut:int):
    """This function performs a blind rotation, i.e. changing the phase of the polynomial without knowing the phase shift we wish to apply.

    Args:
        ciphertext_lwe (Ciphertext): Encrypted message, corresponding to the phase encrypted in LWE
        LUT (Ciphertext): Look-Up Table encrypted in RLWE
        BSK (RGSW): Bootstrapping key
        q (int): Modulus in the ciphertext space
        beta (int): Decomposition base
        l (int): Maximum power for the decomposition
        degree (int): Degree of the Look-Up Table

    Returns:
        RLWE: Polynomial shifted by the LWE ciphertext
    """
    public_key_lwe, b_lwe = ciphertext_lwe[0],ciphertext_lwe[1]
    b_lwe = jnp.round(2*degree*b_lwe/(q*n_lut))*n_lut
    ct_out = rotate_ciphertext(LUT,-b_lwe)
    ct_out = multiply_seq_monomial(ct_out,public_key_lwe,BSK,q,beta,l,degree,collapse, all_rot_fft,n_lut) ## We shift by sum(a_i*s_i) without knowing the s_i

    return ct_out
    

@partial(jax.jit,static_argnames=['q','beta','l','degree','collapse','n_lut'])
def multiply_seq_monomial(ciphertext:Ciphertext,public_key:jnp.array,BSK:Tuple[RGSW],
                          q:int,beta:int,l:int,degree:int,collapse:int, all_rot_fft:jnp.array,n_lut)->Ciphertext:
    """This function performs the sequential multiplication in the encrypted space of M by X^{sum(a_i*s_i)}.

    Args:
        ciphertext (Ciphertext): Ciphertext in RLWE
        public_key (jnp.array): Public key corresponding to the a_i
        BSK (Tuple[RGSW]): Bootstrapping key corresponding to the messages encrypted as RGSW(s_i)
        q (int): Modulus of the ciphertext space
        beta (int): Decomposition base
        l (int): Maximum decomposition power
        degree (int): Degree of the ciphertext polynomials

    Returns:
        Ciphertext: Encrypted message shifted by sum(a_i*s_i)
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
    shape.insert(4, 2**collapse) # We split axis 3 into (n_iter, chunk_size)
    bsk_reshaped = bsk_array.reshape(shape)
    bsk_reshaped = jnp.permute_dims(bsk_reshaped,(3,0,2,1,4,5))

    def step(ct_in, scan_input):
    #     # carry is not used; set to None or any placeholder
        bsk_chunk, rot_indices = scan_input
        dec = decomposition(ct_in, beta, l, q)
        dec_fft = jax_fourier(dec)
        sum_rgsw = jnp.einsum('abcmd, md -> abcd', bsk_chunk, all_rot_fft[rot_indices])

        # # We apply the BSK
        prod = jnp.einsum('abcd, bcd -> ad', sum_rgsw, dec_fft)
        ct_next = centered_mod(jax_inversefourier(prod),q)
        return ct_next, None
    
    ct_final, _ = jax.lax.scan(step,ct_init,(bsk_reshaped, all_rotations_indices),n_iter)
    return ct_final
    


    
@partial(jax.jit,static_argnames=['q','beta','l','degree','collapse','n_lut'])
def many_LUT_multiply_seq_monomial(ciphertext:Ciphertext,public_key:jnp.array,BSK:Tuple[RGSW],
                          q:int,beta:int,l:int,degree:int,collapse:int, all_rot_fft:jnp.array, n_lut:int)->Ciphertext:
    """This function performs the sequential multiplication in the encrypted space of M by X^{sum(a_i*s_i)}.

    Args:
        ciphertext (Ciphertext): Ciphertext in RLWE
        public_key (jnp.array): Public key corresponding to the a_i
        BSK (Tuple[RGSW]): Bootstrapping key corresponding to the messages encrypted as RGSW(s_i)
        q (int): Modulus of the ciphertext space
        beta (int): Decomposition base
        l (int): Maximum decomposition power
        degree (int): Degree of the ciphertext polynomials

    Returns:
        Ciphertext: Encrypted message shifted by sum(a_i*s_i)
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
    shape.insert(4, 2**collapse) # We split axis 3 into (n_iter, chunk_size)
    bsk_reshaped = bsk_array.reshape(shape)
    bsk_reshaped = jnp.permute_dims(bsk_reshaped,(3,0,2,1,4,5))

    def step(ct_in, scan_input):
    #     # carry is not used; set to None or any placeholder
        bsk_chunk, rot_indices = scan_input
        dec = decomposition(ct_in, beta, l, q)
        dec_fft = jax_fourier(dec)
        sum_rgsw = jnp.einsum('abcmd, md -> abcd', bsk_chunk, all_rot_fft[rot_indices])

        # # We apply the BSK
        prod = jnp.einsum('abcd, bcd -> ad', sum_rgsw, dec_fft)
        ct_next = centered_mod(jax_inversefourier(prod),q)
        return ct_next, None
    
    ct_final, _ = jax.lax.scan(step,ct_init,(bsk_reshaped, all_rotations_indices),n_iter)
    return ct_final


def all_binary_vectors(m: int):
    """
    Generates a matrix (3^m, m) containing all the combinations
    of {0, 1} for a vector of size m.
    """
    # Definition of the base space
    values = jnp.array([ 0, 1], dtype=jnp.int32)

    # Creation of the coordinate grids for each dimension
    # indexing='ij' ensures the correct Cartesian order
    grids = jnp.meshgrid(*[values] * m, indexing='ij')

    # Stacking on the last axis and reshaping
    # We transform the list of m grids (3, 3, ..., 3) into (3^m, m)
    return jnp.stack(grids, axis=-1).reshape(-1, m)
