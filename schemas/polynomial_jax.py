import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import time
jax.config.update('jax_enable_x64', True)

@partial(jax.jit, static_argnames=['q'])
def centered_mod(x, q):
    return x - q * jnp.rint(x/q)

@jax.jit
def jax_fourier(p):
    degree = p.shape[-1]
    q_large = jnp.fft.fft(jnp.exp(-1j * jnp.pi * jnp.arange(degree) / degree) * p)
    return q_large[...,:degree // 2]

@jax.jit
def jax_inversefourier(q):
    q_large = jnp.concatenate([q, jnp.flip(jnp.conj(q),axis=-1)],axis=-1)
    n = 2*q.shape[-1]
    return jnp.round(jnp.real(jnp.exp(1j * jnp.pi * jnp.arange(n) / n) * jnp.fft.ifft(q_large)))
    #return jnp.real(jnp.exp(1j * jnp.pi * np.arange(n) / n) * jnp.fft.ifft(q_large))



@partial(jax.jit, static_argnames=['stay_fft'])
def fft_polynomial_multiply(poly1, poly2,stay_fft=False):
    """Function estimating the exact cyclotomic multiplication between two polynomials

    Args:
        poly1 (jnp.ndarray): Polynomial 1
        poly2 (jnp.ndarray): Polynomial 2
        degree (jnp.ndarray): Degree of the polynomials

    Returns:
        jnp.ndarray: poly1*poly2
    """
    fft_poly1 = jax_fourier(poly1)
    fft_poly2 = jax_fourier(poly2)
    if stay_fft:
        return fft_poly1 * fft_poly2
    result = jax_inversefourier(fft_poly1 * fft_poly2)
    return jnp.round(result)

@partial(jax.jit, static_argnames=['stay_fft'])
def fft_product_twice(poly1, poly2, poly3, stay_fft = False):
    polys = jnp.stack([poly1,poly2,poly3])
    fft_poly1, fft_poly2, fft_poly3 = jax.vmap(jax_fourier,0)(polys)
    poly_prod_fft = [fft_poly1 * fft_poly2, fft_poly1 * fft_poly3]
    if stay_fft:
        return poly_prod_fft
    poly_prod_fft = jnp.stack(poly_prod_fft)
    result1, result2 = jax.vmap(jax_inversefourier,0)(poly_prod_fft)
    return result1, result2

@partial(jax.jit, static_argnames=['stay_fft'])
def partial_fft_product(poly1, fft_poly2, fft_poly3, stay_fft = False):
    fft_poly1 = jax_fourier(poly1)
    poly_prod_fft = [fft_poly1 * fft_poly2, fft_poly1 * fft_poly3]
    if stay_fft:
        return poly_prod_fft
    poly_prod_fft = jnp.stack(poly_prod_fft)
    result1, result2 = jax.vmap(jax_inversefourier,0)(poly_prod_fft)
    return result1, result2

@partial(jax.jit, static_argnames=['degree'])
def exact_polynomial_multiply(poly1, poly2,degree):
    """Function computing the exact cyclotomic multiplication between two polynomials

    Args:
        poly1 (jnp.ndarray): Polynomial 1
        poly2 (jnp.ndarray): Polynomial 2
        degree (jnp.ndarray): Degree of the polynomials

    Returns:
        jnp.ndarray: poly1*poly2
    """
    start = time.time()
    prod_mul = jnp.polymul(poly1,poly2)
    print(f"Product time : {time.time() - start}")
    start = time.time()
    prod_mul = prod_mul[:degree] - jnp.pad(prod_mul[degree:], (0, degree - prod_mul[degree:].size))
    print(f"cyclotomique produit : {time.time() - start}")
    return prod_mul





@partial(jax.jit, static_argnames=['beta','l','q'])
def decomposition(gamma, beta, l,q):
    """
    gamma: JAX array or scalar
    q, beta: scalars
    levels: int, number of γ_i digits to extract
    """
    # Start with x = gamma/q (JAX array)
    x0 = gamma / q

    def step(x, _):
        # multiply by base
        x_beta = x * beta
        # digit = floor(x_beta)
        digit = jnp.round(x_beta)
        # update fractional part
        new_x = x_beta - digit
        return new_x, digit

    # Scan through 'levels' iterations
    _, digits = jax.lax.scan(step, x0, None, length=l,unroll=20)
    return digits



@partial(jax.jit, static_argnames=['beta','l','q'])
def GLEV_polynomial(x:jnp.ndarray,beta:int,l:int,q:int):
    """Function used to compute polynomials for the GLWE before encrypting it

    Args:
        x (jnp.ndarray): polynomial
        beta (int): decomposition base
        l (int): Maximum decomposition power
        q (int): Modulus of the ciphertexts
        degree (int): Degree of the polynomial

    Returns:
        jnp.ndarray: l polynomials
    """
    #decomp_basis = jnp.round(jnp.expand_dims(q*jnp.power(1/beta,power),1))
    decomp_basis = jnp.array([q/(beta**i) for i in range(1,l+1)])
    decomp_basis = jnp.expand_dims(decomp_basis,1)
    #decomp_basis = jnp.expand_dims(q*jnp.power(scale,power),1)
    glev = x*decomp_basis
    return glev

@jax.jit
def weights_to_polynomial(weights):
    """Function transforming the weights of a linear layer into a polynomial

    Args:
        weights (jnp.ndarray): Weights W

    Returns:
        jnp.ndarray: Polynomial derived from the linear weights
    """
    polynomial = -jnp.flip(weights.flatten(),axis=-1)
    polynomial = polynomial.at[-1].set(weights[0])
    return polynomial

@jax.jit
def coef_rotation(polynomial:jnp.array, alpha):
    """This function applies a rotation of a polynomial's coefficients by the integer alpha

    Args:
        polynomial (jnp.array): A polynomial
        alpha (_type_): Rotation exponent

    Returns:
        jnp.ndarray: Polynomial after rotation
    """
    n = polynomial.shape[-1]
    rotation = jnp.roll(polynomial,alpha)
    degree_shift_div = alpha//n
    degree_shift_mod = alpha % n
    rotation = rotation*((-1)**(degree_shift_div))
    mask = jnp.arange(n) < degree_shift_mod
    #rotation = jnp.where(mask, -rotation, rotation)
    rotation = mask*(-rotation) + (1-mask)*rotation
    return rotation

@jax.jit
def apply_automorphism(polynomial:jnp.array, alpha):
    # if alpha % 2:
    #     new_p = apply_automorphism(polynomial,alpha+1)
    #     new_p = apply_automorphism(new_p,-1)
    #     return new_p
    N = polynomial.shape[0]
    
    # 1. Computing the destination indices in Z_{2N}
    # k and N being static (N is deduced from the shape),
    # XLA resolves this equation at compile time.
    indices_2n = (jnp.arange(N) * alpha) % (2 * N)

    # 2. Separation of the pure indices (modulo N)
    target_indices = indices_2n % N

    # 3. Handling the sign change (X^N = -1)
    # The modular opposite of poly without risk of underflow on uints.
    poly_neg = -polynomial

    # If the index 2N exceeds N, the term has done a "wrap-around" and takes a negative sign.
    poly_signed = jnp.where(indices_2n >= N, poly_neg, polynomial)

    # 4. Permutation of the coefficients (Scatter)
    # In JAX, this vectorized scatter on constant indices will be compiled
    # by XLA into a pure in-memory permutation (very fast).
    new_poly = jnp.zeros_like(polynomial)
    new_poly = new_poly.at[target_indices].set(poly_signed)
    
    return new_poly



