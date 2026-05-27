#pragma once
#include <numeric>
#include <functional>
#include <algorithm>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "fft.cuh"
#include "dec_fft.cuh"

namespace ffi = xla::ffi;

template <typename ComplexType>
__device__ __forceinline__ ComplexType cx_ldg(const ComplexType *ptr) {
    double2 v = __ldg(reinterpret_cast<const double2 *>(ptr));
    return ComplexType{v.x, v.y};
}

template <typename FFT, typename FFT_INV, typename ComplexType, int N, int L>
__device__ __noinline__ void
step(const ComplexType * __restrict__ bsk,         
     const ComplexType * __restrict__ all_rot_fft, 
     const int32_t * __restrict__ step_indices,    
     int64_t * __restrict__ ct,                    
     double * __restrict__ dec,                    
     ComplexType * __restrict__ dec_fft,
     ComplexType * __restrict__ fft_workspace,     
     ComplexType * __restrict__ prod,
     int64_t * __restrict__ output, 
     const int log2_beta, const int log2_q, const int a_size, const int bc_size,
     const int m_size, const int d_size) {

    decomposition<N, L>(ct, dec, log2_beta, log2_q);
    __syncthreads();
    apply_fft<FFT, ComplexType, N, L>(dec, fft_workspace, dec_fft, d_size);
    __syncthreads();

    constexpr double inv_N = 1.0 / N;
    constexpr int BC = 2 * L;

    for (int a = 0; a < a_size; a++) {
        for (int d = threadIdx.x; d < d_size; d += blockDim.x) {
            ComplexType acc_bc[BC];
            
            #pragma unroll
            for (int bc = 0; bc < BC; bc++) {
                acc_bc[bc] = ComplexType{0.0, 0.0};
            }

            for (int m = 0; m < m_size; m++) {
                ComplexType rot = cx_ldg(&all_rot_fft[step_indices[m] * d_size + d]);
                
                #pragma unroll
                for (int bc = 0; bc < BC; bc++) {
                    ComplexType b_val = cx_ldg(&bsk[((a * BC + bc) * m_size + m) * d_size + d]);
                    acc_bc[bc] += b_val * rot;
                }
            }

            ComplexType thread_data = ComplexType{0.0, 0.0};
            #pragma unroll
            for (int bc = 0; bc < BC; bc++) {
                thread_data += acc_bc[bc] * dec_fft[bc * d_size + d];
            }
            prod[d] = thread_data;
        }
        __syncthreads();

        ComplexType thread_data_inv[FFT_INV::storage_size];
        
        for (int i = 0; i < (int)FFT_INV::storage_size; i++) {
            int k = threadIdx.x + i * blockDim.x;

            if (k < d_size) {
                thread_data_inv[i] = prod[k];
            } else {
                thread_data_inv[i] = ComplexType{prod[N - 1 - k].x, -prod[N - 1 - k].y};
            }
        }
        
        FFT_INV().execute(thread_data_inv, fft_workspace);

        const double q = ldexp(1.0, log2_q);
        const double inv_q = ldexp(1.0, -log2_q);
        for (int i = 0; i < (int)FFT_INV::storage_size; i++) {
            int k = threadIdx.x + i * blockDim.x;
            double sk, ck;
            sincospi((double)k * inv_N, &sk, &ck);
            double val_double = (thread_data_inv[i].x * ck - thread_data_inv[i].y * sk) * inv_N;
            val_double = val_double - q * rint(val_double * inv_q);
            output[a * N + k] = __double2ll_rn(val_double);
        }
    }
}

template <typename FFT, typename FFT_INV, typename ComplexType, int N, int L>
__global__ __launch_bounds__(1024)
void mul_bootstrap_kernel(const int64_t * __restrict__ ct_init,     
                          const ComplexType * __restrict__ bsk,     
                          const ComplexType * __restrict__ all_rot_fft, 
                          const int32_t * __restrict__ indices,         
                          int64_t * __restrict__ output,            
                          const int log2_beta, const int log2_q, const int n_iter,
                          const int a_size, const int bc_size, const int m_size,
                          const int d_size,
                          const int ct_batch_stride) {
    extern __shared__ char raw_smem[];

    int64_t * __restrict__ smem_ct = (int64_t *)raw_smem;
    double * __restrict__ smem_dec = (double *)(raw_smem + sizeof(int64_t) * 2 * N);
    ComplexType * __restrict__ fft_workspace = (ComplexType *)(raw_smem + sizeof(double) * 2 * (L + 1) * N);
    ComplexType * __restrict__ dec_fft_smem = (ComplexType *)(raw_smem + sizeof(double) * 2 * N);
    ComplexType * __restrict__ prod_smem = fft_workspace;

    const int64_t * __restrict__ my_ct_init = ct_init + blockIdx.x * ct_batch_stride;
    for (int idx = threadIdx.x; idx < 2 * N; idx += blockDim.x) {
        smem_ct[idx] = my_ct_init[idx];
    }
    __syncthreads();

    for (int n = 0; n < n_iter; n++) {
        const int32_t * __restrict__ step_indices = &indices[(blockIdx.x * n_iter + n) * m_size];
        step<FFT, FFT_INV, ComplexType, N, L>(
            bsk + n * a_size * bc_size * m_size * d_size,
            all_rot_fft,  
            step_indices, 
            smem_ct,
            smem_dec, dec_fft_smem, fft_workspace, prod_smem, smem_ct,
            log2_beta, log2_q, a_size, bc_size, m_size, d_size);
    }

    for (int idx = threadIdx.x; idx < 2 * N; idx += blockDim.x) {
        output[blockIdx.x * 2 * N + idx] = smem_ct[idx];
    }
}

template <int N, int L>
ffi::Error mul_bootstrap_impl(cudaStream_t stream, ffi::Buffer<ffi::S64> ct_init,
                              ffi::Buffer<ffi::C128> bsk, 
                              ffi::Buffer<ffi::C128> all_rot_fft, 
                              ffi::Buffer<ffi::S32> indices,      
                              int64_t log2_beta, int64_t log2_q,
                              ffi::ResultBuffer<ffi::S64> output) {
    using FFT = FFT_N<N>;
    using FFT_INV = FFT_INV_N<N>;
    using ComplexType = typename FFT::value_type;

    auto bsk_dims = bsk.dimensions();
    int bsk_offset = (int)bsk_dims.size() - 6;
    int n_iter = bsk_dims[bsk_offset + 0];
    int a_size = bsk_dims[bsk_offset + 1];
    int b_size = bsk_dims[bsk_offset + 2];
    int c_size = bsk_dims[bsk_offset + 3];
    int m_size = bsk_dims[bsk_offset + 4];
    int d_size = bsk_dims[bsk_offset + 5];

    auto out_dims = output->dimensions();
    int batch_size = std::accumulate(out_dims.begin(), out_dims.end() - 2, 1, std::multiplies<int>());
    int bc_size = b_size * c_size;
    
    auto ct_dims = ct_init.dimensions();
    int64_t ct_total_elements = std::accumulate(ct_dims.begin(), ct_dims.end(), 1LL, std::multiplies<int64_t>());
    int ct_batch_stride = (ct_total_elements > 2 * N) ? (2 * N) : 0;

    auto *ct_ptr = ct_init.typed_data();
    auto *bsk_ptr = reinterpret_cast<ComplexType *>(bsk.typed_data());
    auto *rot_ptr = reinterpret_cast<ComplexType *>(all_rot_fft.typed_data());
    auto *idx_ptr = indices.typed_data(); 
    auto *out_ptr = output->typed_data();

    size_t required_prod_size = d_size * sizeof(ComplexType);
    size_t max_workspace = std::max((size_t)FFT::shared_memory_size, required_prod_size);
    size_t smem_size = sizeof(int64_t) * 2 * N + sizeof(double) * 2 * L * N + max_workspace;

    auto kernel = mul_bootstrap_kernel<FFT, FFT_INV, ComplexType, N, L>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    kernel<<<batch_size, FFT::block_dim, smem_size, stream>>>(
        ct_ptr, bsk_ptr, rot_ptr, idx_ptr, out_ptr, (int)log2_beta, (int)log2_q, n_iter,
        a_size, bc_size, m_size, d_size, ct_batch_stride);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        char msg[256];
        snprintf(msg, sizeof(msg), "Kernel launch failed: %s", cudaGetErrorString(err));
        return ffi::Error::Internal(msg);
    }
    return ffi::Error::Success();
}

ffi::Error MulBootstrapImpl(cudaStream_t stream, ffi::Buffer<ffi::S64> ct_init,
                            ffi::Buffer<ffi::C128> bsk,
                            ffi::Buffer<ffi::C128> all_rot_fft,
                            ffi::Buffer<ffi::S32> indices,      
                            int64_t log2_beta, int64_t log2_q, int64_t l, 
                            int64_t degree,
                            ffi::ResultBuffer<ffi::S64> output) {
#define CASE_L(N, L)                                                           \
    case L:                                                                    \
        return mul_bootstrap_impl<N, L>(stream, ct_init, bsk, all_rot_fft,     \
                                        indices, log2_beta, log2_q, output);

#define CASE_N(N)                                                              \
    case N:                                                                    \
        switch (l) {                                                           \
            CASE_L(N, 1)                                                       \
            CASE_L(N, 2)                                                       \
            CASE_L(N, 3)                                                       \
            CASE_L(N, 4)                                                       \
        default:                                                               \
            return ffi::Error::InvalidArgument("Unsupported l");               \
        }

    switch (degree) {
        CASE_N(128)
        CASE_N(256)
        CASE_N(512)
        CASE_N(1024)
        CASE_N(2048)
        CASE_N(4096)
    default:
        return ffi::Error::InvalidArgument("Unsupported degree");
    }

#undef CASE_L
#undef CASE_N
}

XLA_FFI_DEFINE_HANDLER(
    MulBootstrap, MulBootstrapImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S64>>()  
        .Arg<ffi::Buffer<ffi::C128>>() 
        .Arg<ffi::Buffer<ffi::C128>>() 
        .Arg<ffi::Buffer<ffi::S32>>()  
        .Attr<int64_t>("log2_beta")
        .Attr<int64_t>("log2_q")
        .Attr<int64_t>("l")
        .Attr<int64_t>("degree")
        .Ret<ffi::Buffer<ffi::S64>>()  
);