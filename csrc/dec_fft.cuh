#pragma once
#include "fft.cuh"

template <int N, int L>
__device__ void decomposition(const int64_t * __restrict__ x0, double * __restrict__ output,
                              const int log2_beta, const int log2_q) {
    uint64_t offset = 0;
    
    #pragma unroll
    for (int l_idx = 1; l_idx <= L; l_idx++) {
        offset += (1ULL << (log2_q - l_idx * log2_beta + log2_beta - 1));
    }

    int rounding_shift = log2_q - L * log2_beta - 1;
    if (rounding_shift >= 0 && rounding_shift < 64) {
        offset += (1ULL << rounding_shift);
    }

    for (int i = threadIdx.x; i < 2 * N; i += blockDim.x) {
        uint64_t val = (uint64_t)x0[i];
        uint64_t state = val + offset;

        #pragma unroll
        for (int l_idx = 1; l_idx <= L; l_idx++) {
            int shift = log2_q - (l_idx * log2_beta);
            uint64_t digit = (state >> shift) & ((1ULL << log2_beta) - 1);
            int64_t centered_digit = (int64_t)digit - (1LL << (log2_beta - 1));
            output[(l_idx - 1) * 2 * N + i] = (double)centered_digit;
        }
    }
}

template <typename FFT, typename ComplexType, int N, int L>
__device__ void apply_fft(double * __restrict__ smem_dec, ComplexType * __restrict__ fft_workspace,
                          ComplexType * __restrict__ output, const int d_size) {
    constexpr double scale = 1. / (double)N;
    ComplexType thread_data[FFT::storage_size];

    for (int poly = 0; poly < 2 * L; poly++) {
        double * __restrict__ poly_in = smem_dec + poly * N;
        ComplexType * __restrict__ poly_out = output + poly * d_size;

        for (int i = 0; i < (int)FFT::storage_size; i++) {
            int k = threadIdx.x + i * blockDim.x;
            double s, c;
            sincospi(-(double)k * scale, &s, &c);
            thread_data[i] = ComplexType{poly_in[k] * c, poly_in[k] * s};
        }

        FFT().execute(thread_data, fft_workspace);
        __syncthreads();

        for (int i = 0; i < (int)FFT::storage_size; i++) {
            int k = threadIdx.x + i * blockDim.x;
            if (k < d_size) poly_out[k] = thread_data[i];
        }
    }
}