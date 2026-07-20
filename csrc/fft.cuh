#pragma once

#include <cufftdx.hpp>

using namespace cufftdx;

// Dynamic computation of the optimal ElementsPerThread at compile time.
// We guarantee that the block size (N / EPT) never exceeds 512 threads.
// This allows allocating 128 registers per thread, preventing the crash on L=3 or L=4.
template <int N>
constexpr unsigned int optimal_ept() {
    return (N >= 1024) ? (N / 512) : 2;
}

// Cible sm_100 (GB200 Blackwell)
template <int N>
using FFT_N = decltype(Size<N>() + Precision<double>() + Type<fft_type::c2c>() +
                       Direction<fft_direction::forward>() + FFTsPerBlock<1>() +
                       ElementsPerThread<optimal_ept<N>()>() + SM<900>() + Block());

// Cible sm_100 (GB200 Blackwell)
template <int N>
using FFT_INV_N =
    decltype(Size<N>() + Precision<double>() + Type<fft_type::c2c>() +
             Direction<fft_direction::inverse>() + FFTsPerBlock<1>() +
             ElementsPerThread<optimal_ept<N>()>() + SM<900>() + Block());
