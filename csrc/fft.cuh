#pragma once

#include <cufftdx.hpp>

using namespace cufftdx;

// Calcul dynamique de l'ElementsPerThread optimal à la compilation.
// On garantit que la taille du bloc (N / EPT) ne dépasse jamais 512 threads.
// Cela permet d'allouer 128 registres par thread, empêchant le crash sur L=3 ou L=4.
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
