#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <type_traits>
#include <utility>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "fft.cuh"

#include "dec_fft.cuh"
#include "mul_bootstrap.cuh"

namespace ffi = xla::ffi;
namespace nb = nanobind;

template <typename T> nb::capsule EncapsulateFfiCall(T *fn) {
    // This check is optional, but it can be helpful for avoiding invalid
    // handlers.
    static_assert(
        std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
        "Encapsulated function must be and XLA FFI handler");
    return nb::capsule(reinterpret_cast<void *>(fn));
}

NB_MODULE(_core, m) {
    m.def("mul_bootstrap", []() { return EncapsulateFfiCall(MulBootstrap); });
}
