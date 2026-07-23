#pragma once

#include "../../core/types.hpp"
#include <cstdint>

namespace gradc {
    class CUDABackend {
    public:
        template <typename T>
        static void fill(T* ptr, T val, int64_t size, Device device);
    };
}
