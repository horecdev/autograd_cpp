#pragma once

#include "../core/tensor.hpp"

namespace gradc {
    template <typename T1, typename T2>
    inline Device infer_assert_device(const Tensor<T1>& t1, const Tensor<T2>& t2) {
        if (t1.device() != t2.device()) {
            throw std::runtime_error("Operation failed: both (2) Tensors must be on the same device.");
        }
        return t1.device();
    }


    template <typename T>
    inline Device infer_assert_device(const std::vector<Tensor<T>>& tensors) {
        // supposes that input vector is NOT empty
        Device target_device = tensors[0].device();
        for (int64_t i = 1; i < std::ssize(tensors); ++i) {
            if (tensors[i].device() != target_device) {
                throw std::runtime_error("Operation failed: all (2+) Tensors must be on the same device.");
            }
        }

        return target_device;
    }
}