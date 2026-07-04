#pragma once

#include "../tensor.hpp"
#include "../graph.hpp"
#include "tensor_utils.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace gradc {
    template <typename T>
    Tensor<T> Tensor<T>::relu() const {
        Tensor<T> result = Tensor<T>(m_shape, m_requires_grad, lazy);
        result.m_state->m_creation_op = std::make_unique<ReLUNode<T>>(*this);
        return result;
    }
}
