#pragma once

#include "../tensor.hpp"
#include "../graph.hpp"
#include "tensor_utils.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace gradc {
    
    template <typename T>
    Tensor<T> Tensor<T>::contiguous() const {
        Tensor result = Tensor(m_shape, m_requires_grad, lazy);
        result.m_state->m_creation_op = std::make_unique<ContiguousNode<T>>(*this);

        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::transpose(int64_t dim0, int64_t dim1) const {
        Tensor<T> result = lobotomized_transpose(*this, dim0, dim1);
        result.m_state->m_creation_op = std::make_unique<TransposeNode<T>>(*this, dim0, dim1); 
        result.m_requires_grad = m_requires_grad;
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::permute(const std::vector<int64_t>& axes) const {
        Tensor<T> permuted = lobotomized_permute(*this, axes);
        permuted.m_state->m_creation_op = std::make_shared<PermuteNode<T>>(*this, axes);
        permuted.m_requires_grad = m_requires_grad;
        return permuted;
    }

    template <typename T>
    Tensor<T> Tensor<T>::reshape(const std::vector<int64_t>& target_shape) const {
        if (this->is_contiguous()) {
            Tensor<T> reshaped = lobotomized_reshape(*this, target_shape);
            reshaped.m_state->m_creation_op = std::make_unique<ReshapeNode<T>>(*this);
            reshaped.m_requires_grad = m_requires_grad;
            return reshaped;
        }
        else {
            Tensor<T> new_contig = this->contiguous();
            Tensor<T> reshaped = lobotomized_reshape(new_contig, target_shape);
            reshaped.m_state->m_creation_op = std::make_unique<ReshapeNode<T>>(new_contig);
            reshaped.m_requires_grad = new_contig.m_requires_grad;
            return reshaped;
        }
    }

    template <typename T>
    Tensor<T> concat(std::vector<Tensor<T>>& tensor_list, int64_t concat_dim) {
        
    }
}