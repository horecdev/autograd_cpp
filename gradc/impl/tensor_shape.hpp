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
        Tensor<T> result = lobotomized_transpose_view(*this, dim0, dim1);
        result.m_state->m_creation_op = std::make_unique<TransposeNode<T>>(*this, dim0, dim1); 
        result.m_requires_grad = m_requires_grad;
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::permute(const std::vector<int64_t>& axes) const {
        Tensor<T> permuted = lobotomized_permute_view(*this, axes);
        permuted.m_state->m_creation_op = std::make_unique<PermuteNode<T>>(*this, axes);
        permuted.m_requires_grad = m_requires_grad;
        return permuted;
    }

    template <typename T>
    Tensor<T> Tensor<T>::reshape(const std::vector<int64_t>& target_shape) const {
        if (this->is_contiguous()) {
            Tensor<T> reshaped = lobotomized_reshape_view(*this, target_shape);
            reshaped.m_state->m_creation_op = std::make_unique<ReshapeNode<T>>(*this);
            reshaped.m_requires_grad = m_requires_grad;
            return reshaped;
        }
        else {
            Tensor<T> new_contig = this->contiguous();
            Tensor<T> reshaped = lobotomized_reshape_view(new_contig, target_shape);
            reshaped.m_state->m_creation_op = std::make_unique<ReshapeNode<T>>(new_contig);
            reshaped.m_requires_grad = new_contig.m_requires_grad;
            return reshaped;
        }
    }

    template <typename T>
    Tensor<T> lazy_concat(std::vector<Tensor<T>>& tensor_list, int64_t concat_dim) {
        const int64_t n_dim = std::ssize(tensor_list[0].m_shape);
        std::vector<int64_t> final_shape = tensor_list[0].m_shape;
        concat_dim = normalize_axis(concat_dim, n_dim);
        final_shape[concat_dim] = 0;
        bool requires_grad = false;

        for (const Tensor<T>& parent : tensor_list) {
            if (std::ssize(parent.m_shape) != n_dim) {
                throw std::runtime_error("Error during concat: Tensors must have the same number of dimensions.");
            }
            for (int64_t i = 0; i < n_dim; ++i) {
                if (i != concat_dim && parent.m_shape[i] != tensor_list[0].m_shape[i]) {
                    throw std::runtime_error("Error during concat: Tensors must match on non-concat dimensions.");
                }
            }
            final_shape[concat_dim] += parent.m_shape[concat_dim];
            requires_grad = requires_grad || parent.m_requires_grad;
        }

        Tensor<T> result = Tensor<T>(final_shape, requires_grad, lazy);
        result.m_state->m_creation_op = std::make_unique<ConcatNode<T>>(tensor_list, concat_dim, std::move(final_shape));
        
        return result;
    }
}