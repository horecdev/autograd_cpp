#pragma once

#include "../tensor.hpp"
#include "../graph.hpp"
#include "tensor_utils.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
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
        const int64_t n_dim = std::ssize(m_shape);
        if (std::ssize(axes) != n_dim) {
            throw std::runtime_error("permute() axes list size must match shape list size.");
        }
        std::vector<int64_t> new_shape = std::vector<int64_t>(n_dim);
        std::vector<int64_t> new_strides = std::vector<int64_t>(n_dim);
        std::vector<bool> seen_axes = std::vector<bool>(n_dim, false);
        for (int64_t target_ax = 0; target_ax < n_dim; ++target_ax) {
            int64_t src_ax = axes[target_ax];
            src_ax = normalize_axis(src_ax, n_dim);
            if (seen_axes[src_ax]) {
                throw std::runtime_error("Passed at least one axis twice inside .permute()");
            }
            seen_axes[src_ax] = true;

            new_shape[target_ax] = m_shape[src_ax];
            new_strides[target_ax] = m_strides[src_ax];
        }
        Tensor result = Tensor(std::move(new_shape), std::move(new_strides), m_offset, m_state->m_storage, m_requires_grad);
        result.m_state->m_creation_op = std::make_unique<PermuteNode<T>>(*this);
        return result;
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
            Tensor<T> new_contig = lobotomized_contiguous(*this);
            new_contig.m_state->m_creation_op = std::make_unique<ContiguousNode<T>>(*this);
            new_contig.m_requires_grad = m_requires_grad;
            Tensor<T> reshaped = lobotomized_reshape(new_contig, target_shape);
            reshaped.m_state->m_creation_op = std::make_unique<ReshapeNode<T>>(new_contig);
            reshaped.m_requires_grad = new_contig.m_requires_grad;
            return reshaped;
        }
    }
}