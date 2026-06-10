#pragma once

#include "../tensor.hpp"
#include "../graph.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gradc {
    template <typename T>
    bool Tensor<T>::is_contiguous() const {
        if (m_shape.empty()) return true;
        if (m_strides[m_strides.size() - 1] != 1) {
            return false;
        }
        for (size_t i = m_shape.size() - 1; i > 0; --i) {
            if (m_strides[i - 1] != m_shape[i] * m_strides[i]) {
                return false;
            }
        }
        return true;
    }
    
    template <typename T>
    Tensor<T> Tensor<T>::contiguous() const {
        Tensor result = Tensor(m_shape, m_requires_grad, lazy);
        result.m_state->m_realize_op = std::make_unique<ContiguousNode<T>>(*this);

        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::transpose(int64_t dim0, int64_t dim1) const {
        if (dim0 >= m_shape.size() || dim1 >= m_shape.size()) {

        }
        if (dim0 < 0) {dim0 = m_shape.size() + dim0;}
        if (dim1 < 0) {dim0 = m_shape.size() + dim0;}
        if (dim0 >= m_shape.size() || dim1 >= m_shape.size()) {
            throw std::out_of_range("Invalid indices for .transpose() - index out of shape bounds");
        }

        std::vector<size_t> new_shape = m_shape;
        std::vector<size_t> new_strides = m_strides;
        std::swap(new_shape[dim0], new_shape[dim1]);
        std::swap(new_strides[dim0], new_strides[dim1]);

        Tensor result = Tensor(std::move(new_shape), std::move(new_strides), m_offset, m_state->m_storage, m_requires_grad);
        result.m_state->m_realize_op = std::make_unique<TransposeNode<T>>(*this); 
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::permute(const std::vector<int64_t>& axes) const {
        size_t n_dim = m_shape.size();
        if (axes.size() != n_dim) {
            throw std::runtime_error("permute() axes list size must match shape list size.");
        }
        std::vector<size_t> new_shape = std::vector<size_t>(n_dim);
        std::vector<size_t> new_strides = std::vector<size_t>(n_dim);
        std::vector<bool> seen_axes = std::vector<bool>(n_dim, false);
        for (size_t target_ax = 0; target_ax < n_dim; ++target_ax) {
            int64_t src_ax = axes[target_ax];
            if (src_ax < 0) {
                src_ax = n_dim + src_ax;
            }
            if (src_ax >= n_dim) {
                throw std::out_of_range("Invalid indices for .permute() - index out of shape bounds");
            }
            if (seen_axes[src_ax]) {
                throw std::runtime_error("Passed at least one axis twice inside .permute()");
            }
            seen_axes[src_ax] = true;
             
            new_shape[target_ax] = m_shape[src_ax];
            new_strides[target_ax] = m_strides[src_ax];
        }
        Tensor result = Tensor(std::move(new_shape), std::move(new_strides), m_offset, m_state->m_storage, m_requires_grad);
        result.m_state->m_realize_op = std::make_unique<PermuteNode<T>>(*this);
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::reshape(const std::vector<int64_t>& target_shape) const {
        std::vector<size_t> new_shape = std::vector<size_t>(target_shape.size());
        std::vector<size_t> new_strides = std::vector<size_t>(target_shape.size());
        size_t total_volume = 1;
        size_t running_volume = 1;
        int64_t neg_one_idx = -1;

        for (size_t i = 0; i < m_shape.size(); ++i) {
            total_volume *= m_shape[i];
        }
        
        for (size_t i = 0; i < target_shape.size(); ++i) {
            if (target_shape[i] == -1 && neg_one_idx == -1) {
                neg_one_idx = i;
            }
            else if (target_shape[i] == -1 && neg_one_idx != -1) {
                throw std::runtime_error("Cannot .reshape() with two or more unknown dimensions.");
            }
            else {
                new_shape[i] = target_shape[i];
                running_volume *= target_shape[i];
            }
        }

        size_t unknown_dim;
        if (total_volume % running_volume != 0) {
            throw std::runtime_error("Invalid reshape parameters.");
        }
        else {
            unknown_dim = total_volume / running_volume;
        }

        if (neg_one_idx != -1) {
            new_shape[neg_one_idx] = unknown_dim;
        }

        new_strides[target_shape.size() - 1] = 1;
        for (size_t i = target_shape.size() - 1; i > 0; --i) {
            new_strides[i - 1] = new_shape[i] * new_strides[i];
        }

        if (this->is_contiguous()) {
            Tensor result = Tensor(std::move(new_shape), std::move(new_strides), m_offset, m_state->m_storage, m_requires_grad);
            result.m_state->m_realize_op = std::make_unique<ReshapeNode<T>>(*this);
            return result;
        }
        else {
            Tensor contiguous_tensor = this->contiguous(); // has right metadata and storage (lazy node)
            Tensor result = Tensor(std::move(new_shape), std::move(new_strides), 0, contiguous_tensor.m_state->m_storage, m_requires_grad);
            result.m_state->m_realize_op = std::make_unique<ReshapeNode<T>>(std::move(contiguous_tensor));
            return result;
        }
    }
}