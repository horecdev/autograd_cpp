#pragma once
#include <stdexcept>
#include <vector>
#include "../tensor.hpp"
namespace gradc {
    inline std::vector<size_t> infer_broadcast(const std::vector<size_t>& a, const std::vector<size_t>& b) {
        size_t size_a = a.size();
        size_t size_b = b.size();

        std::vector<size_t> inferred_shape(std::max(size_a, size_b));
        
        if (size_a >= size_b) {
            for (size_t b_idx = 0; b_idx < size_b; ++b_idx) {
                size_t a_idx = b_idx + (size_a - size_b);
                size_t shape_a = a[a_idx];
                size_t shape_b = b[b_idx];
                if (shape_a == shape_b) {
                    inferred_shape[a_idx] = shape_a;
                }
                else if (shape_a == 1 || shape_b == 1) {
                    inferred_shape[a_idx] = std::max(shape_a, shape_b);
                }
                else {
                    throw std::runtime_error("Incompatible shapes for broadcasting.");
                }
            }
            for (size_t a_idx = 0; a_idx < (size_a - size_b); ++a_idx) {
                inferred_shape[a_idx] = a[a_idx];
            }
        }
        else if (size_a < size_b) {
            for (size_t a_idx = 0; a_idx < size_a; ++a_idx) {
                size_t b_idx = a_idx + (size_b - size_a);
                size_t shape_a = a[a_idx];
                size_t shape_b = b[b_idx];
                if (shape_a == shape_b) {
                    inferred_shape[b_idx] = shape_b;
                }
                else if (shape_a == 1 || shape_b == 1) {
                    inferred_shape[b_idx] = std::max(shape_a, shape_b);
                }
                else {
                    throw std::runtime_error("Incompatible shapes for broadcasting.");
                }
            }
            for (size_t b_idx = 0; b_idx < (size_b - size_a); ++b_idx) {
                inferred_shape[b_idx] = b[b_idx];
            }
        }

        return inferred_shape;   
    }   

    inline bool can_broadcast(const std::vector<size_t>& src, const std::vector<size_t>& target) {
        size_t size_src = src.size();
        size_t size_target = target.size();
        if (size_src > size_target) {
            return false;
        }

        for (size_t src_idx = 0; src_idx < size_src; ++src_idx) {
            size_t target_idx = src_idx + (size_target - size_src);
            if (src[src_idx] != target[target_idx] && src[src_idx] != 1) {
                return false;
            }
        }   
        return true;
    }

    template <typename T>
    Tensor<T> lobotomized_broadcast(const Tensor<T>& source, const std::vector<size_t>& target_shape) {
        int64_t n_dim_orig = static_cast<int64_t>(source.m_shape.size());
        int64_t n_dim_target = static_cast<int64_t>(target_shape.size());
        std::vector<size_t> new_shape = std::vector<size_t>(n_dim_target);
        std::vector<size_t> new_strides = std::vector<size_t>(n_dim_target);

        // align to the right
        for (int64_t i = 0; i < n_dim_orig; ++i) {
            new_shape[i + (n_dim_target - n_dim_orig)] = source.m_shape[i];
            new_strides[i + (n_dim_target - n_dim_orig)] = source.m_strides[i];
        }
        for (int64_t i = n_dim_target - n_dim_orig - 1; i >= 0; --i) { // fill leftovers (dont even have to check later)
            new_shape[i] = target_shape[i];
            new_strides[i] = 0;
        }

        for (int64_t i = (n_dim_target - n_dim_orig); i < n_dim_target; ++i) {
            if (target_shape[i] == new_shape[i]) {
                // literally leave everything as is
            }
            else if (new_shape[i] == 1) {
                new_shape[i] = target_shape[i];
                new_strides[i] = 0;
            }
            else {
                throw std::runtime_error("Violated broadcasting rules in lobotomized_broadcast().");
            }
        }

        return Tensor(std::move(new_shape), std::move(new_strides), source.m_offset, source.m_storage, nullptr, false); // lobotomy (no past or future)
    }

    template <typename T>
    Tensor<T> lobotomized_contiguous(const Tensor<T>& source) {
        if (source.m_shape.empty()) {
            Tensor<T> scalar_tensor = Tensor<T>(std::vector<size_t>{});
            ((*scalar_tensor.m_storage).m_data)[0] = ((*source.m_storage).m_data)[source.m_offset];
            return scalar_tensor;
        }
        Tensor<T> new_contiguous = Tensor<T>(source.m_shape); // right shape and strides, but Tensor will rip out only data
        size_t n_dims = source.m_shape.size();
        std::vector<size_t> odometer(n_dims, 0); // zeroed out
        size_t contiguous_idx = 0;
        while (odometer[0] < source.m_shape[0]) {
            size_t strided_idx = source.m_offset;
            for (size_t i = 0; i < n_dims; ++i) {
                strided_idx += odometer[i] * source.m_strides[i];
            }
            ((*new_contiguous.m_storage).m_data)[contiguous_idx] = ((*source.m_storage).m_data)[strided_idx];
            ++contiguous_idx;
            ++odometer[n_dims - 1];
            size_t i = n_dims - 1;
            while ((odometer[i] == source.m_shape[i]) && i > 0) {
                odometer[i] = 0;
                ++odometer[i - 1];
                --i;
            }
        }
        return new_contiguous;
    }
}