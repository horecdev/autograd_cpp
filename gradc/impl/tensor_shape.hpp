#pragma once
#include "../tensor.hpp"

namespace gradc {
    template <typename T>
    bool Tensor<T>::is_contiguous() const {
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
        if (m_shape.empty()) {
            Tensor scalar_tensor = Tensor(std::vector<size_t>{});
            ((*scalar_tensor.m_storage).m_data)[0] = ((*m_storage).m_data)[m_offset];
            return scalar_tensor;
        }
        Tensor new_contiguous = Tensor(m_shape); // already right size and right contiguous strides
        size_t n_dims = m_shape.size();
        std::vector<size_t> odometer(n_dims, 0); // zeroed out
        size_t contiguous_idx = 0;
        while (odometer[0] < m_shape[0]) {
            size_t strided_idx = m_offset;
            for (size_t i = 0; i < n_dims; ++i) {
                strided_idx += odometer[i] * m_strides[i];
            }
            ((*new_contiguous.m_storage).m_data)[contiguous_idx] = ((*m_storage).m_data)[strided_idx];
            ++contiguous_idx;
            ++odometer[n_dims - 1];
            size_t i = n_dims - 1;
            while ((odometer[i] == m_shape[i]) && i > 0) {
                odometer[i] = 0;
                ++odometer[i - 1];
                --i;
            }
        }
        return new_contiguous;
    }

    template <typename T>
    Tensor<T> Tensor<T>::transpose(const size_t dim0, const size_t dim1) const {
        std::vector<size_t> new_shape = m_shape;
        std::vector<size_t> new_strides = m_strides;
        size_t temp = new_shape[dim0];
        new_shape[dim0] = new_shape[dim1];
        new_shape[dim1] = temp;
        temp = new_strides[dim0];
        new_strides[dim0] = new_strides[dim1];
        new_strides[dim1] = temp;

        return Tensor(std::move(new_shape), std::move(new_strides), m_offset, m_storage);
    }

    template <typename T>
    Tensor<T> Tensor<T>::permute(const std::vector<int64_t>& axes) const {
        size_t n_dim = m_shape.size();
        std::cout << axes.size() << " " << n_dim << std::endl;
        if (axes.size() != n_dim) {
            throw std::runtime_error("permute() axes list size must match shape list size.");
        }
        std::vector<size_t> new_shape = std::vector<size_t>(n_dim);
        std::vector<size_t> new_strides = std::vector<size_t>(n_dim);
        for (size_t target_ax = 0; target_ax < n_dim; ++target_ax) {
            int64_t src_ax = axes[target_ax];
            if (src_ax < 0) {
                src_ax = n_dim + src_ax;
            }
            new_shape[target_ax] = m_shape[src_ax];
            new_strides[target_ax] = m_strides[src_ax];
        }
        return Tensor(std::move(new_shape), std::move(new_strides), m_offset, m_storage);
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
        
        if (this->is_contiguous()) { // cannot reshape a non-contiguous tensor.
            return Tensor(std::move(new_shape), std::move(new_strides), m_offset, m_storage, m_op);
        }

        Tensor reshaped_tensor = this->contiguous(); // contiguous is shape-sensitive (if sliced then only slice is made contiguous)
        reshaped_tensor.m_shape = std::move(new_shape);
        reshaped_tensor.m_strides = std::move(new_strides);

        return reshaped_tensor;
    }

    template <typename T>
    Tensor<T> Tensor<T>::broadcast_to(const std::vector<size_t>& target_shape) const {
        int64_t n_dim_orig = static_cast<int64_t>(m_shape.size());
        int64_t n_dim_target = static_cast<int64_t>(target_shape.size());
        std::vector<size_t> new_shape = std::vector<size_t>(n_dim_target);
        std::vector<size_t> new_strides = std::vector<size_t>(n_dim_target);

        // align to the right
        for (int64_t i = 0; i < n_dim_orig; ++i) {
            new_shape[i + (n_dim_target - n_dim_orig)] = m_shape[i];
            new_strides[i + (n_dim_target - n_dim_orig)] = m_strides[i];
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
                throw std::runtime_error("Violated broadcasting rules in .broadcast_to().");
            }
        }

        return Tensor(std::move(new_shape), std::move(new_strides), m_offset, m_storage);
    }
}