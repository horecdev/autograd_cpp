#pragma once
#include "../tensor.hpp"

namespace gradc {
    template <typename T> // gotta redefine the exact type [since its outside template <typename T> of Tensor class]
    template <typename Func>
    Tensor<T>& Tensor<T>::apply_in_place(const Tensor<T>& other, Func op) { 
        // Idea behind all the variables - track right variables instead of copying vectors on the heap inside copied Tensors.
        // (you would have to either create a tensor or copy it to hold a variable (or worse - write main logic twice) - why not ONLY create if needed?)
        const std::vector<size_t>* other_strides; // promise not to modify data, not reassign the pointer
        size_t other_offset;
        std::shared_ptr<Storage<T>> other_storage;

        Tensor broad_other; // inside Tensor<T>:: space, "Tensor" is replaced with "Tensor<T>" during compilation
        if (m_shape == other.m_shape) {
            other_strides = &other.m_strides;
            other_offset = other.m_offset;
            other_storage = other.m_storage; // shared ptr
        }
        else {
            broad_other = other.broadcast_to(m_shape);

            other_strides = &broad_other.m_strides;
            other_offset = broad_other.m_offset;
            other_storage = broad_other.m_storage; // all stuff that increments shared_ptr dies at the end of function.
        }

        size_t n_dims = m_shape.size();
        std::vector<size_t> odometer(n_dims, 0);
        while (odometer[0] < m_shape[0]) {
            size_t this_strided_idx = m_offset; 
            size_t other_strided_idx = other_offset;

            for (size_t i = 0; i < n_dims; ++i) {
                this_strided_idx += odometer[i] * m_strides[i];
                other_strided_idx += odometer[i] * (*other_strides)[i];
            }
            op(((*m_storage).m_data)[this_strided_idx], ((*other_storage).m_data)[other_strided_idx]);
            ++odometer[n_dims - 1];
            size_t i = n_dims - 1;
            while ((odometer[i] == m_shape[i]) && i > 0) {
                odometer[i] = 0;
                ++odometer[i - 1];
                --i;
            }
        }
        ++m_storage->m_version;
        return *this;
    }

    template <typename T>
    template <typename Func>
    Tensor<T> Tensor<T>::apply_out_of_place(const Tensor<T>& other, Func op) const {
        const std::vector<size_t>* this_strides = &m_strides;
        const std::vector<size_t>* other_strides = &other.m_strides;
        size_t this_offset = m_offset;
        size_t other_offset = other.m_offset; 
        std::shared_ptr<Storage<T>> this_storage = m_storage;
        std::shared_ptr<Storage<T>> other_storage = other.m_storage;

        Tensor broad_this;
        Tensor broad_other;

        std::vector<size_t> target_shape; 

        if (m_shape != other.m_shape) {
            target_shape = infer_broadcast(m_shape, other.m_shape); // crashes if incompatible

            if (m_shape != target_shape) {
                broad_this = this->broadcast_to(target_shape);

                this_strides = &broad_this.m_strides;
                this_offset = broad_this.m_offset;
                this_storage = broad_this.m_storage;
            }
            if (other.m_shape != target_shape) {
                broad_other = other.broadcast_to(target_shape);

                other_strides = &broad_other.m_strides;
                other_offset = broad_other.m_offset;
                other_storage = broad_other.m_storage;
            }
        }
        else {
            target_shape = m_shape;
        }

        Tensor result = Tensor(target_shape);

        size_t n_dims = target_shape.size();
        std::vector<size_t> odometer(n_dims, 0);
        size_t contiguous_idx = 0;
        while (odometer[0] < target_shape[0]) {
            size_t this_strided_idx = this_offset; 
            size_t other_strided_idx = other_offset;

            for (size_t i = 0; i < n_dims; ++i) {
                this_strided_idx += odometer[i] * (*this_strides)[i];
                other_strided_idx += odometer[i] * (*other_strides)[i];
            }
            ((*result.m_storage).m_data)[contiguous_idx] = op(((*this_storage).m_data)[this_strided_idx], ((*other_storage).m_data)[other_strided_idx]); // copied straight into CPU registers from RAM
            ++contiguous_idx;
            ++odometer[n_dims - 1];
            size_t i = n_dims - 1;
            while ((odometer[i] == target_shape[i]) && i > 0) {
                odometer[i] = 0;
                ++odometer[i - 1];
                --i;
            }
        }

        return result;
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator+=(const Tensor<T>& other) {
        return apply_in_place(other, [](T &a, T b){a += b;});
    }
    
    template <typename T>
    Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
        return apply_out_of_place(other, [](T a, T b) {return a + b;});
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator-=(const Tensor<T>& other) {
        return apply_in_place(other, [](T &a, T b){a -= b;});
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator-(const Tensor<T>& other) const {
        return apply_out_of_place(other, [](T a, T b) {return a - b;});
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator*=(const Tensor<T>& other) {
        return apply_in_place(other, [](T &a, T b){a *= b;});
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator*(const Tensor<T>& other) const {
        return apply_out_of_place(other, [](T a, T b) {return a * b;});
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator/=(const Tensor<T>& other) {
        return apply_in_place(other, [](T &a, T b){a /= b;});
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator/(const Tensor<T>& other) const {
        return apply_out_of_place(other, [](T a, T b) {return a / b;});
    }
}