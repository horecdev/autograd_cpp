#pragma once
#include "tensor.hpp" // IWYU pragma: keep
#include "node.hpp"

namespace gradc {
    // gotta redefine the exact type [since its outside template <typename T> of Tensor class]
    template <typename T, typename Func>
    Tensor<T>& apply_in_place(Tensor<T>& left, const Tensor<T>& right, Func op) { 
        // Idea behind all the variables - track right variables instead of copying vectors on the heap inside copied Tensors.
        // (you would have to either create a tensor or copy it to hold a variable (or worse - write main logic twice) - why not ONLY create if needed?)
        const std::vector<size_t>* right_strides; // promise not to modify data, not reassign the pointer
        size_t right_offset;
        std::shared_ptr<Storage<T>> right_storage;

        Tensor<T> broad_right; // inside Tensor<T>:: space, "Tensor" is replaced with "Tensor<T>" during compilation
        if (left.m_shape == right.m_shape) {
            right_strides = &right.m_strides;
            right_offset = right.m_offset;
            right_storage = right.m_storage; // shared ptr
        }
        else {
            broad_right = right.broadcast_to(left.m_shape);

            right_strides = &broad_right.m_strides;
            right_offset = broad_right.m_offset;
            right_storage = broad_right.m_storage; // all stuff that increments shared_ptr dies at the end of function.
        }

        size_t n_dims = left.m_shape.size();
        std::vector<size_t> odometer(n_dims, 0);
        while (odometer[0] < left.m_shape[0]) {
            size_t left_strided_idx = left.m_offset; 
            size_t right_strided_idx = right_offset;

            for (size_t i = 0; i < n_dims; ++i) {
                left_strided_idx += odometer[i] * left.m_strides[i];
                right_strided_idx += odometer[i] * (*right_strides)[i];
            }
            op(((*left.m_storage).m_data)[left_strided_idx], ((*right_storage).m_data)[right_strided_idx]);
            ++odometer[n_dims - 1];
            size_t i = n_dims - 1;
            while ((odometer[i] == left.m_shape[i]) && i > 0) {
                odometer[i] = 0;
                ++odometer[i - 1];
                --i;
            }
        }
        ++left.m_storage->m_version;
        return left;
    }

    template <typename T, typename Func>
    Tensor<T> apply_out_of_place(const Tensor<T>& left, const Tensor<T>& right, Func op) {
        const std::vector<size_t>* left_strides = &left.m_strides;
        const std::vector<size_t>* right_strides = &right.m_strides;
        size_t left_offset = left.m_offset;
        size_t right_offset = right.m_offset; 
        std::shared_ptr<Storage<T>> left_storage = left.m_storage;
        std::shared_ptr<Storage<T>> right_storage = right.m_storage;

        Tensor<T> broad_right;
        Tensor<T> broad_left;

        std::vector<size_t> target_shape; 

        if (left.m_shape != right.m_shape) {
            target_shape = infer_broadcast(left.m_shape, right.m_shape); // crashes if incompatible

            if (left.m_shape != target_shape) {
                broad_left = left.broadcast_to(target_shape);

                left_strides = &broad_left.m_strides;
                left_offset = broad_left.m_offset;
                left_storage = broad_left.m_storage;
            }
            if (right.m_shape != target_shape) {
                broad_right = right.broadcast_to(target_shape);

                right_strides = &broad_right.m_strides;
                right_offset = broad_right.m_offset;
                right_storage = broad_right.m_storage;
            }
        }
        else {
            target_shape = left.m_shape;
        }

        Tensor<T> result = Tensor<T>(target_shape);

        size_t n_dims = target_shape.size();
        std::vector<size_t> odometer(n_dims, 0);
        size_t contiguous_idx = 0;
        while (odometer[0] < target_shape[0]) {
            size_t left_strided_idx = left_offset; 
            size_t right_strided_idx = right_offset;

            for (size_t i = 0; i < n_dims; ++i) {
                left_strided_idx += odometer[i] * (*left_strides)[i];
                right_strided_idx += odometer[i] * (*right_strides)[i];
            }
            ((*result.m_storage).m_data)[contiguous_idx] = op(((*left_storage).m_data)[left_strided_idx], ((*right_storage).m_data)[right_strided_idx]); // copied straight into CPU registers from RAM
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
    class AddNode : public Node<T> {
        private:
            Tensor<T> m_left;
            Tensor<T> m_right;
        public:
            AddNode<T>(Tensor<T> left, Tensor<T> right) : m_left(std::move(left)), m_right(std::move(right)) {}
            
            Tensor<T> realize() {
                m_left.realize();
                m_right.realize();
                return apply_out_of_place(m_left, m_right, [](T a, T b) {return a + b;});
            }

            void backward() {}
    };

    template <typename T>
    class MulNode : public Node<T> {
        private:
            Tensor<T> m_left;
            Tensor<T> m_right;
        public:
            MulNode<T>(Tensor<T> left, Tensor<T> right) : m_left(std::move(left)), m_right(std::move(right)) {}
            
            Tensor<T> realize() {
                m_left.realize();
                m_right.realize();
                return apply_out_of_place(m_left, m_right, [](T a, T b) {return a * b;});
            }

            void backward() {}
    };

    template <typename T>
    class CloneNode : public Node<T> {

    };

    template <typename T>
    class ReshapeNode : public Node<T> {

    };
}