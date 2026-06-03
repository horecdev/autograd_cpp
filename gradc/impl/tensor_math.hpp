#pragma once
#include "../tensor.hpp"
#include "../graph.hpp"
#include "tensor_utils.hpp"
#include <stdexcept>

namespace gradc {
    // gotta redefine the exact type [since its outside template <typename T> of Tensor class]
    template <typename T, typename Func>
    void apply_in_place(Tensor<T>& left, const Tensor<T>& right, Func op) { 
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
    }

    template <typename T, typename Func>
    Tensor<T> apply_out_of_place(const Tensor<T>& left, const Tensor<T>& right, const std::vector<size_t>& target_shape, Func op) {
        const std::vector<size_t>* left_strides = &left.m_strides;
        const std::vector<size_t>* right_strides = &right.m_strides;
        size_t left_offset = left.m_offset;
        size_t right_offset = right.m_offset; 
        std::shared_ptr<Storage<T>> left_storage = left.m_storage;
        std::shared_ptr<Storage<T>> right_storage = right.m_storage;

        Tensor<T> broad_right;
        Tensor<T> broad_left;

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
    Tensor<T> operator+(Tensor<T> left, Tensor<T> right) {
        std::vector<size_t> target_shape;
        if (left.m_shape != right.m_shape) { // TODO: ADD FRIEND TO THESE FUNCTIONS SO THEY CAN ACCES PRIVATE VARIABLES
            target_shape = infer_broadcast(left.m_shape, right.m_shape); // crashes if incompatible on the graph-building stage.
        }
        else {
            target_shape = left.m_shape;
        }

        bool requires_grad = left.m_requires_grad && right.m_requires_grad;
        Tensor<T> new_tensor = Tensor<T>(target_shape, requires_grad, lazy);
        new_tensor.m_op = std::make_shared<AddNode<T>>(std::move(left), std::move(right), std::move(target_shape));
        return new_tensor;
    }

    template <typename T>
    Tensor<T> operator*(Tensor<T> left, Tensor<T> right) {
        std::vector<size_t> target_shape;
        if (left.m_shape != right.m_shape) {
            target_shape = infer_broadcast(left.m_shape, right.m_shape); // crashes if incompatible on the graph-building stage.
        }
        else {
            target_shape = left.m_shape;
        }

        bool requires_grad = left.m_requires_grad && right.m_requires_grad;
        Tensor<T> new_tensor = Tensor<T>(target_shape, requires_grad, lazy);
        new_tensor.m_op = std::make_shared<MulNode<T>>(std::move(left), std::move(right), std::move(target_shape));
        return new_tensor;
    }

    template <typename T>
    Tensor<T>& operator+=(Tensor<T>& main, Tensor<T> other) {
        if (main.m_requires_grad && main.m_op == nullptr) {
            throw std::runtime_error("Cannot mutate leaf tensor that requires gradients in-place.");
        }

        std::vector<size_t> common_shape;
        if (main.m_shape != other.m_shape) { // Validates on the graph building
            if (!can_broadcast(other.m_shape, main.m_shape)) {
                throw std::runtime_error("Could not broadcast RHS to match LHS during in-place operation.");
            }
        }
        
        Tensor<T> old_main = main; // copy history up to this point
        main.m_op = std::make_shared<InPlaceAddNode<T>>(std::move(old_main), std::move(other));
        
        return main;
    }

    template <typename T>
    Tensor<T>& operator*=(Tensor<T>& main, Tensor<T> other) {
        if (main.m_requires_grad && main.m_op == nullptr) { // we are using toposort to get rid of temporary results using leaf nodes as a stop-point, and in-place math literally moves it inside the graph.
            throw std::runtime_error("Cannot mutate leaf tensor that requires gradients in-place.");
        }

        std::vector<size_t> common_shape;
        if (main.m_shape != other.m_shape) { // Validates on the graph building
            if (!can_broadcast(other.m_shape, main.m_shape)) {
                throw std::runtime_error("Could not broadcast RHS to match LHS during in-place operation.");
            }
        }
        
        Tensor<T> old_main = main; // copy history up to this point
        main.m_op = std::make_shared<InPlaceMulNode<T>>(std::move(old_main), std::move(other));
        
        return main;
    }


}