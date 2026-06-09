#pragma once

#include "../tensor.hpp"
#include "../graph.hpp"
#include "tensor_utils.hpp"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gradc {
    template <typename T>
    Tensor<T> operator+(Tensor<T> left, Tensor<T> right) {
        std::vector<size_t> target_shape;
        if (left.m_shape != right.m_shape) {
            target_shape = infer_broadcast(left.m_shape, right.m_shape); // crashes if incompatible on the graph-building stage.
        }
        else {
            target_shape = left.m_shape;
        }

        bool requires_grad = left.m_requires_grad && right.m_requires_grad;
        Tensor<T> new_tensor = Tensor<T>(target_shape, requires_grad, lazy);
        new_tensor.m_state->m_op = std::make_unique<AddNode<T>>(std::move(left), std::move(right), std::move(target_shape));
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
        new_tensor.m_state->m_op = std::make_unique<MulNode<T>>(std::move(left), std::move(right), std::move(target_shape));
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
        TensorState<T> old_tensor_state = TensorState(main.m_state->m_storage, std::move(main.m_state->m_realize_op));
        Tensor<T> old_main = Tensor<T>(main.m_shape, main.m_strides, main.m_offset, std::move(old_tensor_state), main.m_requires_grad);
        main.m_state->m_realize_op = std::make_unique<InPlaceAddNode<T>>(std::move(old_main), std::move(other));
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
        
        TensorState<T> old_tensor_state = TensorState(main.m_state->m_storage, std::move(main.m_state->m_realize_op));
        Tensor<T> old_main = Tensor<T>(main.m_shape, main.m_strides, main.m_offset, std::move(old_tensor_state), main.m_requires_grad);
        main.m_state->m_realize_op = std::make_unique<InPlaceMulNode<T>>(std::move(old_main), std::move(other));
        return main;
    }


}