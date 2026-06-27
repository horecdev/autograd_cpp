#pragma once

#include "../tensor.hpp"
#include "../graph.hpp"
#include "tensor_utils.hpp"

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace gradc {
    template <typename T, typename U>
    auto operator+(Tensor<T> left, Tensor<U> right) {
        using PromotedT = std::common_type_t<T, U>;

        auto [p_left, p_right] = promote_to_common(std::move(left), std::move(right));

        std::vector<int64_t> target_shape;
        if (p_left.m_shape != p_right.m_shape) {
            target_shape = infer_broadcast(p_left.m_shape, p_right.m_shape); // crashes if incompatible on the graph-building stage.
        }
        else {
            target_shape = p_left.m_shape;
        }

        bool requires_grad = p_left.m_requires_grad || p_right.m_requires_grad;
        Tensor<PromotedT> new_tensor = Tensor<PromotedT>(target_shape, requires_grad, lazy);
        new_tensor.m_state->m_creation_op = std::make_unique<AddNode<PromotedT>>(std::move(p_left), std::move(p_right), std::move(target_shape));
        return new_tensor;
    }

    template <typename T, typename U>
    requires std::is_arithmetic_v<U>
    auto operator+(Tensor<T> left, U right_val) { // Tensor + scalar
        return std::move(left) + Tensor<U>(right_val);
    }

    template <typename T, typename U>
    requires std::is_arithmetic_v<T>
    auto operator+(T right_val, Tensor<U> left) { // scalar + Tensor
        return Tensor<T>(right_val) + std::move(left);
    }

    template <typename T, typename U>
    auto operator*(Tensor<T> left, Tensor<U> right) {
        using PromotedT = std::common_type_t<T, U>;

        auto [p_left, p_right] = promote_to_common(std::move(left), std::move(right));

        std::vector<int64_t> target_shape;
        if (p_left.m_shape != p_right.m_shape) {
            target_shape = infer_broadcast(p_left.m_shape, p_right.m_shape); // crashes if incompatible on the graph-building stage.
        }
        else {
            target_shape = p_left.m_shape;
        }

        bool requires_grad = p_left.m_requires_grad || p_right.m_requires_grad;
        Tensor<PromotedT> new_tensor = Tensor<PromotedT>(target_shape, requires_grad, lazy);
        new_tensor.m_state->m_creation_op = std::make_unique<MulNode<PromotedT>>(std::move(p_left), std::move(p_right), std::move(target_shape));
        return new_tensor;
    }

    template <typename T, typename U>
    requires std::is_arithmetic_v<U>
    auto operator*(Tensor<T> left, U right_val) { // Tensor + scalar
        return std::move(left) * Tensor<U>(right_val);
    }

    template <typename T, typename U>
    requires std::is_arithmetic_v<T>
    auto operator*(T right_val, Tensor<U> left) { // scalar + Tensor
        return Tensor<T>(right_val) * std::move(left);
    }

    template <typename T, typename U>
    Tensor<T>& operator+=(Tensor<T>& main, Tensor<U> other) {
        using PromotedT = std::common_type_t<T, U>;
        Tensor<PromotedT> p_other;

        static_assert(std::is_same_v<T, PromotedT>, "FATAL: Cannot promote type of main tensor during in-place operation.");

        if constexpr (!std::is_same_v<U, PromotedT>) {
            p_other = other.template cast<PromotedT>();
        }
        else {
            p_other = std::move(other);
        }

        if (main.m_requires_grad && main.m_state->m_creation_op == nullptr) {
            throw std::runtime_error("Cannot mutate leaf tensor that requires gradients in-place.");
        }

        if (main.m_shape != p_other.m_shape) { // Validates on the graph building
            if (!can_broadcast(other.m_shape, main.m_shape)) {
                throw std::runtime_error("Could not broadcast RHS to match LHS during in-place operation.");
            }
        }

        if (main.m_state.use_count() == 1 && main.m_state->m_storage.use_count() == 1) {
            // going with ELSE and logic below would edit TensorState across already saved graph pieces
            // say: y = a + b, then a += 5. Without ref counting the cached a inside y is edited, its m_op is now InPlaceAddNode. 
            // It executes even though it was done later.
            std::shared_ptr<TensorState<T>> old_tensor_state = std::make_shared<TensorState<T>>(main.m_state->m_storage, std::move(main.m_state->m_creation_op));
            Tensor<T> old_main = Tensor<T>(main.m_shape, main.m_strides, main.m_offset, std::move(old_tensor_state), main.m_requires_grad);
            main.m_state->m_creation_op = std::make_unique<InPlaceAddNode<T>>(std::move(old_main), std::move(p_other));
            main.m_requires_grad = main.m_requires_grad || p_other.m_requires_grad;
            return main;
        }

        else {
            // say a = b = c. 3 lightview tensors, refcount of TensorState is 3
            // you do a += 5. a gets a new tensor state, and keeps the old a lightview tensor inside AddNode. 
            // The refcount doesnt change. State #2 has 1, State #1 has 3. A is the same tensor, but with new state and a node.
            bool requires_grad = main.m_requires_grad || p_other.m_requires_grad;
            std::shared_ptr<TensorState<T>> new_state = std::make_shared<TensorState<T>>();
            new_state->m_creation_op = std::make_unique<AddNode<T>>(main, p_other, main.m_shape);
            main.m_state = std::move(new_state);
            main.m_requires_grad = requires_grad;
            return main;
        }
    }

    template <typename T, typename U>
    requires std::is_arithmetic_v<U>
    Tensor<T>& operator+=(Tensor<T>& main, U other_val) {
        return main += Tensor<U>(other_val);
    }

    template <typename T, typename U>
    Tensor<T>& operator*=(Tensor<T>& main, const Tensor<U> other) {
        using PromotedT = std::common_type_t<T, U>;
        Tensor<PromotedT> p_other;

        static_assert(std::is_same_v<T, PromotedT>, "FATAL: Cannot promote type of main tensor during in-place operation.");
        
        if constexpr (!std::is_same_v<U, PromotedT>) {
            p_other = other.template cast<PromotedT>();
        }
        else {
            p_other = std::move(other);
        }

        if (main.m_requires_grad && main.m_state->m_creation_op == nullptr) { // we are using toposort to get rid of temporary results using leaf nodes as a stop-point, and in-place math literally moves it inside the graph.
            throw std::runtime_error("Cannot mutate leaf tensor that requires gradients in-place.");
        }

        if (main.m_shape != other.m_shape) { // Validates on the graph building
            if (!can_broadcast(other.m_shape, main.m_shape)) {
                throw std::runtime_error("Could not broadcast RHS to match LHS during in-place operation.");
            }
        }
        
        if (main.m_state.use_count() == 1 && main.m_state->m_storage.use_count() == 1) {
            // going with ELSE and logic below would edit TensorState across already saved graph pieces
            // say: y = a + b, then a += 5. Without ref counting the cached a inside y is edited, its m_op is now InPlaceAddNode. 
            // It executes even though it was done later.
            std::shared_ptr<TensorState<T>> old_tensor_state = std::make_shared<TensorState<T>>(main.m_state->m_storage, std::move(main.m_state->m_creation_op));
            Tensor<T> old_main = Tensor<T>(main.m_shape, main.m_strides, main.m_offset, std::move(old_tensor_state), main.m_requires_grad);
            main.m_state->m_creation_op = std::make_unique<InPlaceMulNode<T>>(std::move(old_main), std::move(p_other));
            main.m_requires_grad = main.m_requires_grad || p_other.m_requires_grad;
            return main;
        }

        else {
            bool requires_grad = main.m_requires_grad || p_other.m_requires_grad;
            std::shared_ptr<TensorState<T>> new_state = std::make_shared<TensorState<T>>();
            new_state->m_creation_op = std::make_unique<MulNode<T>>(main, p_other, main.m_shape);
            main.m_state = std::move(new_state);
            main.m_requires_grad = requires_grad;
            return main;
        }
    }

    template <typename T, typename U>
    requires std::is_arithmetic_v<U>
    Tensor<T>& operator*=(Tensor<T>& main, U other_val) {
        return main *= Tensor<U>(other_val);
    }
}