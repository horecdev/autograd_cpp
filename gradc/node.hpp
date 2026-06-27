#pragma once

#include <memory>
#include <vector>

namespace gradc {

    struct TensorStateBase;

    template <typename T>
    struct TensorState;

    template <typename T>
    class Tensor;

    template <typename T>
    class Node {

        public:
            virtual Tensor<T> realize() = 0;

            virtual void backward(const Tensor<T>& out_grad) = 0;

            virtual std::vector<TensorStateBase*> get_input_states() = 0;

            virtual ~Node() {}

        protected: 
            template <typename U> // every node inheriting from Node<T> has access to this function
            // Allows: all Nodes access to all tensors
            static std::shared_ptr<TensorState<U>>& get_state(Tensor<U>& t) {
                return t.m_state;
            }
    };
}