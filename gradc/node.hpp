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
            virtual Tensor<T> realize() const = 0;

            virtual void backward(const Tensor<T>& out_grad) = 0;

            virtual std::vector<TensorStateBase*> get_input_states() = 0;

            virtual ~Node() {}
    };
}