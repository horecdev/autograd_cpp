#pragma once

#include "../node.hpp"
#include "../../core/tensor.hpp"
#include "../../backend/dispatcher.hpp"

namespace gradc {

    template <typename T>
    class ReLUNode : public Node<T> {
        private:
            Tensor<T> m_parent;
        public:
            ReLUNode(Tensor<T> parent) : m_parent(parent) {}

            Tensor<T> realize() override {
                m_parent.realize();
                return apply_unary_out_of_place(m_parent, [](T a){return a > 0 ? a : 0;});
            }

            void backward(const Tensor<T>& out_grad) override {
                if (m_parent.requires_grad()) {
                    Tensor<T> relu_grad = apply_out_of_place(out_grad, m_parent, m_parent.shape(), [](T a, T b){return b > 0 ? a : 0;});
                    m_parent.accumulate_grad(relu_grad);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

}