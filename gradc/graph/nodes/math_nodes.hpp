#pragma once

#include "../node.hpp"
#include "../../core/tensor.hpp"
#include "../../backend/dispatcher.hpp"

namespace gradc {
    template <typename T>
    class AddNode : public Node<T> {
        private:
            Tensor<T> m_left;
            Tensor<T> m_right;
            std::vector<int64_t> m_target_shape;
        public:
            AddNode<T>(Tensor<T> left, Tensor<T> right, std::vector<int64_t> target_shape) : m_left(std::move(left)), m_right(std::move(right)), m_target_shape(std::move(target_shape)) {}
            
            Tensor<T> realize() override {
                m_left.realize();
                m_right.realize();

                if (m_left.is_exclusive() && m_left.shape() == m_target_shape) { // inside addnode m_left storage is used solely for producing a result. 
                // If nobody else uses the m_left EVER, then instead of using new memory, can just edit it and return.
                    apply_in_place(m_left, m_right, [](T& a, T b){a += b;});
                    return m_left;
                }
                else if (m_right.is_exclusive() && m_right.shape() == m_target_shape) {
                    apply_in_place(m_right, m_left, [](T& a, T b){a += b;});
                    return m_right;
                }

                return apply_out_of_place(m_left, m_right, m_target_shape, [](T a, T b) {return a + b;});
            }

            void backward(const Tensor<T>& out_grad) override { // CPU child can only have CPU parents (enforced outside of graph nodes)
                if (m_left.requires_grad()) {
                    m_left.accumulate_grad(unbroadcast_grad(out_grad, m_left.shape()));
                }
                if (m_right.requires_grad()) {
                    m_right.accumulate_grad(unbroadcast_grad(out_grad, m_right.shape()));
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_left._get_state_base(), m_right._get_state_base()};
            }
    };

    template <typename T>
    class MulNode : public Node<T> {
        private:
            Tensor<T> m_left;
            Tensor<T> m_right;
            std::vector<int64_t> m_target_shape;
        public:
            MulNode<T>(Tensor<T> left, Tensor<T> right, std::vector<int64_t> target_shape) : m_left(std::move(left)), m_right(std::move(right)), m_target_shape(std::move(target_shape)) {}
            
            Tensor<T> realize() override {
                m_left.realize();
                m_right.realize();

                if (m_left.is_exclusive() &&  m_left.shape() == m_target_shape && m_right.requires_grad() == false) {
                    apply_in_place(m_left, m_right, [](T& a, T b){a *= b;});
                    return m_left;
                }
                else if (m_right.is_exclusive() && m_right.shape() == m_target_shape && m_left.requires_grad() == false) {
                    apply_in_place(m_right, m_left, [](T& a, T b){a *= b;});
                    return m_right;
                }

                return apply_out_of_place(m_left, m_right, m_target_shape, [](T a, T b) {return a * b;});
            }

            void backward(const Tensor<T>& out_grad) override {
                if (m_left.requires_grad()) {
                    Tensor<T> raw_left_grad = apply_out_of_place(out_grad, m_right, m_target_shape, [](T a, T b){return a * b;});
                    m_left.accumulate_grad(unbroadcast_grad(raw_left_grad, m_left.shape()));
                }
                if (m_right.requires_grad()) {
                    Tensor<T> raw_right_grad = apply_out_of_place(out_grad, m_left, m_target_shape, [](T a, T b){return a * b;});
                    m_right.accumulate_grad(unbroadcast_grad(raw_right_grad, m_right.shape()));
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_left._get_state_base(), m_right._get_state_base()};
            }
    };

} 