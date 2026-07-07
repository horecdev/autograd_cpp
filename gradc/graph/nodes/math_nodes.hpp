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

                Device target_device = m_left.device();

                if (m_left.is_exclusive() && m_left.shape() == m_target_shape) { // inside addnode m_left storage is used solely for producing a result. 
                    // If nobody else uses the m_left EVER, then instead of using new memory, can just edit it and return.
                    dispatch(target_device, BinaryOpInPlace::Add, m_left, m_right);
                    return m_left;
                }
                else if (m_right.is_exclusive() && m_right.shape() == m_target_shape) {
                    dispatch(target_device, BinaryOpInPlace::Add, m_right, m_left);
                    return m_right;
                }

                Tensor<T> result = Tensor<T>(m_target_shape, target_device, uninitialized);
                dispatch(target_device, BinaryOp::Add, result, m_left, m_right);
                return result;
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

                Device target_device = m_left.device();

                if (m_left.is_exclusive() &&  m_left.shape() == m_target_shape && m_right.requires_grad() == false) {
                    dispatch(target_device, BinaryOpInPlace::Add, m_left, m_right);
                    return m_left;
                }
                else if (m_right.is_exclusive() && m_right.shape() == m_target_shape && m_left.requires_grad() == false) {
                    dispatch(target_device, BinaryOpInPlace::Add, m_right, m_left);
                    return m_right;
                }

                Tensor<T> result = Tensor<T>(m_target_shape, target_device, uninitialized);
                dispatch(target_device, BinaryOp::Mul, result, m_left, m_right);
                return result;
            }

            void backward(const Tensor<T>& out_grad) override {
                Device target_device = m_left.device();

                if (m_left.requires_grad()) {
                    Tensor<T> raw_left_grad = Tensor<T>(out_grad.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Mul, raw_left_grad, out_grad, m_right);
                    m_left.accumulate_grad(unbroadcast_grad(raw_left_grad, m_left.shape()));
                }
                if (m_right.requires_grad()) {
                    Tensor<T> raw_right_grad = Tensor<T>(out_grad.shape(), target_device, uninitialized);
                    dispatch(target_device, BinaryOp::Mul, raw_right_grad, out_grad, m_left);
                    m_right.accumulate_grad(unbroadcast_grad(raw_right_grad, m_right.shape()));
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_left._get_state_base(), m_right._get_state_base()};
            }
    };

} 