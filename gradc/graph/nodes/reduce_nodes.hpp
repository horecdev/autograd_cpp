#pragma once

#include "../node.hpp"
#include "../../core/tensor.hpp"
#include "../../backend/dispatcher.hpp"

namespace gradc {

    template <typename T>
    class SumNode : public Node<T> {
        private:
            Tensor<T> m_parent;
            ReductionMetadata m_reduction_metadata;
        public:
            SumNode(Tensor<T> parent, ReductionMetadata reduction_metadata) : m_parent(parent), m_reduction_metadata(reduction_metadata) {}

            Tensor<T> realize() override {
                m_parent.realize();
                return apply_reduction_operation(m_parent, m_reduction_metadata, T(), [](T a, T b){return a + b;});
            }

            void backward(const Tensor<T>& out_grad) override {
                if (m_parent.requires_grad()) {
                    m_parent.accumulate_grad(Tensor<T>(m_parent.shape(), m_reduction_metadata.temp_strides, 0, out_grad._get_storage(), false));
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class MeanNode : public Node<T> {
        private:
            Tensor<T> m_parent;
            ReductionMetadata m_reduction_metadata;
        public:
            MeanNode(Tensor<T> parent, ReductionMetadata reduction_metadata) : m_parent(parent), m_reduction_metadata(reduction_metadata) {}

            Tensor<T> realize() override {
                m_parent.realize();
                Tensor<T> summed = apply_reduction_operation(m_parent, m_reduction_metadata, T(), [](T a, T b){return a + b;});
                apply_in_place(summed, Tensor<T>(static_cast<T>(m_reduction_metadata.reduced_vol), summed.device()), [](T& a, T b){a /= b;});
                return summed;
            }

            void backward(const Tensor<T>& out_grad) override {
                if (m_parent.requires_grad()) {
                    Tensor<T> divided_grad = apply_out_of_place(out_grad, Tensor<T>(static_cast<T>(m_reduction_metadata.reduced_vol), m_parent.device()), out_grad.shape(), [](T a, T b){return a / b;});
                    Tensor<T> strided_mean_grad = Tensor<T>(m_parent.shape(), m_reduction_metadata.temp_strides, 0, divided_grad._get_storage(), false);
                    m_parent.accumulate_grad(strided_mean_grad);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };
}