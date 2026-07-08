#pragma once

#include "../../backend/dispatcher.hpp"
#include "../../core/detail/tensor_lob_alloc.hpp"
#include "../../core/detail/tensor_lob_view.hpp"
#include "../../core/tensor.hpp"
#include "../node.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace gradc {

    template <typename T>
    class CloneNode : public Node<T> {
        private:
            Tensor<T> m_parent;
        public:
            CloneNode(Tensor<T> parent) : m_parent(std::move(parent)) {}

            Tensor<T> realize() override {
                m_parent.realize();
                Tensor<T> result = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                dispatch(m_parent.device(), UnaryOp::Identity, result, m_parent);
                return result;
            }

            void backward(const Tensor<T>& out_grad) override {
                if (m_parent.requires_grad()) {
                    m_parent.accumulate_grad(out_grad); // literally just copy over
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class ContiguousNode: public Node<T> {
        private: 
            Tensor<T> m_parent;
        public:
            ContiguousNode(Tensor<T> parent) : m_parent(std::move(parent)) {}

            Tensor<T> realize() override {
                m_parent.realize();
                Tensor<T> result = Tensor<T>(m_parent.shape(), m_parent.device(), uninitialized);
                dispatch(m_parent.device(), UnaryOp::Identity, result, m_parent);
                return result;
            }

            void backward(const Tensor<T>& out_grad) override {
                if (m_parent.requires_grad()) {
                    m_parent.accumulate_grad(out_grad);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    }; 

    template <typename InT, typename OutT>
    class CastNode : public Node<OutT> {
        private: 
            Tensor<InT> m_parent;
        public:
            CastNode(Tensor<InT> parent) : m_parent(std::move(parent)) {}

            Tensor<OutT> realize() override {
                m_parent.realize();
                Tensor<OutT> result = Tensor<OutT>(m_parent.shape(), m_parent.device(), uninitialized); 
                dispatch_cast(m_parent.device(), result, m_parent);
                return result;
            }

            void backward(const Tensor<OutT>& out_grad) override {
                if (m_parent.requires_grad()) {
                    Tensor<InT> cast_grad = Tensor<InT>(out_grad.shape(), out_grad.device(), uninitialized);
                    dispatch_cast(out_grad.device(), cast_grad, out_grad);
                    m_parent.accumulate_grad(cast_grad);
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent._get_state_base()};
            }
    };

    template <typename T>
    class ConcatNode : public Node<T> {
        private:
            std::vector<Tensor<T>> m_parents_list;
            int64_t m_concat_dim;
            std::vector<int64_t> m_final_shape;
        public:
            ConcatNode(std::vector<Tensor<T>> parents_list, int64_t concat_dim, std::vector<int64_t> final_shape) : m_parents_list(std::move(parents_list)), m_concat_dim(concat_dim), m_final_shape(std::move(final_shape)) {}

            Tensor<T> realize() override {
                for (Tensor<T>& parent : m_parents_list) {
                    parent.realize();
                }

                return lobotomized_concat_alloc(m_parents_list, m_concat_dim, m_final_shape);
            }

            void backward(const Tensor<T>& out_grad) override {
                int64_t n_dim = std::ssize(m_final_shape);
                int64_t concat_dim_progress = 0;
                for (Tensor<T>& parent : m_parents_list) {
                    if (parent.requires_grad()) {
                        std::vector<IndexDesc> descriptors;
                        descriptors.reserve(n_dim);

                        for (int64_t i = 0; i < n_dim; ++i) {
                            if (i != m_concat_dim) {
                                descriptors.push_back(IndexDesc(_));
                            }
                            else {
                                descriptors.push_back(IndexDesc(Slice(concat_dim_progress, concat_dim_progress + parent.shape()[m_concat_dim])));

                            }
                        }
                        Tensor<T> grad_view = create_lobotomized_slice_view(out_grad, descriptors);
                        parent.accumulate_grad(grad_view);
                    }
                    concat_dim_progress += parent.shape()[m_concat_dim];
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                std::vector<TensorStateBase*> dependencies;
                dependencies.reserve(std::ssize(m_parents_list));
                for (const Tensor<T>& parent : m_parents_list) {
                    dependencies.push_back(parent._get_state_base());
                }
                return dependencies;
            }
    };
}

