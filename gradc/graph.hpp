#pragma once

#include "impl/tensor_indexing.hpp"
#include "node.hpp"
#include "tensor.hpp"
#include "impl/tensor_utils.hpp"

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

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
                return apply_out_of_place(m_left, m_right, m_target_shape, [](T a, T b) {return a + b;});
            }

            void backward(const Tensor<T>& out_grad) {
                m_left.accumulate_grad(unbroadcast_grad(out_grad, m_left));
                m_right.accumulate_grad(unbroadcast_grad(out_grad, m_right));
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_left.m_state.get(), m_right.m_state.get()};
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
                return apply_out_of_place(m_left, m_right, m_target_shape, [](T a, T b) {return a * b;});
            }

            void backward(const Tensor<T>& out_grad) {
                Tensor<T> raw_left_grad = apply_out_of_place(out_grad, m_right, m_target_shape, [](T a, T b){return a * b;});
                Tensor<T> raw_right_grad = apply_out_of_place(out_grad, m_left, m_target_shape, [](T a, T b){return a * b;});

                m_left.accumulate_grad(unbroadcast_grad(raw_left_grad, m_left));
                m_right.accumulate_grad(unbroadcast_grad(raw_right_grad, m_right));
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_left.m_state.get(), m_right.m_state.get()};
            }
    };

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
                m_parent.accumulate_grad(Tensor<T>(m_parent.m_shape, m_reduction_metadata.temp_strides, 0, out_grad.m_state->m_storage, false));
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent.m_state.get()};
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
                apply_in_place(summed, Tensor<T>(static_cast<T>(m_reduction_metadata.reduced_vol)), [](T& a, T b){a /= b;});
                return summed;
            }

            void backward(const Tensor<T>& out_grad) override {
                Tensor<T> divided_grad = apply_out_of_place(out_grad, Tensor<T>(static_cast<T>(m_reduction_metadata.reduced_vol)), out_grad.m_shape, [](T a, T b){return a / b;});
                Tensor<T> strided_mean_grad = Tensor<T>(m_parent.m_shape, m_reduction_metadata.temp_strides, 0, divided_grad.m_state->m_storage, false);
                m_parent.accumulate_grad(strided_mean_grad);
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent.m_state.get()};
            }
    };


    template <typename T>
    class CloneNode : public Node<T> {
        private:
            Tensor<T> m_parent;
        public:
            CloneNode(Tensor<T> parent) : m_parent(std::move(parent)) {}

            Tensor<T> realize() override {
                m_parent.realize();
                return lobotomized_contiguous_alloc(m_parent); // does not copy whole data for a slice
            }

            void backward(const Tensor<T>& out_grad) override {
                m_parent.accumulate_grad(out_grad); // literally just copy over
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent.m_state.get()};
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
                return lobotomized_contiguous_alloc(m_parent);
            }

            void backward(const Tensor<T>& out_grad) override {
                m_parent.accumulate_grad(out_grad);
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent.m_state.get()};
            }
    }; 

    template <typename T>
    class TransposeNode: public Node<T> {
        private: 
            Tensor<T> m_parent;
            int64_t m_dim0;
            int64_t m_dim1;

        public:
            TransposeNode(Tensor<T> parent, int64_t dim0, int64_t dim1) : m_parent(std::move(parent)), m_dim0(dim0), m_dim1(dim1) {}

            Tensor<T> realize() override {
                m_parent.realize();
                return m_parent;
            }

            void backward(const Tensor<T>& out_grad) override {
                Tensor<T> transposed_grad = lobotomized_transpose_view(out_grad, m_dim0,  m_dim1);
                m_parent.accumulate_grad(transposed_grad);
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent.m_state.get()};
            }
    };

    template <typename T>
    class ReshapeNode : public Node<T> {
        private:
            Tensor<T> m_parent;
        public:
            ReshapeNode(Tensor<T> parent) : m_parent(std::move(parent)) {}

            Tensor<T> realize() override {
                m_parent.realize();
                return m_parent;
            }

            void backward(const Tensor<T>& out_grad) override {
                Tensor<T> reshaped_grad = out_grad;
                reshaped_grad.m_shape = m_parent.m_shape; // parent and grad are contiguous, so can just take same strides.
                reshaped_grad.m_strides = m_parent.m_strides;
                m_parent.accumulate_grad(reshaped_grad);
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent.m_state.get()};
            }
    };

    template <typename T>
    class PermuteNode: public Node<T> {
        private:
            Tensor<T> m_parent;
            std::vector<int64_t> m_axes;
        public:
            PermuteNode(Tensor<T> parent, std::vector<int64_t> axes) : m_parent(std::move(parent)), m_axes(std::move(axes)) {}

            Tensor<T> realize() override {
                m_parent.realize();
                return m_parent;
            }

            void backward(const Tensor<T>& out_grad) override {
                const int64_t n_dim = std::ssize(m_parent.m_shape);
                std::vector<int64_t> normalized_axes = normalize_axes_vector(m_axes, n_dim);
                std::vector<int64_t> backward_axes(n_dim, -1);
                for (int64_t i = 0; i < n_dim; ++i) {
                    backward_axes[normalized_axes[i]] = i; // if 0 went to 2, then 2 goes to 0
                }

                Tensor<T> permuted_grad = lobotomized_permute_view(out_grad, backward_axes);
                m_parent.accumulate_grad(permuted_grad);
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent.m_state.get()};
            }
    };

    template <typename T>
    class SliceNode: public Node<T> {
        private: 
            Tensor<T> m_parent;
            std::vector<IndexDesc> m_descriptors;
        public:
            SliceNode(Tensor<T> parent, std::vector<IndexDesc> descriptors) : m_parent(std::move(parent)), m_descriptors(std::move(descriptors)) {}

            Tensor<T> realize() override {
                m_parent.realize();
                return m_parent;
            }

            void backward(const Tensor<T>& out_grad) override {
                Tensor<T> full_grad = Tensor<T>(m_parent.m_shape, T()); // Incoming grad is [5, 32] but the parent is [5, 10, 32]. 
                // You add dimension back for the temporary tensor (filled with 0s).
                Tensor<T> grad_view = full_grad.create_lobotomized_slice_view(m_descriptors);
                apply_in_place(grad_view, out_grad, [](T& a, T b){a = b;}); // grad_view and out_grad have the same dimensions.
                // grad_view has buffer of full_grad that matches shape of m_parent, so in-place assignment actually edits full_grad.
                m_parent.accumulate_grad(full_grad);
            } 

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent.m_state.get()};
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
                Tensor<OutT> result = lobotomized_cast_alloc<InT, OutT>(m_parent);
                return result;
            }

            void backward(const Tensor<OutT>& out_grad) override {
                Tensor<InT> cast_grad = lobotomized_cast_alloc<OutT, InT>(out_grad);
                m_parent.accumulate_grad(cast_grad);
            }

            std::vector<TensorStateBase*> get_input_states() override {
                return {m_parent.m_state.get()};
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
                    std::vector<IndexDesc> descriptors;
                    descriptors.reserve(n_dim);

                    for (int64_t i = 0; i < n_dim; ++i) {
                        if (i != m_concat_dim) {
                            descriptors.push_back(IndexDesc(_));
                        }
                        else {
                            descriptors.push_back(IndexDesc(Slice(concat_dim_progress, concat_dim_progress + parent.m_shape[m_concat_dim])));

                        }
                    }
                    Tensor<T> grad_view = create_lobotomized_slice_view(out_grad, descriptors);
                    parent.accumulate_grad(grad_view);
                    concat_dim_progress += parent.m_shape[m_concat_dim];
                }
            }

            std::vector<TensorStateBase*> get_input_states() override {
                std::vector<TensorStateBase*> dependencies;
                dependencies.reserve(std::ssize(m_parents_list));
                for (const Tensor<T>& parent : m_parents_list) {
                    dependencies.push_back(parent.m_state.get());
                }
                return dependencies;
            }
    };


}