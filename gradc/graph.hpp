#pragma once

#include "node.hpp"
#include "tensor.hpp"
#include "impl/tensor_utils.hpp"

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
            
            Tensor<T> realize() const override {
                m_left.realize();
                m_right.realize();
                return apply_out_of_place(m_left, m_right, m_target_shape, [](T a, T b) {return a + b;});
            }

            std::vector<TensorStateBase*> get_input_states() const override {
                return {m_left.m_state.get(), m_right.m_state.get()};
            }

            void backward(const Tensor<T>& out_grad) {
                m_left.accumulate_grad(unbroadcast_grad(out_grad, m_left));
                m_right.accumulate_grad(unbroadcast_grad(out_grad, m_right));
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
            
            Tensor<T> realize() const override {
                m_left.realize();
                m_right.realize();
                return apply_out_of_place(m_left, m_right, m_target_shape, [](T a, T b) {return a * b;});
            }

            std::vector<TensorStateBase*> get_input_states() const override {
                return {m_left.m_state, m_right.m_state};
            }

            void backward(const Tensor<T>& out_grad) {
                Tensor<T> raw_left_grad = apply_out_of_place(out_grad, m_right, m_target_shape, [](T a, T b){return a * b;});
                Tensor<T> raw_right_grad = apply_out_of_place(out_grad, m_left, m_target_shape, [](T a, T b){return a * b;});

                m_left.accumulate_grad(unbroadcast_grad(raw_left_grad, m_left));
                m_right.accumulate_grad(unbroadcast_grad(raw_right_grad, m_right));
            }
    };


    template <typename T>
    class InPlaceAddNode : public Node<T> {
        private:
            Tensor<T> m_left; // pre-addition tensor
            Tensor<T> m_right; 
        public:
            InPlaceAddNode(Tensor<T> left, Tensor<T> right) : m_left(std::move(left)), m_right(std::move(right)) {}
        
            Tensor<T> realize() override {
                m_left.realize();
                m_right.realize();
                apply_in_place(m_left, m_right, [](T& a, T b){a += b;}); 
                return m_left;
            }
        };

    template <typename T>
    class InPlaceMulNode : public Node<T> {
        private:
            Tensor<T> m_left;
            Tensor<T> m_right; 
        public:
            InPlaceMulNode(Tensor<T> left, Tensor<T> right) : m_left(std::move(left)), m_right(std::move(right)) {}
        
            Tensor<T> realize() override {
                m_left.realize();
                m_right.realize();
                apply_in_place(m_left, m_right, [](T& a, T b){a *= b;}); 
                return m_left;
            }
    };

    template <typename T>
    class SumNode : public Node<T> {
        private:
            Tensor<T> m_parent;
            ReductionMetadata m_reduction_metadata;
        public:
            SumNode(Tensor<T> parent, ReductionMetadata reduction_metadata) : m_parent(parent), m_reduction_metadata(reduction_metadata) {}

            Tensor<T> realize() const override {
                m_parent.realize();
                return apply_reduction_operation(m_parent, m_reduction_metadata, T(), [](T a, T b){return a + b;});
            }

            void backward(const Tensor<T>& out_grad) const override {
                m_parent.accumulate_grad(Tensor<T>(m_parent.m_shape, m_reduction_metadata.temp_strides, 0, out_grad.m_state->m_storage, false));
            }
    };

    template <typename T>
    class MeanNode : public Node<T> {
        private:
            Tensor<T> m_parent;
            ReductionMetadata m_reduction_metadata;
        public:
            MeanNode(Tensor<T> parent, ReductionMetadata reduction_metadata) : m_parent(parent), m_reduction_metadata(reduction_metadata) {}

            Tensor<T> realize() const override {
                m_parent.realize();
                Tensor<T> summed = apply_reduction_operation(m_parent, m_reduction_metadata, T(), [](T a, T b){return a + b;});
                apply_in_place(summed, Tensor<T>(static_cast<T>(m_reduction_metadata.reduced_vol)), [](T& a, T b){a /= b;});
                return summed;
            }

            void backward(const Tensor<T>& out_grad) const override {
                Tensor<T> divided_grad = apply_out_of_place(out_grad, Tensor<T>(static_cast<T>(m_reduction_metadata.reduced_vol)), out_grad.m_shape, [](T a, T b){return a / b;});
                Tensor<T> strided_mean_grad = Tensor<T>(m_parent.m_shape, m_reduction_metadata.temp_strides, 0, divided_grad.m_state->m_storage, false);
                m_parent.accumulate_grad(strided_mean_grad);
            }
    };


    template <typename T>
    class CloneNode : public Node<T> {
        private:
            Tensor<T> m_parent;
        public:
            CloneNode(Tensor<T> parent) : m_parent(std::move(parent)) {}

            Tensor<T> realize() const override {
                m_parent.realize();
                return lobotomized_contiguous(m_parent); // does not copy whole data for a slice
            }

            void backward(const Tensor<T>& out_grad) const override {
                m_parent.accumulate_grad(out_grad); // literally just copy over
            }
    };

    template <typename T>
    class ContiguousNode: public Node<T> {
        private: 
            Tensor<T> m_parent;
        public:
            ContiguousNode(Tensor<T> parent) : m_parent(std::move(parent)) {}

            Tensor<T> realize() const override {
                m_parent.realize();
                return lobotomized_contiguous(m_parent);
            }

            void backward(const Tensor<T>& out_grad) const override {
                m_parent.accumulate_grad(out_grad);
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

            Tensor<T> realize() const override {
                m_parent.realize();
                return m_parent;
            }

            void backward(const Tensor<T>& out_grad) const override {
                Tensor<T> transposed_grad = lobotomized_transpose(out_grad, m_dim0,  m_dim1);
                m_parent.accumulate_grad(transposed_grad);
            }
    };

    template <typename T>
    class ReshapeNode : public Node<T> {
        private:
            Tensor<T> m_parent;
        public:
            ReshapeNode(Tensor<T> parent) : m_parent(std::move(parent)) {}

            Tensor<T> realize() const override {
                m_parent.realize();
                return m_parent;
            }

            void backward(const Tensor<T>& out_grad) const override {
                Tensor<T> reshaped_grad = out_grad;
                reshaped_grad.m_shape = m_parent.m_shape; // parent and grad are contiguous, so can just take same strides.
                reshaped_grad.m_strides = m_parent.m_strides;
                m_parent.accumulate_grad(reshaped_grad);
            }
    };

    template <typename T>
    class PermuteNode: public Node<T> {
        private:
            Tensor<T> m_parent;
            std::vector<int64_t> m_axes;
        public:
            PermuteNode(Tensor<T> parent, std::vector<int64_t> axes) : m_parent(std::move(parent)), m_axes(std::move(axes)) {}

            Tensor<T> realize() const override {
                m_parent.realize();
                return m_parent;
            }

            void backward(const Tensor<T>& out_grad) const override {
                const int64_t n_dim = std::ssize(m_parent.m_shape);
                std::vector<int64_t> normalized_axes = normalize_axes_vector(m_axes, n_dim);
                std::vector<int64_t> backward_axes(n_dim, -1);
                for (int64_t i = 0; i < n_dim; ++i) {
                    backward_axes[normalized_axes[i]] = i; // if 0 went to 2, then 2 goes to 0
                }

                Tensor<T> permuted_grad = lobotomized_permute(out_grad, backward_axes);
                m_parent.accumulate_grad(permuted_grad);
            }
    };

    template <typename T>
    class SliceNode: public Node<T> {
        private: 
            Tensor<T> m_parent;
            std::vector<IndexDesc> m_descriptors;
        public:
            SliceNode(Tensor<T> parent, std::vector<IndexDesc> descriptors) : m_parent(std::move(parent)), m_descriptors(std::move(descriptors)) {}

            Tensor<T> realize() const override {
                m_parent.realize();
                return m_parent;
            }

            void backward(const Tensor<T>& out_grad) const override {
                Tensor<T> full_grad = Tensor<T>(m_parent.m_shape, T()); // Incoming grad is [5, 32] but the parent is [5, 10, 32]. 
                // You add dimension back for the temporary tensor (filled with 0s).
                Tensor<T> grad_view = full_grad.create_lobotomized_slice_view(m_descriptors);
                apply_in_place(grad_view, out_grad, [](T& a, T b){a = b;}); // grad_view and out_grad have the same dimensions.
                // grad_view has buffer of full_grad that matches shape of m_parent, so in-place assignment actually edits full_grad.
                m_parent.accumulate_grad(full_grad);
            } 
    };

    template <typename InT, typename OutT>
    class CastNode : public Node<OutT> {
        private: 
            Tensor<InT> m_parent;
        public:
            CastNode(Tensor<InT> parent) : m_parent(std::move(parent)) {}

            Tensor<OutT> realize() const override {
                m_parent.realize();
                Tensor<OutT> result = lobotomized_cast<InT, OutT>(m_parent);
                return result;
            }

            void backward(const Tensor<OutT>& out_grad) const override {
                Tensor<InT> cast_grad = lobotomized_cast<OutT, InT>(out_grad);
                m_parent.accumulate_grad(cast_grad);
            }
    };

    template <typename T>
    class ConcatNode : public Node<T> {
        private:
            std::vector<Tensor<T>> m_parents_list;
            int64_t m_concat_dim;
        public:
            ConcatNode(std::vector<Tensor<T>> parents_list, int64_t concat_dim) : m_parents_list(std::move(parents_list)), m_concat_dim(concat_dim) {}

            Tensor<T> realize() const override {
                for (Tensor<T>& parent : m_parents_list) {
                    parent.realize();
                }

                const int64_t n_dim = std::ssize(m_parents_list[0].m_shape);
                std::vector<int64_t> final_shape = m_parents_list[0].m_shape;
                final_shape[m_concat_dim] = 0;

                for (const Tensor<T>& parent : m_parents_list) {
                    if (std::ssize(parent.m_shape) != n_dim) {
                        throw std::runtime_error("Error during concat operation: ");
                    }
                    for (int64_t i = 0; i < n_dim; ++i) {

                    }
                }

            }
    };


}