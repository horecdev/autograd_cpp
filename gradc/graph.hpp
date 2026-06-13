#pragma once

#include "node.hpp"
#include "tensor.hpp"
#include "impl/tensor_utils.hpp"

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

            Tensor<T> realize() override {
                m_parent.realize();
                return apply_reduction_operation(m_parent, m_reduction_metadata, T(), [](T a, T b){return a + b;});
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
    };


    template <typename T>
    class CloneNode : public Node<T> {
        private:
            Tensor<T> m_parent;
        public:
            CloneNode(Tensor<T> parent) : m_parent(std::move(parent)) {}

            Tensor<T> realize() override {
                m_parent.realize();
                return lobotomized_contiguous(m_parent); // does not copy whole data for a slice
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
                return lobotomized_contiguous(m_parent);
            }
    }; 

    template <typename T>
    class TransposeNode: public Node<T> {
        private: 
            Tensor<T> m_parent;
        public:
            TransposeNode(Tensor<T> parent) : m_parent(std::move(parent)) {}

            Tensor<T> realize() override {
                m_parent.realize();
                return m_parent;
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
    };

    template <typename T>
    class PermuteNode: public Node<T> {
        private:
            Tensor<T> m_parent;
        public:
            PermuteNode(Tensor<T> parent) : m_parent(std::move(parent)) {}

            Tensor<T> realize() override {
                m_parent.realize();
                return m_parent;
            }
    };

    template <typename T>
    class SliceNode: public Node<T> {
        private: 
            Tensor<T> m_parent;
        public:
            SliceNode(Tensor<T> parent) : m_parent(std::move(parent)) {}

            Tensor<T> realize() override {
                m_parent.realize();
                return m_parent;
            }
    };

    template <typename InT, typename OutT>
    class CastNode : public Node<OutT> {
        private: 
            Tensor<InT> m_parent;
        public:
            CastNode(Tensor<InT> parent) : m_parent(std::move(parent)) {}

            Tensor<OutT> realize() {
                m_parent.realize();
                Tensor<OutT> result = Tensor<OutT>(m_parent.shape());
                std::shared_ptr<TensorState<InT>> parent_state = this->get_state(m_parent);
                std::shared_ptr<TensorState<OutT>> result_state = this->get_state(result);
                for (int64_t i = 0; i < m_parent.volume(); ++i) {
                    (result_state->m_storage->m_data)[i] = static_cast<OutT>((parent_state->m_storage->m_data)[i]);
                }

                return result;
            }

    };
}