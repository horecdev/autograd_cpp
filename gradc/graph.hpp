#pragma once
#include "tensor.hpp" // IWYU pragma: keep
#include "node.hpp"
#include "impl/tensor_math.hpp"

namespace gradc {
    template <typename T>
    class AddNode : public Node<T> {
        private:
            Tensor<T> m_left;
            Tensor<T> m_right;
            std::vector<size_t> m_target_shape;
        public:
            AddNode<T>(Tensor<T> left, Tensor<T> right, std::vector<size_t> target_shape) : m_left(std::move(left)), m_right(std::move(right)), m_target_shape(std::move(target_shape)) {}
            
            Tensor<T> realize() override {
                m_left.realize();
                m_right.realize();
                return apply_out_of_place(m_left, m_right, m_target_shape, [](T a, T b) {return a + b;});
            }

            void backward() override {}
    };

    template <typename T>
    class MulNode : public Node<T> {
        private:
            Tensor<T> m_left;
            Tensor<T> m_right;
            std::vector<size_t> m_target_shape;
        public:
            MulNode<T>(Tensor<T> left, Tensor<T> right, std::vector<size_t> target_shape) : m_left(std::move(left)), m_right(std::move(right)), m_target_shape(std::move(target_shape)) {}
            
            Tensor<T> realize() override {
                m_left.realize();
                m_right.realize();
                return apply_out_of_place(m_left, m_right, m_target_shape, [](T a, T b) {return a * b;});
            }

            void backward() override {}
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
                apply_in_place(m_left, m_right, [](T &a, T b){a += b;}); // version is bumped up, modifies m_left
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
                apply_in_place(m_left, m_right, [](T &a, T b){a *= b;}); // version is bumped up, modifies m_left
                return m_left;
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

                Tensor<T> result = m_parent; // same metadata as parent (could be just Tensor() presumably)
                result.m_storage = std::make_shared<Storage<T>>(*m_parent.m_storage); // totally new data vector

                return result;
            }
    };

    template <typename T>
    class ReshapeNode : public Node<T> {

    };
}