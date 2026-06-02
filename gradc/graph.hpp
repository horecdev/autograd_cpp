#pragma once
#include "tensor.hpp"

namespace gradc {
     template <typename T>
    class Node {
        public:
            Tensor<T> m_grad;
            virtual Tensor<T> forward() = 0;
            virtual T backward() = 0;

            virtual ~Node() {

            }
    };

    template <typename T>
    class TensorNode : public Node<T> {
        private:
            Tensor<T> m_tensor;
            size_t m_version; // TODO: REMOVE FROM HERE AND ADD TO STORAGE
        public:
            Tensor<T> forward() const override {
                return m_tensor;
            }

            void backward() const override {
                // on TensorNode it doesnt do anything.
            }
    };

    template <typename T>
    class AddNode : public Node<T> { // must hold pointers, not just raw TensorNode. If a, b are declared somewhere, then do a + b (inside some scope) and save a, b, when 
        // AddNode dies, it totally wipes a, b shape/stride/offset.
        // If you wrap a, b inside pointers - their ref count is 1, then 2, then 1.
        private:
            std::shared_ptr<Node<T>> m_left;
            std::shared_ptr<Node<T>> m_right;
        public:
            Tensor<T> forward() const override {
                return m_left->forward() + m_right->forward(); // AddNode doesnt require caching
            }

            void backward() const override {
                if (m_grad) 
            }

    };

    template <typename T>
    class MulNode : public Node<T> {
        private:
            std::shared_ptr<Node<T>> m_left;
            std::shared_ptr<Node<T>> m_right;

            Tensor<T> m_left_fwd;
            Tensor<T> m_right_fwd; 

        public:
            Tensor<T> forward() override {
                m_left_fwd = m_left->forward();
                m_right_fwd = m_right->forward();
                return m_left_fwd * m_right_fwd;
            }

            void backward() override {
                
            }
    };

    template <typename T>
    class InPlaceAddNode : public Node<T> {
        private:
            std::shared_ptr<TensorNode<T>> base;
            std::shared_ptr<Node<T>> incoming;
        public:
            Tensor<T>& forward() override {
                (*base).m_tensor += 
            }
    }

    template <typename T> 
    std::shared_ptr<AddNode<T>> operator+(std::shared_ptr<Node<T>> left, std::shared_ptr<Node<T>> right) {
        return std::make_shared<AddNode<T>>(std::move(left), std::move(right));
    }

    template <typename T> 
    std::shared_ptr<MulNode<T>> operator*(std::shared_ptr<Node<T>> left, std::shared_ptr<Node<T>> right) {
        return std::make_shared<MulNode<T>>(std::move(left), std::move(right));
    }
}