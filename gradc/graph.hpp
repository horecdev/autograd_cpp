// #pragma once
// #include "tensor.hpp"

// namespace gradc {
//      template <typename T>
//     class Node {
//         public:
//             Tensor<T> m_grad;
//             virtual T forward() = 0;
//             virtual T backward() = 0;

//             virtual ~Node() {

//             }
//     };

//     template <typename T>
//     class TensorNode : public Node<T> {
//         private:
//             Tensor<T> m_tensor;
//         public:
//             Tensor<T> forward() const override {
//                 return m_tensor;
//             }

//             void backward() const override {
                
//             }
//     };

//     template <typename T>
//     class AddNode : public Node<T> { // must hold pointers, not just raw TensorNode. If a, b are declared somewhere, then do a + b (inside some scope) and save a, b, when 
//         // AddNode dies, it totally wipes a, b shape/stride/offset.
//         // If you wrap a, b inside pointers - their ref count is 1, then 2, then 1.
//         private:
//             std::shared_ptr<Node<T>> m_left;
//             std::shared_ptr<Node<T>> m_right;
//         public:
//             Tensor<T> forward() const override {
//                 return m_left->eval() + m_right->eval();
//             }

//             void backward() const override { // by the time node.backward() is called, its m_grad is already fully populated
//                 (*m_left).m_grad += m_grad;
//                 (*m_right).m_grad += m_grad;
//             }
//     };

//     template <typename T>
//     class MulNode : public Node<T> {
//         private:
//             std::shared_ptr<Node<T>> m_left;
//             std::shared_ptr<Node<T>> m_right;
//         public:
//             Tensor<T> forward() const override {
//                 return m_left->eval() * m_right->eval();
//             }

//             void backward() const override { // by the time node.backward() is called, its m_grad is already fully populated
//                 (*m_left).m_grad += m_grad * (*m_right.);
//                 (*m_right).m_grad += m_grad;
//             }
//     };
// }