#pragma once
#include <iostream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <vector>
#include <array>

template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& vector) {
    for (size_t i = 0; i < vector.size(); ++i) {
        if (i == vector.size() - 1) {
            stream << vector[i];
        }
        else {
            stream << vector[i] << " ";
        }
        
    }
    return stream;
}

namespace gradc {

    struct Placeholder{};
    inline constexpr Placeholder _ = Placeholder(); // inline doesnt confuse linker if its #included in 2+ files.

    struct IndexDesc {
        bool m_is_all;
        int64_t m_value;

        IndexDesc(int64_t value)
            :   m_value(value), m_is_all(false) {}

        IndexDesc(Placeholder) 
            : m_value(0), m_is_all(true) {}
    };

    template<typename T>
    class Tensor {
        private:
            std::vector<size_t> m_shape;
            std::vector<size_t> m_strides;
            size_t m_offset;
            std::shared_ptr<std::vector<T>> m_data;
        
        public:
            // LIFECYCLE 
            Tensor();
            Tensor(std::vector<size_t> shape);
            Tensor(std::vector<size_t> shape, std::vector<size_t> strides, size_t offset, std::shared_ptr<std::vector<T>> data);
            ~Tensor();
            Tensor(const Tensor& source);
            Tensor(Tensor&& source);
            Tensor& operator=(const Tensor& source);
            Tensor& operator=(Tensor&& source);
            Tensor clone() const;


            
            // INDEXING
            template<typename... Args> // ... attached to typename means: create a bucket of types and name it Args.
            Tensor operator[](Args... args) const { // ... next to the type bucket, before args: Look element-wise at both. Put type inside type bucket, argument inside args.
                // The compiler sees you passed x: short, y: int, z: size_t, and automatically creates a template (short var_0, int var_1, size_t var_2)
                // For class templates: pass <float> etc. For funcion/operator templates you dont have to.
                if (sizeof...(args) != m_shape.size()) { // sizeof... literally means "count elements in the bucket". Its a built-in token.
                    throw std::out_of_range("Coordinate count does not match tensor dimensions.");
                }

                std::array<IndexDesc, sizeof...(args)> descriptors = {IndexDesc(args)...}; // when you do [expression(args)...] it means: apply expression to every arg, and separate it with comas.
                std::vector<size_t> new_shape;
                std::vector<size_t> new_strides;
                size_t new_offset = m_offset;
                new_shape.reserve(sizeof...(args)); // max amount of elements reserved upfront (evades dynamic reallocations)
                new_strides.reserve(sizeof...(args));
                for (int i = 0; i < sizeof...(args); ++i) {
                    if (descriptors[i].m_is_all) {
                        new_shape.push_back(m_shape[i]); // worked it out on paper
                        new_strides.push_back(m_strides[i]);   
                    }
                    else {
                        int64_t coord = descriptors[i].m_value;
                        if (descriptors[i].m_value < 0) { // (-1 on shape 4 becomes 3)
                            coord += static_cast<int64_t>(m_shape[i]);
                        }

                        if (coord < 0 || coord >= static_cast<int64_t>(m_shape[i])) {
                            throw std::out_of_range("Index out of bounds for tensor dimension.");
                        }
                        new_offset += static_cast<size_t>(coord) * m_strides[i];
                    }
                }
                return Tensor(std::move(new_shape), std::move(new_strides), new_offset, m_data); // backdoor construct shallow copy
            }

            T item() {
                if (m_shape.size() == 0) {
                    return m_data[m_offset];
                }
                else {
                    throw std::runtime_error(".item() can be called only on 0-dimensional tensors.");
                }
            }

            // GETTERS

            const std::vector<size_t>& shape() const {
                return m_shape;
            }

            const std::vector<size_t>& strides() const {
                return m_strides;
            }

            const std::size_t offset() const {
                return m_offset;
            }

            // SHAPES
            bool is_contiguous() const;
            Tensor contiguous() const;
            Tensor transpose(const size_t dim0, const size_t dim1) const;
            Tensor permute(const std::vector<int64_t>& axes) const;
            Tensor reshape(const std::vector<int64_t>& target_shape) const;
            Tensor broadcast_to(const std::vector<size_t>& target_shape) const;
            std::vector<size_t> infer_broadcast(const std::vector<size_t>& a, const std::vector<size_t>& b) const;

            // MATH
            template <typename Func>
            Tensor& apply_in_place(const Tensor& other, Func op);
            template <typename Func>
            Tensor apply_out_of_place(const Tensor& other, Func op) const;

            Tensor& operator+=(const Tensor& other);
            Tensor operator+(const Tensor& other) const;

            Tensor& operator-=(const Tensor& other);
            Tensor operator-(const Tensor& other) const;

            Tensor& operator*=(const Tensor& other);
            Tensor operator*(const Tensor& other) const;

            Tensor& operator/=(const Tensor& other);
            Tensor operator/(const Tensor& other) const;
    };
} 

#include "impl/tensor_shape.hpp"
#include "impl/tensor_math.hpp"

//     template <typename T>
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