#pragma once
#include "../tensor.hpp"
#include "../graph.hpp"

namespace gradc {
    
    // template <typename T>
    // Tensor<T>& Tensor<T>::operator+=(const Tensor<T>& other) {
    //     return apply_in_place(other, [](T &a, T b){a += b;});
    // }
    
    template <typename T>
    Tensor<T> operator+(Tensor<T> left, Tensor<T> right) {
        std::vector<size_t> target_shape;
        if (left.m_shape != right.m_shape) { // TODO: ADD FRIEND TO THESE FUNCTIONS SO THEY CAN ACCES PRIVATE VARIABLES
            target_shape = infer_broadcast(left.m_shape, right.m_shape); // crashes if incompatible on the graph-building stage.
        }
        else {
            target_shape = left.m_shape;
        }

        Tensor<T> new_tensor = Tensor<T>(target_shape, lazy);
        new_tensor.m_op = std::make_shared<AddNode<T>>(std::move(left), std::move(right));
        return new_tensor;
    }

    // template <typename T>
    // Tensor<T>& Tensor<T>::operator*=(const Tensor<T>& other) {
    //     return apply_in_place(other, [](T &a, T b){a *= b;});
    // }

    template <typename T>
    Tensor<T> operator*(Tensor<T> left, Tensor<T> right) {
        std::vector<size_t> target_shape;
        if (left.m_shape != right.m_shape) {
            target_shape = infer_broadcast(left.m_shape, right.m_shape); // crashes if incompatible on the graph-building stage.
        }
        else {
            target_shape = left.m_shape;
        }

        Tensor<T> new_tensor = Tensor<T>(target_shape, lazy);
        new_tensor.m_op = std::make_shared<MulNode<T>>(std::move(left), std::move(right));
        return new_tensor;
    }



    // template <typename T>
    // Tensor<T>& Tensor<T>::operator-=(const Tensor<T>& other) {
    //     return apply_in_place(other, [](T &a, T b){a -= b;});
    // }

    // template <typename T>
    // Tensor<T> Tensor<T>::operator-(const Tensor<T>& other) const {
    //     return apply_out_of_place(other, [](T a, T b) {return a - b;});
    // }


    // template <typename T>
    // Tensor<T>& Tensor<T>::operator/=(const Tensor<T>& other) {
    //     return apply_in_place(other, [](T &a, T b){a /= b;});
    // }

    // template <typename T>
    // Tensor<T> Tensor<T>::operator/(const Tensor<T>& other) const {
    //     return apply_out_of_place(other, [](T a, T b) {return a / b;});
    // }
}