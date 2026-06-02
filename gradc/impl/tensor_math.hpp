#pragma once
#include "../tensor.hpp"
#include "../graph.hpp"

namespace gradc {
    
    template <typename T>
    Tensor<T>& Tensor<T>::operator+=(const Tensor<T>& other) {
        return apply_in_place(other, [](T &a, T b){a += b;});
    }
    
    template <typename T>
    Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
        
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator-=(const Tensor<T>& other) {
        return apply_in_place(other, [](T &a, T b){a -= b;});
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator-(const Tensor<T>& other) const {
        return apply_out_of_place(other, [](T a, T b) {return a - b;});
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator*=(const Tensor<T>& other) {
        return apply_in_place(other, [](T &a, T b){a *= b;});
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator*(const Tensor<T>& other) const {
        return apply_out_of_place(other, [](T a, T b) {return a * b;});
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator/=(const Tensor<T>& other) {
        return apply_in_place(other, [](T &a, T b){a /= b;});
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator/(const Tensor<T>& other) const {
        return apply_out_of_place(other, [](T a, T b) {return a / b;});
    }
}