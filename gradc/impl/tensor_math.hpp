#pragma once
#include "../tensor.hpp"
#include "../graph.hpp"
#include <stdexcept>

namespace gradc {
    

    
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
        new_tensor.m_op = std::make_shared<AddNode<T>>(std::move(left), std::move(right), std::move(target_shape));
        return new_tensor;
    }

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

    template <typename T>
    Tensor<T>& operator+=(Tensor<T>& main, Tensor<T> other) {
        std::vector<size_t> common_shape;
        if (main.m_shape != other.m_shape) { // Graph validator of shapes
            common_shape = infer_broadcast(main.m_shape, other.m_shape);
            if (common_shape != main.m_shape) {
                throw std::runtime_error("LHS cannot expand its size during in-place math.");
            }
        }
        Tensor<T> old_main = main; // copy history up to this point
        main.m_op = std::make_shared<InPlaceAddNode<T>>(std::move(old_main), std::move(other));
        
        return main;
    }


}