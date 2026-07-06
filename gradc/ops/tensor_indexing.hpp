#pragma once

#include "../core/tensor.hpp"
#include "../graph/nodes/view_nodes.hpp"

namespace gradc {
    template <typename T>
    template <typename... Args> // ... attached to typename means: create a bucket of types and name it Args.
    Tensor<T> Tensor<T>::operator[](Args... args) const { // ... next to the type bucket, before args: Look element-wise at both. Put type inside type bucket, argument inside args.
        // The compiler sees you passed x: short, y: int, z: int64_t, and automatically creates a template (short var_0, int var_1, int64_t var_2)
        // For class templates: pass <float> etc. For funcion/operator templates you dont have to.
        if (sizeof...(args) != std::ssize(m_shape)) { // sizeof... literally means "count elements in the bucket". Its a built-in token.
            throw std::out_of_range("Coordinate count does not match tensor dimensions.");
        }

        std::array<IndexDesc, sizeof...(args)> arr = {IndexDesc(args)...}; // when you do [expression(args)...] it means: apply expression to every arg, and separate it with comas.
        std::vector<IndexDesc> descriptors(arr.begin(), arr.end());

        Tensor<T> result = create_lobotomized_slice_view(*this, descriptors);

        result.m_requires_grad = m_requires_grad;
        result.m_state->m_creation_op = std::make_unique<SliceNode<T>>(*this, descriptors);

        return result;
    }

    template <typename T>
    T Tensor<T>::item() const {
        if (std::ssize(m_shape) == 0) {
            if (m_state->m_storage->data() == nullptr) {
                throw std::runtime_error("Called .item() on a tensor with size of 0.");
            }
            return (m_state->m_storage->m_data)[m_offset];
        }
        else {
            throw std::runtime_error(".item() can be called only on 0-dimensional tensors.");
        }
    }
}