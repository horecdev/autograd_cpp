#pragma once 

#include "../tensor.hpp"
#include "../graph.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gradc {
    template <typename T>
    template <typename... Args> // ... attached to typename means: create a bucket of types and name it Args.
    Tensor<T> Tensor<T>::operator[](Args... args) const { // ... next to the type bucket, before args: Look element-wise at both. Put type inside type bucket, argument inside args.
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
        Tensor<T> result = Tensor(std::move(new_shape), std::move(new_strides), new_offset, m_state->m_storage, m_requires_grad);
        result.m_state->m_op = std::make_unique<SliceNode<T>>(*this);
        return result;
    }

    template <typename T>
    T Tensor<T>::item() const {
        if (m_shape.size() == 0) {
            return (m_state->m_storage->m_data)[m_offset];
        }
        else {
            throw std::runtime_error(".item() can be called only on 0-dimensional tensors.");
        }
    }
}