#pragma once 

#include "../tensor.hpp"
#include "../graph.hpp"
#include "tensor_utils.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gradc {

    template <typename T> 
    Tensor<T> Tensor<T>::create_lobotomized_slice_view(const std::vector<IndexDesc>& descriptors) {
        const int64_t n_dim = std::ssize(descriptors);
        if (n_dim != std::ssize(m_shape)) {
            throw std::out_of_range("Coordinate count does not match tensor dimensions.");
        }
        std::vector<int64_t> new_shape;
        std::vector<int64_t> new_strides;
        int64_t new_offset = m_offset;
        new_shape.reserve(n_dim); // max amount of elements reserved upfront (evades dynamic reallocations)
        new_strides.reserve(n_dim);

        for (int i = 0; i < n_dim; ++i) {
            if (descriptors[i].m_is_all) {
                new_shape.push_back(m_shape[i]); // worked it out on paper
                new_strides.push_back(m_strides[i]);   
            }
            else {
                int64_t coord = descriptors[i].m_value;
                coord = normalize_axis(coord, m_shape[i]);
                new_offset += coord * m_strides[i];
            }
        }
        Tensor<T> result = Tensor(std::move(new_shape), std::move(new_strides), new_offset, m_state->m_storage, false);
        
        return result;
    }

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

        Tensor<T> result = this->create_lobotomized_slice_view(descriptors);

        result.m_requires_grad = m_requires_grad;
        result.m_state->m_creation_op = std::make_unique<SliceNode<T>>(*this, descriptors);

        return result;
    }

    template <typename T>
    T Tensor<T>::item() const {
        if (std::ssize(m_shape) == 0) {
            if (m_state->m_storage->m_data.empty()) {
                throw std::runtime_error("Called .item() on a tensor without data.");
            }
            return (m_state->m_storage->m_data)[m_offset];
        }
        else {
            throw std::runtime_error(".item() can be called only on 0-dimensional tensors.");
        }
    }
}