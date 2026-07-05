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
    Tensor<T> create_lobotomized_slice_view(const Tensor<T>& source, const std::vector<IndexDesc>& descriptors) {
        const int64_t n_dim = std::ssize(descriptors);
        if (n_dim != std::ssize(source.m_shape)) {
            throw std::out_of_range("Coordinate count does not match tensor dimensions.");
        }
        std::vector<int64_t> new_shape;
        std::vector<int64_t> new_strides;
        int64_t new_offset = source.m_offset;
        new_shape.reserve(n_dim); // max amount of elements reserved upfront (evades dynamic reallocations)
        new_strides.reserve(n_dim);

        for (int64_t i = 0; i < n_dim; ++i) {
            if (descriptors[i].m_type == IndexType::All) {
                new_shape.push_back(source.m_shape[i]); // worked it out on paper
                new_strides.push_back(source.m_strides[i]);   
            }
            else if (descriptors[i].m_type == IndexType::Single) {
                int64_t coord = descriptors[i].m_single_val;
                coord = normalize_axis(coord, source.m_shape[i]);
                new_offset += coord * source.m_strides[i];
            }
            else if (descriptors[i].m_type == IndexType::Range) {
                int64_t step = descriptors[i].m_slice_val.step.value_or(1);
                if (step == 0) {
                    throw std::runtime_error("Slice cannot have a step of 0.");
                }
                // [start, stop) so start has can be 0 or shape[i] - 1 since its inclusive
                // stop can have -1 when its step < 0 or m_shape since its exclusive
                int64_t start = descriptors[i].m_slice_val.start.value_or(step > 0 ? 0 : source.m_shape[i] - 1);
                int64_t stop = descriptors[i].m_slice_val.stop.value_or(step > 0 ? source.m_shape[i] : -1);

                if (start < 0) { // start cannot be < 0 in any scenario
                    start += source.m_shape[i];
                }
                if (descriptors[i].m_slice_val.stop.has_value() && stop < 0) {
                    stop += source.m_shape[i];
                }

                if (start < 0 || start >= source.m_shape[i]) {
                    throw std::out_of_range("Range start is out of bounds.");
                }
                if (stop < -1 || stop > source.m_shape[i]) {
                    throw std::out_of_range("Range stop is out of bounds.");
                }
                if (step > 0 && start >= stop) {
                    throw std::runtime_error("Invalid forward slice: start >= stop with positive step.");
                }
                if (step < 0 && start <= stop) {
                    throw std::runtime_error("Invalid backward slice: start <= stop with negative step.");
                }

                int64_t n_dist;
                int64_t abs_step;
                if (step > 0) {
                    n_dist = stop - start;
                    abs_step = step;
                }
                else {
                    n_dist = start - stop;
                    abs_step = -step;
                }
                int64_t slice_len;
                if (n_dist % abs_step == 0) { // If step_size perfectly divides N, then we have N / step_size. Otherwise N / step_size + 1. (works for [a, b) range)
                    slice_len = n_dist / abs_step;
                }
                else {
                    slice_len = (n_dist / abs_step) + 1;
                }

                new_shape.push_back(slice_len);
                new_strides.push_back(source.m_strides[i] * step); // for neg strides just go backwards from the end. For pos just skip few elems.
                new_offset += start * source.m_strides[i]; // for neg starting from the back of the axis (say 4:2:-1). For pos from the front.
            }
        }

        Tensor<T> result = Tensor(std::move(new_shape), std::move(new_strides), new_offset, source.m_state->m_storage, false);
        
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