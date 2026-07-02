#pragma once

#include "../tensor.hpp"
#include "../graph.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace gradc {
    template <typename T>
    Tensor<T>::Tensor() : m_shape({}), m_strides({}), m_offset(0), m_state(nullptr), m_requires_grad(false) {} // default constructor
    
    template <typename T>
    Tensor<T>::Tensor(T value) : m_shape({}), m_strides({}), m_offset(0), m_state(std::make_shared<TensorState<T>>(std::vector<T>(1, value))), m_requires_grad(false) {}

    template <typename T>
    Tensor<T>::Tensor(std::vector<int64_t> shape, T init_val) 
        // can pass integer as m_strides, because it implicitly constructs a std::vector by just variable(arguments)
        : m_shape(std::move(shape)), m_strides(std::ssize(m_shape)), m_offset(0), m_requires_grad(false) {
            if (std::ssize(m_shape) == 0) { // a scalar (0-dimensional)
                m_state = std::make_shared<TensorState<T>>(std::vector<T>(1, init_val));
            }
            else {
                m_strides[std::ssize(m_shape) - 1] = 1; 
                for (int64_t i = std::ssize(m_shape) - 1; i > 0; --i) {
                    m_strides[i - 1] = m_shape[i] * m_strides[i];
                }
                m_state = std::make_shared<TensorState<T>>(std::vector<T>(m_shape[0] * m_strides[0], init_val));
            }
        }

    template <typename T> // makes a contiguous eager copy (figures out stides) with shared storage.
    Tensor<T>::Tensor(std::vector<int64_t> shape, std::shared_ptr<Storage<T>> storage) 
        : m_shape(std::move(shape)), m_strides(std::ssize(shape)), m_offset(0), m_state(std::make_shared<TensorState<T>>(std::move(storage))), m_requires_grad(false) {
            if (std::ssize(m_shape) > 0) {
                m_strides[std::ssize(m_shape) - 1] = 1; 
                for (int64_t i = std::ssize(m_shape) - 1; i > 0; --i) {
                    m_strides[i - 1] = m_shape[i] * m_strides[i];
                }
            }
        }

    template <typename T>
    Tensor<T>::Tensor(std::vector<int64_t> shape, bool requires_grad, LazyTag)
        // can pass integer as m_strides, because it implicitly constructs a std::vector by just variable(arguments)
        : m_shape(std::move(shape)), m_strides(std::ssize(m_shape)), m_offset(0), m_requires_grad(validate_requires_grad(requires_grad)) {
            if (std::ssize(m_shape) == 0) { // a scalar (0-dimensional)
                m_state = std::make_shared<TensorState<T>>();
            }
            else {
                m_strides[std::ssize(m_shape) - 1] = 1; 
                for (int64_t i = std::ssize(m_shape) - 1; i > 0; --i) {
                    m_strides[i - 1] = m_shape[i] * m_strides[i];
                }
                m_state = std::make_shared<TensorState<T>>();
            }
        }

    template <typename T> // backdoor
    Tensor<T>::Tensor(std::vector<int64_t> shape, std::vector<int64_t> strides, int64_t offset, std::shared_ptr<TensorState<T>> state, bool requires_grad) 
        : m_shape(std::move(shape)), m_strides(std::move(strides)), m_offset(offset), m_state(std::move(state)), m_requires_grad(validate_requires_grad(requires_grad)) {} 

    template <typename T> // lobotomy constructor
    Tensor<T>::Tensor(std::vector<int64_t> shape, std::vector<int64_t> strides, int64_t offset, std::shared_ptr<Storage<T>> storage, bool requires_grad) 
        : m_shape(std::move(shape)), m_strides(std::move(strides)), m_offset(offset), m_requires_grad(validate_requires_grad(requires_grad)) {
            m_state = std::make_shared<TensorState<T>>(std::move(storage)); // op is nullptr
        }
    
    template <typename T>
    Tensor<T>::~Tensor() {
        // std::cout << "Tensor Destroyed" << std::endl;
    }

    template <typename T> // realizing one alias realizes all (shared TensorState)
    Tensor<T>::Tensor(const Tensor& source) 
        : m_shape(source.m_shape), m_strides(source.m_strides), m_offset(source.m_offset), m_state(source.m_state), m_requires_grad(source.m_requires_grad) {}


    template <typename T>
    Tensor<T>::Tensor(Tensor&& source) // move constructor [Tensor c = a + b]
        : m_shape(std::move(source.m_shape)), m_strides(std::move(source.m_strides)), m_offset(source.m_offset), m_state(std::move(source.m_state)), m_requires_grad(source.m_requires_grad) {}

    template <typename T>
    Tensor<T>& Tensor<T>::operator=(const Tensor& source) { // copy assignment operator [c = b]
        if (this != &source) {
            m_shape = source.m_shape;
            m_strides = source.m_strides;
            m_offset = source.m_offset;
            m_state = source.m_state;
            m_requires_grad = source.m_requires_grad;
        }
        return *this;
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator=(Tensor&& source) { // move assignment operator [a = b + c]
        if (this != &source) { // [a = transpose(a)], and transpose modifies it in-place, then it would trigger
            m_shape = std::move(source.m_shape);
            m_strides = std::move(source.m_strides);
            m_offset = source.m_offset;
            m_state = std::move(source.m_state);
            m_requires_grad = source.m_requires_grad;
        }   
        return *this;
    }

    template <typename T>
    Tensor<T> Tensor<T>::clone() const { 
        Tensor<T> tensor_copy = Tensor(m_shape, m_requires_grad, lazy);
        tensor_copy.m_state->m_creation_op = std::make_unique<CloneNode<T>>(*this);
        return tensor_copy;
    }

    template <typename T>
    template <typename TargetT>
    Tensor<TargetT> Tensor<T>::cast() const {
        if constexpr (std::is_same_v<T, TargetT>) {
            return *this;
        }
        else {
            Tensor<TargetT> new_tensor = Tensor<TargetT>(m_shape, m_requires_grad, lazy);
            new_tensor.m_state->m_creation_op = std::make_unique<CastNode<T, TargetT>>(*this);
            return new_tensor;
        }
    }
}