#pragma once
#include "../tensor.hpp"

namespace gradc {
    template <typename T>
    Tensor<T>::Tensor() : m_shape({}), m_strides({}), m_offset(0), m_storage(nullptr) {} // default constructor
    
    template <typename T>
    Tensor<T>::Tensor(std::vector<size_t> shape) 
        // can pass integer as m_strides, because it implicitly constructs a std::vector by just variable(arguments)
        : m_shape(std::move(shape)), m_strides(m_shape.size()), m_offset(0) {
            if (m_shape.size() == 0) { // a scalar (0-dimensional)
                m_storage = std::make_shared<Storage<T>>(std::vector<T>(1));
            }
            else {
                m_strides[m_shape.size() - 1] = 1; 
                for (size_t i = m_shape.size() - 1; i > 0; --i) {
                    m_strides[i - 1] = m_shape[i] * m_strides[i];
                }
                m_storage = std::make_shared<Storage<T>>(std::vector<T>(m_shape[0] * m_strides[0]));
            }
        }

    template <typename T>
    Tensor<T>::Tensor(std::vector<size_t> shape, std::vector<size_t> strides, size_t offset, std::shared_ptr<Storage<T>> storage) // backdoor
        : m_shape(std::move(shape)), m_strides(std::move(strides)), m_offset(offset), m_storage(std::move(storage)) {} 
    
    template <typename T>
    Tensor<T>::~Tensor() {
        std::cout << "Tensor Destroyed" << std::endl;
    }

    template <typename T>
    Tensor<T>::Tensor(const Tensor& source) 
        : m_shape(source.m_shape), m_strides(source.m_strides), m_offset(source.m_offset), m_storage(source.m_storage) {} // copy constructor (shallow copy) [Tensor b = a]
        // boosts ref count by 1 (shallow copy)

    template <typename T>
    Tensor<T>::Tensor(Tensor&& source) // move constructor [Tensor c = a + b]
        : m_shape(std::move(source.m_shape)), m_strides(std::move(source.m_strides)), m_offset(source.m_offset), m_storage(std::move(source.m_storage)) {}

    template <typename T>
    Tensor<T>& Tensor<T>::operator=(const Tensor& source) { // copy assignment operator [c = b]
        if (this != &source) {
            m_shape = source.m_shape;
            m_strides = source.m_strides;
            m_offset = source.m_offset;
            m_storage = source.m_storage; // all are copied. This line increments ref count by 1, and decrements the prev ref count.
            // we dont have to manually delete everything because its std::vector and std::shared_ptr. Smart classes.
            // SHALLOW COPY
        }
        return *this;
    }

    template <typename T>
    Tensor<T>& Tensor<T>::operator=(Tensor&& source) { // move assignment operator [a = b + c]
        if (this != &source) { // [a = transpose(a)], and transpose modifies it in-place, then it would trigger
            m_shape = std::move(source.m_shape);
            m_strides = std::move(source.m_strides);
            m_offset = source.m_offset;
            m_storage = std::move(source.m_storage);
        }   
        return *this;
    }

    template <typename T>
    Tensor<T> Tensor<T>::clone() const {
        // Triggers Storage copy constructor - new heap memo allocated
        auto ptr = std::make_shared<Storage<T>>(*m_storage); // shared_ptr pointing to a new vector with copied data. (ref_count = 1)
        return Tensor(m_shape, m_strides, m_offset, std::move(ptr));
    }
}