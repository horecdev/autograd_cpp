#pragma once
#include <iostream>
#include <memory>
#include <ostream>
#include <stdexcept> // IWYU pragma: keep
#include <vector>
#include <array> // IWYU pragma: keep

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

    struct LazyTag{};
    inline constexpr LazyTag lazy = LazyTag();

    template <typename T>
    class Node;

    struct IndexDesc {
        bool m_is_all;
        int64_t m_value;

        IndexDesc(int64_t value)
            :   m_value(value), m_is_all(false) {}

        IndexDesc(Placeholder) 
            : m_value(0), m_is_all(true) {}
    };

    template <typename T>
    struct Storage{
        std::vector<T> m_data;
        size_t m_version;

        Storage() {
            m_data = std::vector<T>{};
            m_version = 0;
        }

        Storage(std::vector<T>&& data) : m_data(std::move(data)), m_version(0) {}
        // If there is a new buffer - zero out the version.
        Storage(const Storage& other) : m_data(other.m_data), m_version(0) {}
        Storage(Storage&& other) : m_data(std::move(other.m_data)), m_version(0) {}
        
        Storage& operator=(const Storage& other) {
            if (this != &other) {
                m_data = other.m_data; // copy deep
                m_version = 0;
            }
            return *this;

        }
        Storage& operator=(Storage&& other) {
            if (this != &other) {
                m_data = std::move(other.m_data);
                m_version = 0;
            }
            return *this;
        } 

        ~Storage() {
            std::cout << "Storage Destroyed" << std::endl;
        }
    };

    template <typename T>
    class Tensor {
        private:
            std::vector<size_t> m_shape;
            std::vector<size_t> m_strides;
            size_t m_offset;
            std::shared_ptr<Storage<T>> m_storage;
            //bool m_requires_grad;
            std::shared_ptr<Node<T>> m_op; // if its a not a shared_ptr you get (2^N) exponential growth of tree with each op 
            // (to copy a tensor you copy a node and to copy node you copy a tensor...)
        
        public:
            Tensor& realize() {
                if (m_storage->m_data.empty() && m_op != nullptr) { // do I need to do math AND I know how to do math?
                    Tensor computed_result = m_op->realize();
                    m_storage->m_data = std::move(computed_result.m_storage->m_data);
                    m_shape = std::move(computed_result.m_shape);
                }

                return *this;
            }
            // LIFECYCLE 
            Tensor();
            Tensor(std::vector<size_t> shape);
            Tensor(std::vector<size_t> shape, LazyTag);
            Tensor(std::vector<size_t> shape, std::vector<size_t> strides, size_t offset, std::shared_ptr<Storage<T>> data, std::shared_ptr<Node<T>> op);
            ~Tensor();
            Tensor(const Tensor& source);
            Tensor(Tensor&& source);
            Tensor& operator=(const Tensor& source);
            Tensor& operator=(Tensor&& source);
            Tensor clone() const;

            // INDEXING
            template <typename... Args>
            Tensor operator[](Args... args) const;

            T item() const;

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

            //Tensor& operator+=(const Tensor& other);
            //Tensor operator+(const Tensor& other) const;

            // Tensor& operator-=(const Tensor& other);
            // Tensor operator-(const Tensor& other) const;

            //Tensor& operator*=(const Tensor& other);
            //Tensor operator*(const Tensor& other) const;

            // Tensor& operator/=(const Tensor& other);
            // Tensor operator/(const Tensor& other) const;
    };
} 

#include "impl/tensor_lifecycle.hpp" // IWYU pragma: keep
#include "impl/tensor_indexing.hpp" // IWYU pragma: keep
#include "impl/tensor_shape.hpp" // IWYU pragma: keep
#include "impl/tensor_math.hpp" // IWYU pragma: keep

   