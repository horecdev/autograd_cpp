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
        std::shared_ptr<std::vector<T>> m_data;
        size_t m_version;
    };

    template <typename T>
    class Tensor {
        private:
            std::vector<size_t> m_shape;
            std::vector<size_t> m_strides;
            size_t m_offset;
            std::shared_ptr<std::vector<T>> m_data;
        
        public:
            // LIFECYCLE 
            Tensor();
            Tensor(std::vector<size_t> shape);
            Tensor(std::vector<size_t> shape, std::vector<size_t> strides, size_t offset, std::shared_ptr<std::vector<T>> data);
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

            Tensor& operator+=(const Tensor& other);
            Tensor operator+(const Tensor& other) const;

            Tensor& operator-=(const Tensor& other);
            Tensor operator-(const Tensor& other) const;

            Tensor& operator*=(const Tensor& other);
            Tensor operator*(const Tensor& other) const;

            Tensor& operator/=(const Tensor& other);
            Tensor operator/(const Tensor& other) const;
    };
} 

#include "impl/tensor_lifecycle.hpp" // IWYU pragma: keep
#include "impl/tensor_indexing.hpp" // IWYU pragma: keep
#include "impl/tensor_shape.hpp" // IWYU pragma: keep
#include "impl/tensor_math.hpp" // IWYU pragma: keep

   