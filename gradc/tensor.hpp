#pragma once
#include "node.hpp"
#include "graph_fwd.hpp" // IWYU pragma: keep

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <ostream>
#include <utility>
#include <vector>

namespace gradc {

    struct Placeholder{};
    inline constexpr Placeholder _ = Placeholder(); // inline doesnt confuse linker if its #included in 2+ files.

    struct LazyTag{};
    inline constexpr LazyTag lazy = LazyTag();

    template <typename T>
    class Node;

    struct PrintOptions;

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

        Storage() : m_data(std::vector<T>{}), m_version(0) {}

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
            //std::cout << "Storage Destroyed" << std::endl;
        }
    };

    template <typename T>
    struct TensorState {
        std::shared_ptr<Storage<T>> m_storage; // nodes can share just storage
        std::shared_ptr<Node<T>> m_realize_op; // nodes can share operation (if aliases of same variable)

        TensorState() : m_storage(std::make_shared<Storage<T>>()), m_realize_op(nullptr) {}
        
        TensorState(std::vector<T>&& data) : m_storage(std::make_shared<Storage<T>>(std::move(data))), m_realize_op(nullptr) {}

        TensorState(std::shared_ptr<Storage<T>> storage) : m_storage(std::move(storage)), m_realize_op(nullptr) {} // copy tensor storage, set m_r_op to be nothing

        TensorState(const TensorState& other) : m_storage(other.m_storage), m_realize_op(other.m_realize_op) {}
        TensorState(TensorState&& other) : m_storage(std::move(other.m_storage)), m_realize_op(std::move(m_realize_op)) {}

        TensorState& operator=(const TensorState& other) {
            if (this != &other) {
                m_storage = other.m_storage;
                m_realize_op = other.m_realize_op;
            }
            return *this;
        }
        TensorState& operator=(TensorState&& other) {
            if (this != &other) {
                m_storage = std::move(other.m_storage);
                m_realize_op = std::move(other.m_realize_op);
            }
            return *this;
        } 
    };

    template <typename T>
    class Tensor {
        private:
            std::vector<size_t> m_shape;
            std::vector<size_t> m_strides;
            size_t m_offset;
            std::shared_ptr<TensorState<T>> m_state;
            bool m_requires_grad;

        public:
            void realize() {
                if (m_state->m_realize_op != nullptr) { // is there an m_op? if so, execute it
                    Tensor computed_result = m_state->m_realize_op->realize();
                    if (m_state->m_storage != computed_result.m_state->m_storage) {
                        m_state->m_storage->m_data = std::move(computed_result.m_state->m_storage->m_data);
                    }
                }

                m_state->m_op = nullptr; // so we dont realize() twice (reflected across multiple aliases)
            }

            // LIFECYCLE 
            Tensor();
            Tensor(std::vector<size_t> shape);
            Tensor(std::vector<size_t> shape, bool requires_grad, LazyTag);
            Tensor(std::vector<size_t> shape, std::vector<size_t> strides, size_t offset, std::shared_ptr<Storage<T>> storage, bool requires_grad);
            Tensor(std::vector<size_t> shape, std::vector<size_t> strides, size_t offset, std::shared_ptr<TensorState<T>> state, bool requires_grad);
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

            const bool requires_grad() const {
                return m_requires_grad;
            }

            // SETTERS
            void set_data(std::vector<T> data) {
                this->m_state->m_storage->m_data = data; // copy vector
            }
            void set_requires_grad(bool value) {
                m_requires_grad = value;
            }

            // SHAPES
            bool is_contiguous() const;
            Tensor contiguous() const;
            Tensor transpose(const size_t dim0, const size_t dim1) const;
            Tensor permute(const std::vector<int64_t>& axes) const;
            Tensor reshape(const std::vector<int64_t>& target_shape) const;

            // MATH
            template <typename U, typename Func> friend void apply_in_place(Tensor<U>& left, const Tensor<U>& right, Func op);
            template <typename U, typename Func> friend Tensor<U> apply_out_of_place(const Tensor<U>& left, const Tensor<U>& right, const std::vector<size_t>& target_shape, Func op);

            template <typename U> friend Tensor<U> operator+(Tensor<U> left, Tensor<U> right); // we befriend whole family of functions named operator+. 
            template <typename U> friend Tensor<U> operator*(Tensor<U> left, Tensor<U> right); // It operates on type U and U can be virtually anything

            template <typename U> friend Tensor<U>& operator+=(Tensor<U>& main, Tensor<U> other);
            template <typename U> friend Tensor<U>& operator*=(Tensor<U>& main, Tensor<U> other);

            // UTILS
            template <typename U> friend Tensor<U> lobotomized_broadcast(const Tensor<U>& source, const std::vector<size_t>& target_shape);
            template <typename U> friend Tensor<U> lobotomized_contiguous(const Tensor<U>& source);

            template <typename U> friend std::ostream& print_tensor(std::ostream& stream, const Tensor<U>& source, PrintOptions opts);
            template <typename U> friend void print_dim(std::ostream& stream, const Tensor<U>& source, const PrintOptions& opts, size_t current_dim, size_t base_offset, bool is_last);
    };
} 

   