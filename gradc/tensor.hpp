#pragma once

#include <iostream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <vector>
#include <array>

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

    template<typename T>
    class Tensor {
        private:
            std::vector<size_t> m_shape;
            std::vector<size_t> m_strides;
            size_t m_offset;
            std::shared_ptr<std::vector<T>> m_data;
        
        public:
            // LIFECYCLE 
            Tensor(std::vector<size_t> shape) 
                // can pass integer as m_strides, because it implicitly constructs a std::vector by just variable(arguments)
                : m_shape(std::move(shape)), m_strides(m_shape.size()), m_offset(0) {
                    if (m_shape.size() == 0) { // a scalar (0-dimensional)
                        m_data = std::make_shared<std::vector<T>>(1);
                    }
                    else {
                        m_strides[m_shape.size() - 1] = 1; 
                        for (size_t i = m_shape.size() - 1; i > 0; --i) {
                            m_strides[i - 1] = m_shape[i] * m_strides[i];
                        }
                        m_data = std::make_shared<std::vector<T>>(m_shape[0] * m_strides[0]);
                        // at first shared_ptr points to null. We gotta create it with a size.
                    }
                }

            Tensor(std::vector<size_t> shape, std::vector<size_t> strides, size_t offset, std::shared_ptr<std::vector<T>> data) // backdoor
                : m_shape(std::move(shape)), m_strides(std::move(strides)), m_offset(offset), m_data(std::move(data)) {} 

            ~Tensor() {
                std::cout << "Destructor called" << std::endl;
            }

            Tensor(const Tensor& source) 
                : m_shape(source.m_shape), m_strides(source.m_strides), m_offset(source.m_offset), m_data(source.m_data) {} // copy constructor (shallow copy) [Tensor b = a]
                // boosts ref count by 1 (shallow copy)

            Tensor(Tensor&& source) // move constructor [Tensor c = a + b]
                : m_shape(std::move(source.m_shape)), m_strides(std::move(source.m_strides)), m_offset(source.m_offset), m_data(std::move(source.m_data)) {}

            Tensor& operator=(const Tensor& source) { // copy assignment operator [c = b]
                if (this != &source) {
                    m_shape = source.m_shape;
                    m_strides = source.m_strides;
                    m_offset = source.m_offset;
                    m_data = source.m_data; // all are copied. This line increments ref count by 1, and decrements the prev ref count.
                    // we dont have to manually delete everything because its std::vector and std::shared_ptr. Smart classes.
                    // SHALLOW COPY
                }
                return *this;
            }

            Tensor& operator=(Tensor&& source) { // move assignment operator [a = b + c]
                if (this != &source) { // [a = transpose(a)], and transpose modifies it in-place, then it would trigger
                    m_shape = std::move(source.m_shape);
                    m_strides = std::move(source.m_strides);
                    m_offset = source.m_offset;
                    m_data = std::move(source.m_data);
                }   
                return *this;
            }

            Tensor clone() const {
                // *m_data is forwarded straight into vector construction zone. Sees a vector entering (*m_data is a vector) and triggers a totally new vector construction (new heap memory)
                auto ptr = std::make_shared<std::vector<T>>(*m_data); // shared_ptr pointing to a new vector with copied data. (ref_count = 1)
                return Tensor(m_shape, m_strides, m_offset, std::move(ptr));
            }
            // END OF LIFECYCLE REGION
            
            // INDEXING
            template<typename... Args> // ... attached to typename means: create a bucket of types and name it Args.
            Tensor operator[](Args... args) const { // ... next to the type bucket, before args: Look element-wise at both. Put type inside type bucket, argument inside args.
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
                return Tensor(std::move(new_shape), std::move(new_strides), new_offset, m_data); // backdoor construct shallow copy
            }

            T item() {
                if (m_shape.size() == 0) {
                    return m_data[m_offset];
                }
                else {
                    throw std::runtime_error(".item() can be called only on 0-dimensional tensors.");
                }
            }

            // GETTERS

            const std::vector<size_t>& shape() const {
                return m_shape;
            }

            const std::vector<size_t>& strides() const {
                return m_strides;
            }

            bool is_contiguous() const {
                if (m_strides[m_strides.size() - 1] != 1) {
                    return false;
                }
                for (size_t i = m_shape.size() - 1; i > 0; --i) {
                    if (m_strides[i - 1] != m_shape[i] * m_strides[i]) {
                        return false;
                    }
                }
                return true;
            }
            
            Tensor contiguous() const {
                if (m_shape.empty()) {
                    Tensor scalar_tensor = Tensor(std::vector<size_t>{});
                    (*scalar_tensor.m_data)[0] = (*m_data)[m_offset];
                }
                Tensor new_contiguous = Tensor(m_shape); // already right size and right contiguous strides
                size_t n_dims = m_shape.size();
                std::vector<size_t> odometer(n_dims); // zeroed out
                size_t contiguous_idx = 0;
                while (odometer[0] < m_shape[0]) {
                    size_t strided_idx = m_offset;
                    for (size_t i = 0; i < n_dims; ++i) {
                        strided_idx += odometer[i] * m_strides[i];
                    }
                    (*new_contiguous.m_data)[contiguous_idx] = (*m_data)[strided_idx];
                    ++contiguous_idx;
                    ++odometer[n_dims - 1];
                    size_t i = n_dims - 1;
                    while ((odometer[i] == m_shape[i]) && i > 0) {
                        odometer[i] = 0;
                        ++odometer[i - 1];
                        --i;
                    }
                }
                return new_contiguous;
            }

            Tensor transpose(const size_t dim0, const size_t dim1) const {
                std::vector<size_t> new_shape = m_shape;
                std::vector<size_t> new_strides = m_strides;
                size_t temp = new_shape[dim0];
                new_shape[dim0] = new_shape[dim1];
                new_shape[dim1] = temp;
                temp = new_strides[dim0];
                new_strides[dim0] = new_strides[dim1];
                new_strides[dim1] = temp;

                return Tensor(std::move(new_shape), std::move(new_strides), m_offset, m_data);
            }

            Tensor permute(const std::vector<int64_t>& axes) const {
                size_t n_dim = m_shape.size();
                std::cout << axes.size() << " " << n_dim << std::endl;
                if (axes.size() != n_dim) {
                    throw std::runtime_error("permute() axes list size must match shape list size.");
                }
                std::vector<size_t> new_shape = std::vector<size_t>(n_dim);
                std::vector<size_t> new_strides = std::vector<size_t>(n_dim);
                for (size_t target_ax = 0; target_ax < n_dim; ++target_ax) {
                    int64_t src_ax = axes[target_ax];
                    if (src_ax < 0) {
                        src_ax = n_dim + src_ax;
                    }
                    new_shape[target_ax] = m_shape[src_ax];
                    new_strides[target_ax] = m_strides[src_ax];
                }
                return Tensor(std::move(new_shape), std::move(new_strides), m_offset, m_data);
            }

            Tensor reshape(const std::vector<int64_t>& target_shape) const {
                std::vector<size_t> new_shape = std::vector<size_t>(target_shape.size());
                std::vector<size_t> new_strides = std::vector<size_t>(target_shape.size());
                size_t total_volume = 1;
                size_t running_volume = 1;
                int64_t neg_one_idx = -1;

                for (size_t i = 0; i < m_shape.size(); ++i) {
                    total_volume *= m_shape[i];
                }
                
                for (size_t i = 0; i < target_shape.size(); ++i) {
                    if (target_shape[i] == -1 && neg_one_idx == -1) {
                        neg_one_idx = i;
                    }
                    else if (target_shape[i] == -1 && neg_one_idx != -1) {
                        throw std::runtime_error("Cannot .reshape() with two or more unknown dimensions.");
                    }
                    else {
                        new_shape[i] = target_shape[i];
                        running_volume *= target_shape[i];
                    }
                }

                size_t unknown_dim;
                if (total_volume % running_volume != 0) {
                    throw std::runtime_error("Invalid reshape parameters.");
                }
                else {
                    unknown_dim = total_volume / running_volume;
                }

                if (neg_one_idx != -1) {
                    new_shape[neg_one_idx] = unknown_dim;
                }

                new_strides[target_shape.size() - 1] = 1;
                for (size_t i = target_shape.size() - 1; i > 0; --i) {
                    new_strides[i - 1] = new_shape[i] * new_strides[i];
                }
                
                if (this->is_contiguous()) { // cannot reshape a non-contiguous tensor.
                    return Tensor(std::move(new_shape), std::move(new_strides), m_offset, m_data);
                }

                Tensor reshaped_tensor = this->contiguous(); // contiguous is shape-sensitive (if sliced then only slice is made contiguous)
                reshaped_tensor.m_shape = std::move(new_shape);
                reshaped_tensor.m_strides = std::move(new_strides);

                return reshaped_tensor;
            }

        Tensor broadcast_to(const std::vector<size_t>& target_shape) const {
            int64_t n_dim_orig = static_cast<int64_t>(m_shape.size());
            int64_t n_dim_target = static_cast<int64_t>(target_shape.size());
            std::vector<size_t> new_shape = std::vector<size_t>(n_dim_target);
            std::vector<size_t> new_strides = std::vector<size_t>(n_dim_target);

            // align to the right
            for (int64_t i = 0; i < n_dim_orig; ++i) {
                new_shape[i + (n_dim_target - n_dim_orig)] = m_shape[i];
                new_strides[i + (n_dim_target - n_dim_orig)] = m_strides[i];
            }
            for (int64_t i = n_dim_target - n_dim_orig - 1; i >= 0; --i) { // fill leftovers (dont even have to check later)
                new_shape[i] = target_shape[i];
                new_strides[i] = 0;
            }

            for (size_t i = (n_dim_target - n_dim_orig); i < n_dim_target; ++i) {
                if (target_shape[i] == new_shape[i]) {
                    // literally leave everything as is
                }
                else if (new_shape[i] == 1) {
                    new_shape[i] = target_shape[i];
                    new_strides[i] = 0;
                }
                else {
                    throw std::runtime_error("Violated broadcasting rules in .broadcast_to().");
                }
            }

            return Tensor(std::move(new_shape), std::move(new_strides), m_offset, m_data);
        }

        std::vector<size_t> infer_broadcast(const std::vector<size_t>& a, const std::vector<size_t>& b) const {
            size_t size_a = a.size();
            size_t size_b = b.size();
            
        }
    };
}
