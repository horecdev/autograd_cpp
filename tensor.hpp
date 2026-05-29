#pragma once

#include <iostream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace gradc {
    struct Slice{};
    inline constexpr Slice _ = Slice();

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
            Tensor& operator[](Args... args) { // ... next to the type bucket, before args: Look element-wise at both. Put type inside type bucket, argument inside args.
                // The compiler sees you passed x: short, y: int, z: size_t, and automatically creates a template (short var_0, int var_1, size_t var_2)
                // For class templates: pass <float> etc. For funcion/operator templates you dont have to.
                if (sizeof...(args) != m_shape.size()) { // sizeof... literally means "count elements in the bucket". Its a built-in token.
                    throw std::out_of_range("Coordinate count does not match tensor dimensions.");
                }
                std::array<int64_t, sizeof...(args)> raw_coords = {static_cast<int64_t>(args)...}; // when you do [expression(args)...] it means: apply expression to every arg, and seperate it with comas.
                std::array<size_t, sizeof...(args)> coords;
                for (int i = 0; i < sizeof...(args); ++i) {
                    if (raw_coords[i] >= 0) {
                        coords[i] = static_cast<size_t>(raw_coords[i]);
                    }
                    else {
                        coords[i] = static_cast<size_t>(m_shape[i] + raw_coords[i]);
                    } // TODO: ADD STRUCT SLICEALL, CALCULATE OFFSETS, SHAPES AND STRIDES TO RETURN A TENSOR SHALLOW COPY
                }


            }
            // GETTERS

            const std::vector<size_t>& shape() const {
                return m_shape;
            }

            const std::vector<size_t>& strides() const {
                return m_strides;
            }



    };
}
