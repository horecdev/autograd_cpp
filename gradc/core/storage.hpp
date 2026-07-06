#pragma once

#include "types.hpp"

#include <stdexcept>

namespace gradc {
    template <typename T>
    struct Storage{
        T* m_data = nullptr; // ptr to start of chunk of memory
        int64_t m_size = 0;
        Device m_device;

        Storage() : m_data(nullptr), m_size(0), m_device(Device::CPU) {}

        Storage(int64_t size, T init_val = T(), Device device = Device::CPU, bool allocate = true, bool fill = true) : m_size(size), m_device(device) {
            int64_t bytes = size * sizeof(T);
                if (allocate) {
                    if (m_device == Device::CPU) {
                        int64_t aligned_bytes = ((bytes + 31) / 32) * 32; // must be 32 multiple
                        // by default _aligned_malloc returns void* (pointer to very first byte) so u cast it to T*
                        m_data = static_cast<T*>(_aligned_malloc(aligned_bytes, 32));
                        if (fill) {
                            std::fill(m_data, m_data + m_size, init_val); // m_data + m_size does pointer arithmetic (applies for sizeof)
                        }
                        
                    }
                    
                    else if (m_device == Device::CUDA) {
                        // some CUDA stuff later
                    }
                }
        }

        Storage(std::initializer_list<T> data, Device device = Device::CPU) : m_size(std::ssize(data)), m_device(device) {
            if (m_size < 1) {
                throw std::runtime_error("Storage constructed with empty initializer_list. Use a different constructor.");
            }

            if (m_device == Device::CPU) {
                int64_t bytes = m_size * sizeof(T);
                int64_t aligned_bytes = ((bytes + 31) / 32) * 32;
                m_data = static_cast<T*>(_aligned_malloc(aligned_bytes, 32));
                std::memcpy(m_data, data.begin(), data.size() * sizeof(T));
            }
        }

        T* data() const {
            return m_data;
        }

        int64_t size() const {
            return m_size;
        }

        Device device() const {
            return m_device;
        }

        ~Storage() {
            _aligned_free(m_data); // accepts void* but any pointer can implicitly convert to void*
        }

        Storage(const Storage&) = delete; // since we manually free memory copying should not exist.
        Storage& operator=(const Storage&) = delete; // we dont even copy storage anywhere so its cool
        
        Storage(Storage&& other) : m_data(std::move(other.m_data)), m_size(other.m_size), m_device(other.m_device) {}
        Storage& operator=(Storage&& other) {
            if (this != &other) {
                m_data = std::move(other.m_data);
                m_size = other.m_size;
                m_device = other.m_device;
            }
            return *this;
        } 
    };
}
