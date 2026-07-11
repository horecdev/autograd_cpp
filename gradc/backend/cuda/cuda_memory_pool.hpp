#pragma once

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <unordered_map>

class CUDAMemPool {
    private:
        std::unordered_map<int64_t, std::vector<void*>> m_free_blocks;
        CUDAMemPool() = default; 
    public:
        CUDAMemPool(const CUDAMemPool&) = delete;
        CUDAMemPool& operator=(const CUDAMemPool) = delete;

        static CUDAMemPool& get() {
            static CUDAMemPool inst;
            return inst;
        }

        void* allocate(int64_t bytes) {
            std::vector<void*>& blocks = m_free_blocks[bytes];
            if (!blocks.empty()) {
                void* ptr = blocks.back();
                blocks.pop_back();
                return ptr;
            }
            else {
                void* ptr = nullptr;
                cudaError_t err = cudaMalloc(&ptr, bytes);

                if (err != cudaSuccess) {
                    clear();
                    void* ptr = nullptr;
                    cudaError_t err = cudaMalloc(&ptr, bytes);
                    if (err != cudaSuccess) {
                        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
                    }
                }
            }
        }

        void free(int64_t bytes, void* ptr) {
            if (ptr != nullptr) {
                m_free_blocks[bytes].push_back(ptr);
            }
        }

        void clear() {
            for (auto& [size, block] : m_free_blocks) {
                for (void* ptr : block) {
                    cudaFree(&ptr);
                }
            }
            m_free_blocks.clear();
        }
};