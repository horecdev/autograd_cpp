#pragma once
#include <stdexcept>
#include <vector>

namespace gradc {

    inline std::vector<int64_t> infer_broadcast(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
        // works for scalars
        const int64_t size_a = std::ssize(a);
        const int64_t size_b = std::ssize(b);

        std::vector<int64_t> inferred_shape(std::max(size_a, size_b));
        
        if (size_a >= size_b) {
            for (int64_t b_idx = 0; b_idx < size_b; ++b_idx) {
                int64_t a_idx = b_idx + (size_a - size_b);
                int64_t shape_a = a[a_idx];
                int64_t shape_b = b[b_idx];
                if (shape_a == shape_b) {
                    inferred_shape[a_idx] = shape_a;
                }
                else if (shape_a == 1 || shape_b == 1) {
                    inferred_shape[a_idx] = std::max(shape_a, shape_b);
                }
                else {
                    throw std::runtime_error("Incompatible shapes for broadcasting.");
                }
            }
            for (int64_t a_idx = 0; a_idx < (size_a - size_b); ++a_idx) {
                inferred_shape[a_idx] = a[a_idx];
            }
        }
        else if (size_a < size_b) {
            for (int64_t a_idx = 0; a_idx < size_a; ++a_idx) {
                int64_t b_idx = a_idx + (size_b - size_a);
                int64_t shape_a = a[a_idx];
                int64_t shape_b = b[b_idx];
                if (shape_a == shape_b) {
                    inferred_shape[b_idx] = shape_b;
                }
                else if (shape_a == 1 || shape_b == 1) {
                    inferred_shape[b_idx] = std::max(shape_a, shape_b);
                }
                else {
                    throw std::runtime_error("Incompatible shapes for broadcasting.");
                }
            }
            for (int64_t b_idx = 0; b_idx < (size_b - size_a); ++b_idx) {
                inferred_shape[b_idx] = b[b_idx];
            }
        }

        return inferred_shape;  
    }   

    inline bool can_broadcast(const std::vector<int64_t>& src, const std::vector<int64_t>& target) {
        int64_t size_src = std::ssize(src);
        int64_t size_target = std::ssize(target);
        if (size_src > size_target) {
            return false;
        }

        for (int64_t src_idx = 0; src_idx < size_src; ++src_idx) {
            int64_t target_idx = src_idx + (size_target - size_src);
            if (src[src_idx] != target[target_idx] && src[src_idx] != 1) {
                return false;
            }
        }   
        return true;
    }


}