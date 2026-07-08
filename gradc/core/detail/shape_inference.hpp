#pragma once

#include "../types.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace gradc {

    inline int64_t normalize_axis(int64_t axis, int64_t n_dim) {
        if (axis < 0) {axis += n_dim;}
        if (axis < 0 || axis >= n_dim) {
            throw std::out_of_range("Axis out of bounds");
        }
        return axis;
    }

    inline std::vector<int64_t> normalize_axes_vector(const std::vector<int64_t>& axes, int64_t n_dim) {
        std::vector<int64_t> normalized_axes = std::vector<int64_t>(std::ssize(axes), -1);
        for (int64_t i = 0; i < n_dim; ++i) {
            normalized_axes[i] = normalize_axis(axes[i], n_dim);
        }

        return normalized_axes;
    }

    inline int64_t calculate_dim_product(const std::vector<int64_t>& shape) {
        int64_t dim_product = 1;
        for (int64_t i = 0; i < std::ssize(shape); ++i) {
            dim_product *= shape[i];
        }

        return dim_product;
    }

    
    inline std::vector<int64_t> find_broadcast_axes(const std::vector<int64_t>& broadcast_shape, const std::vector<int64_t>& initial_shape) {
        std::vector<int64_t> found_axes;
        found_axes.reserve(std::ssize(broadcast_shape));
        for (int64_t i = 0; i < std::ssize(broadcast_shape) - std::ssize(initial_shape); ++i) {
            found_axes.push_back(i);
        }
        for (int64_t init_idx = 0; init_idx < std::ssize(initial_shape); ++init_idx) {
            int64_t res_idx = init_idx + std::ssize(broadcast_shape) - std::ssize(initial_shape);
            if (broadcast_shape[res_idx] != initial_shape[init_idx]) {
                found_axes.push_back(res_idx);
            }
        }

        return found_axes;
    }

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

    inline ReductionMetadata infer_reduction_metadata(const std::vector<int64_t>& source_shape, const std::vector<int64_t>& red_axes, bool keepdims) {
        const int64_t n_dim = std::ssize(source_shape);
        std::vector<int64_t> positive_red_axes;
        positive_red_axes.reserve(n_dim);
        for (const int64_t x : red_axes) {
            positive_red_axes.push_back(normalize_axis(x, n_dim));
        }

        std::vector<int64_t> temp_shape = source_shape;
        std::vector<int64_t> temp_strides = std::vector<int64_t>(n_dim);
        std::vector<int64_t> result_shape;
        std::vector<int64_t> result_strides;
        int64_t result_vol = 1;
        int64_t reduced_vol = 1;

        // first calculate output shape. Then calculate temporary values that allow operations
        for (const int64_t ax : positive_red_axes) {
            reduced_vol *= temp_shape[ax];
            temp_shape[ax] = 1; // later just copy temp as result_shape and retain/remove axes
        }

        result_shape = temp_shape;
        result_vol = calculate_dim_product(result_shape);
        

        temp_strides[n_dim - 1] = 1;
        for (int64_t i = n_dim - 1; i > 0; --i) {
            temp_strides[i - 1] = temp_shape[i] * temp_strides[i];
        }

        result_strides = temp_strides; // based on keepdims we will retain/remove certain indices

        for (const int64_t ax : positive_red_axes) {
            temp_strides[ax] = 0;
        }

        if (!keepdims) {
            std::vector<bool> axes_to_keep = std::vector<bool>(n_dim, true);
            int64_t new_size = n_dim;

            for (const int64_t ax : positive_red_axes) {
                axes_to_keep[ax] = false;
                --new_size;
            }

            std::vector<int64_t> collapsed_result_shape = std::vector<int64_t>{};
            std::vector<int64_t> collapsed_result_strides = std::vector<int64_t>{};
            collapsed_result_shape.reserve(new_size);
            collapsed_result_strides.reserve(new_size);

            for (int64_t i = 0; i < n_dim; ++i) {
                if (axes_to_keep[i]) {
                    collapsed_result_shape.push_back(result_shape[i]);
                    collapsed_result_strides.push_back(result_strides[i]);
                }
            }
            return ReductionMetadata(temp_shape, temp_strides, collapsed_result_shape, collapsed_result_strides, result_vol, reduced_vol);
        }

        else {
            return ReductionMetadata(temp_shape, temp_strides, result_shape, result_strides, result_vol, reduced_vol);
        }
        
    }


}