#pragma once

#include "../tensor.hpp"

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

    template <typename T>
    Tensor<T> unbroadcast_grad(const Tensor<T>& raw_grad, const std::vector<int64_t>& orig_shape) {
        std::vector<int64_t> broadcast_axes = find_broadcast_axes(raw_grad.m_shape, orig_shape);

        if (broadcast_axes.empty()) {
            return raw_grad;
        }

        ReductionMetadata red_meta = infer_reduction_metadata(raw_grad.m_shape, broadcast_axes, false);
        Tensor<T> reduced = apply_reduction_operation(raw_grad, red_meta, T(), [](T a, T b){return a + b;});

        return Tensor<T>(orig_shape, std::move(reduced.m_state->m_storage)); // contiguous tensor (strides must be generated and shape must be kept as the one of parent)
        // We keepdims=false so [2, 1] broadcast into [2, 5] turns into [2]. To fix that, we just use the same shape as the original (parent).
        // It works because reduction operation return a contiguous tensor.
    }

    template <typename T, typename U>
    auto promote_to_common(Tensor<T> left, Tensor<U> right) {
        // .template is a promise to the compiler that cast is a template (it doesnt know what T is during first read, 
        // and since its dependent - inside Tensor class, not on its own - it assumes its a variable by default. .template forces it to look at it as a template.)
        using PromotedT = std::common_type_t<T, U>;

        Tensor<PromotedT> p_left;
        Tensor<PromotedT> p_right;

        if constexpr (std::is_same_v<T, PromotedT>) {
            p_left = std::move(left);
        }
        else {
            p_left = left.template cast<PromotedT>();
        }

        if constexpr (std::is_same_v<U, PromotedT>) {
            p_right = std::move(right);
        }
        else {
            p_right = right.template cast<PromotedT>();
        }

        return std::make_pair(std::move(p_left), std::move(p_right));
    }

    
}