#pragma once

#include "../tensor.hpp"
#include "../../backend/dispatcher.hpp"
#include "shape_inference.hpp"

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace gradc {

    template <typename T>
    Tensor<T> unbroadcast_grad(const Tensor<T>& raw_grad, const std::vector<int64_t>& orig_shape) {
        std::vector<int64_t> broadcast_axes = find_broadcast_axes(raw_grad.m_shape, orig_shape);

        if (broadcast_axes.empty()) {
            return raw_grad;
        }

        ReductionMetadata red_meta = infer_reduction_metadata(raw_grad.m_shape, broadcast_axes, false);
        Tensor<T> reduced = Tensor<T>(orig_shape, raw_grad.device(), uninitialized);
        dispatch(raw_grad.device(), ReduceOp::Sum, T(), red_meta, reduced, raw_grad);
        return reduced; 

        // contiguous tensor (strides must be generated and shape must be kept as the one of parent)
        // We keepdims=false so [2, 1] broadcast into [2, 5] turns into [2]. To fix that, we just use the same shape as the original (parent).
        // It works because reduction operation returns a contiguous tensor. You can just fake dimensions of one, and since we dont know which one we reduced
        // and which one were here beforehand, just collapse both and artificially add them later.
        // From now on since shape/strides of reduced isnt used in reduction operation, its done at the start.
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