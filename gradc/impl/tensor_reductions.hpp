#include "../tensor.hpp"
#include "../graph.hpp"

namespace gradc {
    template <typename T>
    Tensor<T> Tensor<T>::sum(std::vector<int64_t>& red_axes, bool keepdims) {
        ReductionMetadata reduction_metadata = infer_reduction_metadata(m_shape, red_axes, keepdims);
        Tensor result = Tensor(std::move(reduction_metadata.result_shape), m_requires_grad, lazy);
        result.m_state->m_realize_op = std::make_unique<SumNode<T>>(*this, reduction_metadata);
    }
}