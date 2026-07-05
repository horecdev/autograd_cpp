#include "../tensor.hpp"
#include "../graph.hpp"
#include "tensor_utils.hpp"

namespace gradc {
    template <typename T>
    Tensor<T> Tensor<T>::sum(const std::vector<int64_t>& red_axes, bool keepdims) const {
        ReductionMetadata reduction_metadata = infer_reduction_metadata(m_shape, red_axes, keepdims);
        Tensor result = Tensor(reduction_metadata.result_shape, m_requires_grad, lazy, this->device());
        result.m_state->m_creation_op = std::make_unique<SumNode<T>>(*this, reduction_metadata);

        return result;
    }

    template <typename T>
    template <typename OutT>
    Tensor<OutT> Tensor<T>::mean(const std::vector<int64_t>& red_axes, bool keepdims) const {
        Tensor<OutT> promoted_self = this->template cast<OutT>(); // first: cast source into right type. Then just add MeanNode.
        ReductionMetadata reduction_metadata = infer_reduction_metadata(promoted_self.m_shape, red_axes, keepdims);
        Tensor<OutT> result = Tensor<OutT>(reduction_metadata.result_shape, m_requires_grad, lazy, this->device());
        result.m_state->m_creation_op = std::make_unique<MeanNode<OutT>>(std::move(promoted_self), reduction_metadata);

        return result;
    }
}