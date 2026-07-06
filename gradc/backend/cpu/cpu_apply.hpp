#include "../../core/tensor.hpp"

namespace gradc::cpu {

    template <typename T, typename Func>
    Tensor<T> apply_out_of_place(const Tensor<T>& left, const Tensor<T>& right, const std::vector<int64_t>& target_shape, Func op) {
        Device target_device = infer_assert_device(left, right);

        if (std::ssize(target_shape) == 0) { // op on 2 scalars
            Tensor<T> result = Tensor<T>(target_shape, target_device, uninitialized);
            (result.m_state->m_storage->m_data)[0] = op((left.m_state->m_storage->m_data)[left.m_offset], (right.m_state->m_storage->m_data)[right.m_offset]);
            return result;
        }

        const std::vector<int64_t>* left_strides = &left.m_strides;
        const std::vector<int64_t>* right_strides = &right.m_strides;
        int64_t left_offset = left.m_offset;
        int64_t right_offset = right.m_offset; 
        std::shared_ptr<Storage<T>> left_storage = left.m_state->m_storage; // broadcasting does not alter memory
        std::shared_ptr<Storage<T>> right_storage = right.m_state->m_storage;

        Tensor<T> broad_right;
        Tensor<T> broad_left;

        if (left.m_shape != target_shape) {
            broad_left = lobotomized_broadcast_view(left, target_shape);

            left_strides = &broad_left.m_strides;
            left_offset = broad_left.m_offset;
        }
        if (right.m_shape != target_shape) {
            broad_right = lobotomized_broadcast_view(right, target_shape);

            right_strides = &broad_right.m_strides;
            right_offset = broad_right.m_offset;
        }
        

        Tensor<T> result = Tensor<T>(target_shape, target_device, uninitialized);

        const int64_t n_dim = std::ssize(target_shape);
        std::vector<int64_t> odometer(n_dim, 0);
        int64_t contiguous_idx = 0;
        while (odometer[0] < target_shape[0]) {
            int64_t left_strided_idx = left_offset; 
            int64_t right_strided_idx = right_offset;

            for (int64_t i = 0; i < n_dim; ++i) {
                left_strided_idx += odometer[i] * (*left_strides)[i];
                right_strided_idx += odometer[i] * (*right_strides)[i];
            }
            (result.m_state->m_storage->m_data)[contiguous_idx] = op((left_storage->m_data)[left_strided_idx], (right_storage->m_data)[right_strided_idx]); // copied straight into CPU registers from RAM
            ++contiguous_idx;
            ++odometer[n_dim - 1];
            int64_t i = n_dim - 1;
            while ((odometer[i] == target_shape[i]) && i > 0) {
                odometer[i] = 0;
                ++odometer[i - 1];
                --i;
            }
        }

        return result;
    }

    template <typename T1, typename T2, typename Func> // two types so you can cast one to another in lambda
    void apply_in_place(Tensor<T1>& left, const Tensor<T2>& right, Func op) { 
        infer_assert_device(left, right); // throws (returns device but its not needed here)

        if (left.m_shape.empty() && right.m_shape.empty()) {
            op(left.m_state->m_storage->m_data[left.m_offset], right.m_state->m_storage->m_data[right.m_offset]);
            return;
        }
        // Idea behind all the variables - track the right variables instead of copying vectors on the heap inside copied Tensors.
        // (you would have to either create a tensor or copy it to hold a variable (or worse - write main logic twice) - why not ONLY create if needed?)
        const std::vector<int64_t>* right_strides; // promise not to modify data, (can reassign the pointer)
        int64_t right_offset;

        Tensor<T2> broad_right;
        if (left.m_shape == right.m_shape) {
            right_strides = &right.m_strides;
            right_offset = right.m_offset;
        }
        else {
            broad_right = lobotomized_broadcast_view(right, left.m_shape);

            right_strides = &broad_right.m_strides;
            right_offset = broad_right.m_offset;
        }

        const int64_t n_dim = std::ssize(left.m_shape);
        std::vector<int64_t> odometer(n_dim, 0);
        while (odometer[0] < left.m_shape[0]) {
            int64_t left_strided_idx = left.m_offset; 
            int64_t right_strided_idx = right_offset;

            for (int64_t i = 0; i < n_dim; ++i) {
                left_strided_idx += odometer[i] * left.m_strides[i];
                right_strided_idx += odometer[i] * (*right_strides)[i];
            }
            op((left.m_state->m_storage->m_data)[left_strided_idx], (right.m_state->m_storage->m_data)[right_strided_idx]);
            ++odometer[n_dim - 1];
            int64_t i = n_dim - 1;
            while ((odometer[i] == left.m_shape[i]) && i > 0) {
                odometer[i] = 0;
                ++odometer[i - 1];
                --i;
            }
        }
    }

    template <typename T, typename Func>
    Tensor<T> apply_unary_out_of_place(const Tensor<T>& source, Func op) {
        if (source.m_shape.empty()) {
            Tensor<T> result = Tensor<T>(source.m_shape, source.device(), uninitialized); // scalar tensor
            (result.m_state->m_storage->m_data)[0] = op((source.m_state->m_storage->m_data)[source.m_offset]);
            return result;
        }
        
        if (source.is_contiguous()) {
            // fast path
        }
        
        Tensor<T> result = Tensor<T>(source.m_shape, source.device(), uninitialized);

        const int64_t n_dim = std::ssize(source.m_shape);
        std::vector<int64_t> odometer(n_dim, 0);
        int64_t contiguous_idx = 0;
        while (odometer[0] < source.m_shape[0]) {
            int64_t strided_idx = source.m_offset;

            for (int64_t i = 0; i < n_dim; ++i) {
                strided_idx += odometer[i] * (source.m_strides)[i];
            }
            (result.m_state->m_storage->m_data)[contiguous_idx] = op((source.m_state->m_storage->m_data)[strided_idx]); 
            ++contiguous_idx;
            ++odometer[n_dim - 1];
            int64_t i = n_dim - 1;
            while ((odometer[i] == source.m_shape[i]) && i > 0) {
                odometer[i] = 0;
                ++odometer[i - 1];
                --i;
            }
        }

        return result;
    }

    template <typename T, typename Func>
    inline Tensor<T> apply_reduction_operation(const Tensor<T>& source, const ReductionMetadata& reduction_metadata, T init_value, Func op) {
        const int64_t n_dim = std::ssize(source.m_shape);
        if (n_dim == 0) {
            throw std::runtime_error("Tried reducing a 0-Dimensional Tensor.");
        }

        Tensor<T> result = Tensor<T>(reduction_metadata.result_shape, init_value, source.device());

        std::vector<int64_t> odometer(n_dim, 0);
        while (odometer[0] < source.m_shape[0]) {
            int64_t in_strided_idx = source.m_offset; 
            int64_t out_strided_idx = 0;

            for (int64_t i = 0; i < n_dim; ++i) {
                in_strided_idx += odometer[i] * source.m_strides[i];
                out_strided_idx += odometer[i] * reduction_metadata.temp_strides[i];
            }
            (result.m_state->m_storage->m_data)[out_strided_idx] = op((result.m_state->m_storage->m_data)[out_strided_idx], (source.m_state->m_storage->m_data)[in_strided_idx]);
            ++odometer[n_dim - 1];
            int64_t i = n_dim - 1;
            while ((odometer[i] == source.m_shape[i]) && i > 0) {
                odometer[i] = 0;
                ++odometer[i - 1];
                --i;
            }
        }

        return result;
    }

}