#include "../tensor.hpp"
#include "tensor_utils.hpp"

namespace gradc {
    template <typename T>
    Tensor<T> Tensor<T>::sum(std::vector<int64_t>& red_axes, bool keepdims) {
        const int64_t n_dim = static_cast<int64_t>(this->m_shape.size());
        std::vector<int64_t> positive_red_axes;
        positive_red_axes.reserve(n_dim);
        for (const int64_t x : red_axes) {
            positive_red_axes.push_back(normalize_axis(x, n_dim));
        }

        std::vector<size_t> temp_shape = m_shape;
        std::vector<size_t> temp_strides = std::vector<size_t>(n_dim);
        std::vector<size_t> result_shape;
        std::vector<size_t> result_strides;

        // first calculate output shape. Then calculate temporary values that allow operations
        for (const size_t ax : positive_red_axes) {
            temp_shape[ax] = 1; // later just copy temp as result_shape and retain/remove axes
        }

        result_shape = temp_shape;

        temp_strides[n_dim - 1] = 1;
        for (int64_t i = n_dim - 1; i > 0; --i) {
            temp_strides[i - 1] = temp_shape[i] * temp_strides[i];
        }

        result_strides = temp_strides; // based on keepdims we will retain/remove certain indices

        for (const size_t ax : positive_red_axes) {
            temp_strides[ax] = 0;
        }

        // on [10, 5, 2].sum(1) get shape[10, 1, 2] and strides[2, 0, 1]
        // we need new storage to store results somewhere
        size_t new_size = calculate_dim_product(temp_shape);
        std::shared_ptr<Storage<T>> new_storage = std::make_shared<Storage<T>>(std::vector<T>(new_size));

        std::vector<size_t> odometer(n_dim, 0);
        while (odometer[0] < m_shape[0]) {
            size_t in_strided_idx = m_offset; 
            size_t out_strided_idx = 0;

            for (size_t i = 0; i < n_dim; ++i) {
                in_strided_idx += odometer[i] * m_strides[i];
                out_strided_idx += odometer[i] * temp_strides[i];
            }
            (new_storage->m_data)[out_strided_idx] += (m_state->m_storage->m_data)[in_strided_idx]; 
            ++odometer[n_dim - 1];
            size_t i = n_dim - 1;
            while ((odometer[i] == m_shape[i]) && i > 0) {
                odometer[i] = 0;
                ++odometer[i - 1];
                --i;
            }
        }

        // now storage is populated with aggregated stuff. Can proceed to keeping/removing dims
        if (!keepdims) {
            std::vector<bool> axes_to_keep = std::vector<bool>(n_dim, true);
            size_t new_size = n_dim;

            for (const size_t ax : positive_red_axes) {
                axes_to_keep[ax] = false;
                --new_size;
            }

            std::vector<size_t> collapsed_result_shape = std::vector<size_t>{};
            std::vector<size_t> collapsed_result_strides = std::vector<size_t>{};
            collapsed_result_shape.reserve(new_size);
            collapsed_result_strides.reserve(new_size);

            for (size_t i = 0; i < n_dim; ++i) {
                if (axes_to_keep[i]) {
                    collapsed_result_shape.push_back(result_shape[i]);
                    collapsed_result_strides.push_back(result_strides[i]);
                }
            }

            Tensor<T> result = Tensor<T> // finish
        }
        else {
            return;
        }

    }
}