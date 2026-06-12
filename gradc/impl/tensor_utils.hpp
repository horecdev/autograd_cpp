#pragma once
#include "../tensor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gradc {
    inline std::vector<size_t> infer_broadcast(const std::vector<size_t>& a, const std::vector<size_t>& b) {
        // works for scalars
        size_t size_a = a.size();
        size_t size_b = b.size();

        std::vector<size_t> inferred_shape(std::max(size_a, size_b));
        
        if (size_a >= size_b) {
            for (size_t b_idx = 0; b_idx < size_b; ++b_idx) {
                size_t a_idx = b_idx + (size_a - size_b);
                size_t shape_a = a[a_idx];
                size_t shape_b = b[b_idx];
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
            for (size_t a_idx = 0; a_idx < (size_a - size_b); ++a_idx) {
                inferred_shape[a_idx] = a[a_idx];
            }
        }
        else if (size_a < size_b) {
            for (size_t a_idx = 0; a_idx < size_a; ++a_idx) {
                size_t b_idx = a_idx + (size_b - size_a);
                size_t shape_a = a[a_idx];
                size_t shape_b = b[b_idx];
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
            for (size_t b_idx = 0; b_idx < (size_b - size_a); ++b_idx) {
                inferred_shape[b_idx] = b[b_idx];
            }
        }

        return inferred_shape;  
    }   

    inline bool can_broadcast(const std::vector<size_t>& src, const std::vector<size_t>& target) {
        size_t size_src = src.size();
        size_t size_target = target.size();
        if (size_src > size_target) {
            return false;
        }

        for (size_t src_idx = 0; src_idx < size_src; ++src_idx) {
            size_t target_idx = src_idx + (size_target - size_src);
            if (src[src_idx] != target[target_idx] && src[src_idx] != 1) {
                return false;
            }
        }   
        return true;
    }

    template <typename T>
    Tensor<T> lobotomized_broadcast(const Tensor<T>& source, const std::vector<size_t>& target_shape) {
        const int64_t n_dim_orig = static_cast<int64_t>(source.m_shape.size());
        const int64_t n_dim_target = static_cast<int64_t>(target_shape.size());
        std::vector<size_t> new_shape = std::vector<size_t>(n_dim_target);
        std::vector<size_t> new_strides = std::vector<size_t>(n_dim_target);

        // align to the right
        for (int64_t i = 0; i < n_dim_orig; ++i) {
            new_shape[i + (n_dim_target - n_dim_orig)] = source.m_shape[i];
            new_strides[i + (n_dim_target - n_dim_orig)] = source.m_strides[i];
        }
        for (int64_t i = n_dim_target - n_dim_orig - 1; i >= 0; --i) { // fill leftovers (dont even have to check later)
            new_shape[i] = target_shape[i];
            new_strides[i] = 0;
        }

        for (int64_t i = (n_dim_target - n_dim_orig); i < n_dim_target; ++i) {
            if (target_shape[i] == new_shape[i]) {
                // literally leave everything as is
            }
            else if (new_shape[i] == 1) {
                new_shape[i] = target_shape[i];
                new_strides[i] = 0;
            }
            else {
                throw std::runtime_error("Violated broadcasting rules in lobotomized_broadcast().");
            }
        }
        return Tensor<T>(std::move(new_shape), std::move(new_strides), source.m_offset, source.m_state->m_storage, false); // lobotomy (no past or future)
    }

    template <typename T>
    Tensor<T> lobotomized_contiguous(const Tensor<T>& source) {
        if (source.m_shape.empty()) {
            Tensor<T> scalar_tensor = Tensor<T>(std::vector<size_t>{});
            (scalar_tensor.m_state->m_storage->m_data)[0] = (source.m_state->m_storage->m_data)[source.m_offset];
            return scalar_tensor;
        }
        Tensor<T> new_contiguous = Tensor<T>(source.m_shape);
        const size_t n_dim = source.m_shape.size();
        std::vector<size_t> odometer(n_dim, 0); 
        size_t contiguous_idx = 0;
        while (odometer[0] < source.m_shape[0]) {
            size_t strided_idx = source.m_offset;
            for (size_t i = 0; i < n_dim; ++i) {
                strided_idx += odometer[i] * source.m_strides[i];
            }
            (new_contiguous.m_state->m_storage->m_data)[contiguous_idx] = (source.m_state->m_storage->m_data)[strided_idx];
            ++contiguous_idx;
            ++odometer[n_dim - 1];
            size_t i = n_dim - 1;
            while ((odometer[i] == source.m_shape[i]) && i > 0) {
                odometer[i] = 0;
                ++odometer[i - 1];
                --i;
            }
        }
        return new_contiguous;
    }

    template <typename T, typename Func>
    void apply_in_place(Tensor<T>& left, const Tensor<T>& right, Func op) { 
        if (left.m_shape.empty() && right.m_shape.empty()) {
            op(left.m_state->m_storage->m_data[left.m_offset], right.m_state->m_storage->m_data[right.m_offset]);
            ++left.m_state->m_storage->m_version;
            return;
        }
        // Idea behind all the variables - track right variables instead of copying vectors on the heap inside copied Tensors.
        // (you would have to either create a tensor or copy it to hold a variable (or worse - write main logic twice) - why not ONLY create if needed?)
        const std::vector<size_t>* right_strides; // promise not to modify data, (can reassign the pointer)
        size_t right_offset;

        Tensor<T> broad_right;
        if (left.m_shape == right.m_shape) {
            right_strides = &right.m_strides;
            right_offset = right.m_offset;
        }
        else {
            broad_right = lobotomized_broadcast(right, left.m_shape);

            right_strides = &broad_right.m_strides;
            right_offset = broad_right.m_offset;
        }

        const size_t n_dim = left.m_shape.size();
        std::vector<size_t> odometer(n_dim, 0);
        while (odometer[0] < left.m_shape[0]) {
            size_t left_strided_idx = left.m_offset; 
            size_t right_strided_idx = right_offset;

            for (size_t i = 0; i < n_dim; ++i) {
                left_strided_idx += odometer[i] * left.m_strides[i];
                right_strided_idx += odometer[i] * (*right_strides)[i];
            }
            op((left.m_state->m_storage->m_data)[left_strided_idx], (right.m_state->m_storage->m_data)[right_strided_idx]);
            ++odometer[n_dim - 1];
            size_t i = n_dim - 1;
            while ((odometer[i] == left.m_shape[i]) && i > 0) {
                odometer[i] = 0;
                ++odometer[i - 1];
                --i;
            }
        }
        ++left.m_state->m_storage->m_version;
    }

    template <typename T, typename Func>
    Tensor<T> apply_out_of_place(const Tensor<T>& left, const Tensor<T>& right, const std::vector<size_t>& target_shape, Func op) {
        if (target_shape.size() == 0) { // op on 2 scalars
            Tensor<T> result = Tensor<T>(target_shape);
            (result.m_state->m_storage->m_data)[0] = op((left.m_state->m_storage->m_data)[left.m_offset], (right.m_state->m_storage->m_data)[right.m_offset]);
            return result;
        }

        const std::vector<size_t>* left_strides = &left.m_strides;
        const std::vector<size_t>* right_strides = &right.m_strides;
        size_t left_offset = left.m_offset;
        size_t right_offset = right.m_offset; 
        std::shared_ptr<Storage<T>> left_storage = left.m_state->m_storage; // broadcasting does not alter memory
        std::shared_ptr<Storage<T>> right_storage = right.m_state->m_storage;

        Tensor<T> broad_right;
        Tensor<T> broad_left;

        if (left.m_shape != target_shape) {
            broad_left = lobotomized_broadcast(left, target_shape);

            left_strides = &broad_left.m_strides;
            left_offset = broad_left.m_offset;
        }
        if (right.m_shape != target_shape) {
            broad_right = lobotomized_broadcast(right, target_shape);

            right_strides = &broad_right.m_strides;
            right_offset = broad_right.m_offset;
        }
        

        Tensor<T> result = Tensor<T>(target_shape);

        const size_t n_dim = target_shape.size();
        std::vector<size_t> odometer(n_dim, 0);
        size_t contiguous_idx = 0;
        while (odometer[0] < target_shape[0]) {
            size_t left_strided_idx = left_offset; 
            size_t right_strided_idx = right_offset;

            for (size_t i = 0; i < n_dim; ++i) {
                left_strided_idx += odometer[i] * (*left_strides)[i];
                right_strided_idx += odometer[i] * (*right_strides)[i];
            }
            (result.m_state->m_storage->m_data)[contiguous_idx] = op((left_storage->m_data)[left_strided_idx], (right_storage->m_data)[right_strided_idx]); // copied straight into CPU registers from RAM
            ++contiguous_idx;
            ++odometer[n_dim - 1];
            size_t i = n_dim - 1;
            while ((odometer[i] == target_shape[i]) && i > 0) {
                odometer[i] = 0;
                ++odometer[i - 1];
                --i;
            }
        }

        return result;
    }

    template <typename T>
    std::ostream& operator<<(std::ostream& stream, const std::vector<T>& vector) {
        stream << "[";
        for (size_t i = 0; i < vector.size(); ++i) {
            if (i == vector.size() - 1) {
                stream << vector[i];
            }
            else {
                stream << vector[i] << ", ";
            }
            
        }
        stream << "]";
        return stream;
    }

    struct PrintOptions {
        size_t edge_items = 5;
        int dim_indentation = 2;
        bool show_metadata = true;
    };

    template <typename T>
    void print_dim(std::ostream& stream, const Tensor<T>& source, const PrintOptions& opts, size_t current_dim, size_t base_offset, bool is_last) {
        if (current_dim == source.m_shape.size() - 1) {
            stream << std::string(current_dim * opts.dim_indentation, ' ') << "[";
            // print values along the dim
            if (source.m_shape[current_dim] > 2 * opts.edge_items) {
                // print edge_items first, edge_items last
                for (size_t i = 0; i < opts.edge_items; ++i) {
                    stream << source.m_state->m_storage->m_data[base_offset + i * source.m_strides[current_dim]];
                    stream << ", ";
                }
                stream << "..., ";

                for (size_t i = source.m_shape[current_dim] - opts.edge_items; i < source.m_shape[current_dim]; ++i) {
                    stream << source.m_state->m_storage->m_data[base_offset + i * source.m_strides[current_dim]];
                    if (i != source.m_shape[current_dim] - 1) {
                        stream << ", ";
                    }
                }
            }

            else {
                // print everything
                for (size_t i = 0; i < source.m_shape[current_dim]; ++i) {
                    stream << source.m_state->m_storage->m_data[base_offset + i * source.m_strides[current_dim]];
                    if (i != source.m_shape[current_dim] - 1) {
                         stream << ", ";
                    }
                } 
            }
            stream << "]";
            if (!is_last) {
                stream << ",";
            }
            stream << std::endl;
        }

        else {
            stream << std::string(current_dim * opts.dim_indentation, ' ') << "[" << std::endl;
            if (source.m_shape[current_dim] > 2 * opts.edge_items) {
                // print edge_items first, edge_items last
                for (size_t i = 0; i < opts.edge_items; ++i) {
                    print_dim(stream, source, opts, current_dim + 1, base_offset + i * source.m_strides[current_dim], false);
                }
                stream << std::string((current_dim + 1) * opts.dim_indentation, ' ') << "..." << std::endl;

                for (size_t i = source.m_shape[current_dim] - opts.edge_items; i < source.m_shape[current_dim]; ++i) {
                    if (i == source.m_shape[current_dim] - 1) {
                        print_dim(stream, source, opts, current_dim + 1, base_offset + i * source.m_strides[current_dim], true);
                    }
                    else {
                        print_dim(stream, source, opts, current_dim + 1, base_offset + i * source.m_strides[current_dim], false);
                    }
                }
            }

            else {
                // print everything
                for (size_t i = 0; i < source.m_shape[current_dim]; ++i) {

                    if (i == source.m_shape[current_dim] - 1) {
                        print_dim(stream, source, opts, current_dim + 1, base_offset + i * source.m_strides[current_dim], true);
                    }
                    else {
                        print_dim(stream, source, opts, current_dim + 1, base_offset + i * source.m_strides[current_dim], false);
                    }
                }
            } 

            stream << std::string(current_dim * opts.dim_indentation, ' ');
            stream << "]";
            if (!is_last) {
                stream << ",";
            }
            stream << std::endl;
        }
    }

    template <typename T>
    std::ostream& print_tensor(std::ostream& stream, const Tensor<T>& source, const PrintOptions opts = {}) { // = {} means use initializer list if nothing is provided (but empty so default construct)
        if (source.m_shape.size() == 0) {
            if (opts.show_metadata) {
                stream << "Grad: " << source.m_requires_grad << std::endl;
            }
            if (!source.m_state->m_storage->m_data.empty()) {
                stream << "Tensor(" << source.item() << ")" << std::endl;
                return stream;
            }
            else {
                stream << "Attempted printing tensor without data: aborting (call .realize() first)" << std::endl;
                return stream;
            }
        }
        
        else {
            if (opts.show_metadata) {
                std::cout << "Shape: " << source.m_shape << " | Strides: " << source.m_strides << " | Grad: " << source.m_requires_grad << std::endl;
            }
            if (!source.m_state->m_storage->m_data.empty()) {
                print_dim(stream, source, opts, 0, source.m_offset, true);
                return stream;
            }
            else {
                stream << "Attempted printing tensor without data: aborting (call .realize() first)" << std::endl;
                return stream;
            }
        }
    }

    inline int64_t normalize_axis(int64_t axis, int64_t n_dim) {
        if (axis < 0) {axis += n_dim;}
        if (axis < 0 || axis >= n_dim) {
            throw std::out_of_range("Axis out of bounds");
        }
        return axis;
    }

    inline size_t calculate_dim_product(const std::vector<size_t>& shape) {
        size_t dim_product = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            dim_product *= shape[i];
        }

        return dim_product;
    }


    inline ReductionMetadata infer_reduction_metadata(const std::vector<size_t>& source_shape, const std::vector<int64_t>& red_axes, bool keepdims) {
        const int64_t n_dim = static_cast<int64_t>(source_shape.size());
        std::vector<int64_t> positive_red_axes;
        positive_red_axes.reserve(n_dim);
        for (const int64_t x : red_axes) {
            positive_red_axes.push_back(normalize_axis(x, n_dim));
        }

        std::vector<size_t> temp_shape = source_shape;
        std::vector<size_t> temp_strides = std::vector<size_t>(n_dim);
        std::vector<size_t> result_shape;
        std::vector<size_t> result_strides;
        size_t result_vol = 1;
        size_t reduced_vol = 1;

        // first calculate output shape. Then calculate temporary values that allow operations
        for (const size_t ax : positive_red_axes) {
            result_vol *= temp_shape[ax];
            temp_shape[ax] = 1; // later just copy temp as result_shape and retain/remove axes
        }

        result_shape = temp_shape;
        result_vol = calculate_dim_product(result_shape);
        

        temp_strides[n_dim - 1] = 1;
        for (int64_t i = n_dim - 1; i > 0; --i) {
            temp_strides[i - 1] = temp_shape[i] * temp_strides[i];
        }

        result_strides = temp_strides; // based on keepdims we will retain/remove certain indices

        for (const size_t ax : positive_red_axes) {
            temp_strides[ax] = 0;
        }

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
            return ReductionMetadata(temp_shape, temp_strides, collapsed_result_shape, collapsed_result_strides, result_vol, reduced_vol);
        }

        else {
            return ReductionMetadata(temp_shape, temp_strides, result_shape, result_strides, result_vol, reduced_vol);
        }
        
    }

    template <typename T, typename Func>
    inline Tensor<T> apply_reduction_operation(const Tensor<T>& source, const ReductionMetadata& reduction_metadata, T init_value, Func op) {
        const int64_t n_dim = static_cast<int64_t>(source.m_shape.size());
        Tensor<T> result = Tensor<T>(reduction_metadata.result_shape, init_value);

        std::vector<size_t> odometer(n_dim, 0);
        while (odometer[0] < source.m_shape[0]) {
            size_t in_strided_idx = source.m_offset; 
            size_t out_strided_idx = 0;

            for (size_t i = 0; i < n_dim; ++i) {
                in_strided_idx += odometer[i] * source.m_strides[i];
                out_strided_idx += odometer[i] * reduction_metadata.temp_strides[i];
            }
            (result.m_state->m_storage->m_data)[out_strided_idx] = op((result.m_state->m_storage->m_data)[out_strided_idx], (source.m_state->m_storage->m_data)[in_strided_idx]);
            ++odometer[n_dim - 1];
            size_t i = n_dim - 1;
            while ((odometer[i] == source.m_shape[i]) && i > 0) {
                odometer[i] = 0;
                ++odometer[i - 1];
                --i;
            }
        }

        return result;
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

        if constexpr (std::is_same_v<T, PromotedT>) {
            p_left = std::move(left);
        }
        else {
            p_left = left.template cast<PromotedT>();
        }

        return std::make_pair(std::move(p_left), std::move(p_right));
    }


}