#pragma once
#include "../tensor.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
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

    template <typename T>
    Tensor<T> lobotomized_broadcast_view(const Tensor<T>& source, const std::vector<int64_t>& target_shape) {
        const int64_t n_dim_orig = std::ssize(source.m_shape);
        const int64_t n_dim_target = std::ssize(target_shape);
        std::vector<int64_t> new_shape = std::vector<int64_t>(n_dim_target);
        std::vector<int64_t> new_strides = std::vector<int64_t>(n_dim_target);

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
                throw std::runtime_error("Violated broadcasting rules in lobotomized_broadcast_view().");
            }
        }
        return Tensor<T>(std::move(new_shape), std::move(new_strides), source.m_offset, source.m_state->m_storage, false); // lobotomy (no past or future)
    }

    template <typename T>
    Tensor<T> lobotomized_contiguous_alloc(const Tensor<T>& source) {
        if (source.m_shape.empty()) {
            Tensor<T> scalar_tensor = Tensor<T>(std::vector<int64_t>{});
            (scalar_tensor.m_state->m_storage->m_data)[0] = (source.m_state->m_storage->m_data)[source.m_offset];
            return scalar_tensor;
        }
        Tensor<T> new_contiguous = Tensor<T>(source.m_shape);
        const int64_t n_dim = std::ssize(source.m_shape);
        std::vector<int64_t> odometer(n_dim, 0); 
        int64_t contiguous_idx = 0;
        while (odometer[0] < source.m_shape[0]) {
            int64_t strided_idx = source.m_offset;
            for (int64_t i = 0; i < n_dim; ++i) {
                strided_idx += odometer[i] * source.m_strides[i];
            }
            (new_contiguous.m_state->m_storage->m_data)[contiguous_idx] = (source.m_state->m_storage->m_data)[strided_idx];
            ++contiguous_idx;
            ++odometer[n_dim - 1];
            int64_t i = n_dim - 1;
            while ((odometer[i] == source.m_shape[i]) && i > 0) {
                odometer[i] = 0;
                ++odometer[i - 1];
                --i;
            }
        }
        return new_contiguous;
    }



    template <typename T1, typename T2, typename Func> // two types so you can cast one to another in lambda
    void apply_in_place(Tensor<T1>& left, const Tensor<T2>& right, Func op) { 
        if (left.m_shape.empty() && right.m_shape.empty()) {
            op(left.m_state->m_storage->m_data[left.m_offset], right.m_state->m_storage->m_data[right.m_offset]);
            ++left.m_state->m_storage->m_version;
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
        ++left.m_state->m_storage->m_version;
    }

    template <typename T, typename Func>
    Tensor<T> apply_out_of_place(const Tensor<T>& left, const Tensor<T>& right, const std::vector<int64_t>& target_shape, Func op) {
        if (std::ssize(target_shape) == 0) { // op on 2 scalars
            Tensor<T> result = Tensor<T>(target_shape);
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
        

        Tensor<T> result = Tensor<T>(target_shape);

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

    template <typename T>
    std::ostream& operator<<(std::ostream& stream, const std::vector<T>& vector) {
        stream << "[";
        for (int64_t i = 0; i < std::ssize(vector); ++i) {
            if (i == std::ssize(vector) - 1) {
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
        int64_t edge_items = 5;
        int dim_indentation = 2;
        bool show_metadata = true;
    };

    template <typename T>
    void print_dim(std::ostream& stream, const Tensor<T>& source, const PrintOptions& opts, int64_t current_dim, int64_t base_offset, bool is_last) {
        if (current_dim == std::ssize(source.m_shape) - 1) {
            stream << std::string(current_dim * opts.dim_indentation, ' ') << "[";
            // print values along the dim
            if (source.m_shape[current_dim] > 2 * opts.edge_items) {
                // print edge_items first, edge_items last
                for (int64_t i = 0; i < opts.edge_items; ++i) {
                    stream << source.m_state->m_storage->m_data[base_offset + i * source.m_strides[current_dim]];
                    stream << ", ";
                }
                stream << "..., ";

                for (int64_t i = source.m_shape[current_dim] - opts.edge_items; i < source.m_shape[current_dim]; ++i) {
                    stream << source.m_state->m_storage->m_data[base_offset + i * source.m_strides[current_dim]];
                    if (i != source.m_shape[current_dim] - 1) {
                        stream << ", ";
                    }
                }
            }

            else {
                // print everything
                for (int64_t i = 0; i < source.m_shape[current_dim]; ++i) {
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
                for (int64_t i = 0; i < opts.edge_items; ++i) {
                    print_dim(stream, source, opts, current_dim + 1, base_offset + i * source.m_strides[current_dim], false);
                }
                stream << std::string((current_dim + 1) * opts.dim_indentation, ' ') << "..." << std::endl;

                for (int64_t i = source.m_shape[current_dim] - opts.edge_items; i < source.m_shape[current_dim]; ++i) {
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
                for (int64_t i = 0; i < source.m_shape[current_dim]; ++i) {

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
        if (std::ssize(source.m_shape) == 0) {
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

    template <typename T, typename Func>
    inline Tensor<T> apply_reduction_operation(const Tensor<T>& source, const ReductionMetadata& reduction_metadata, T init_value, Func op) {
        const int64_t n_dim = std::ssize(source.m_shape);
        if (n_dim == 0) {
            throw std::runtime_error("Tried reducing a 0-Dimensional Tensor.");
        }

        Tensor<T> result = Tensor<T>(reduction_metadata.result_shape, init_value);

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
    Tensor<T> unbroadcast_grad(const Tensor<T>& raw_grad, const Tensor<T>& parent) {
        std::vector<int64_t> broadcast_axes = find_broadcast_axes(raw_grad.m_shape, parent.m_shape);

        if (broadcast_axes.empty()) {
            return raw_grad;
        }

        ReductionMetadata red_meta = infer_reduction_metadata(raw_grad.m_shape, broadcast_axes, false);
        Tensor<T> reduced = apply_reduction_operation(raw_grad, red_meta, T(), [](T a, T b){return a + b;});

        return Tensor<T>(parent.m_shape, std::move(reduced.m_state->m_storage)); // contiguous tensor (strides must be generated and shape must be kept as the one of parent)
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

    template <typename T>
    bool Tensor<T>::is_contiguous() const {
        if (m_shape.empty()) return true;
        if (m_strides[std::ssize(m_strides) - 1] != 1) {
            return false;
        }
        for (int64_t i = std::ssize(m_shape) - 1; i > 0; --i) {
            if (m_strides[i - 1] != m_shape[i] * m_strides[i]) {
                return false;
            }
        }
        return true;
    }

    template <typename T>
    Tensor<T> lobotomized_transpose_view(const Tensor<T>& source, int64_t dim0, int64_t dim1) {
        const int64_t n_dim = std::ssize(source.m_shape);
        dim0 = normalize_axis(dim0, n_dim);
        dim1 = normalize_axis(dim1, n_dim);

        std::vector<int64_t> new_shape = source.m_shape;
        std::vector<int64_t> new_strides = source.m_strides;
        std::swap(new_shape[dim0], new_shape[dim1]);
        std::swap(new_strides[dim0], new_strides[dim1]);

        Tensor<T> result = Tensor<T>(std::move(new_shape), std::move(new_strides), source.m_offset, source.m_state->m_storage, false);
        return result;
    }

    template <typename T>
    Tensor<T> lobotomized_reshape_view(const Tensor<T>& source, const std::vector<int64_t>& target_shape) {
        std::vector<int64_t> new_shape = std::vector<int64_t>(std::ssize(target_shape));
        std::vector<int64_t> new_strides = std::vector<int64_t>(std::ssize(target_shape));
        int64_t running_volume = 1;
        int64_t neg_one_idx = -1;

        int64_t total_volume = source.volume();
        
        for (int64_t i = 0; i < std::ssize(target_shape); ++i) {
            if (target_shape[i] == -1 && neg_one_idx == -1) {
                neg_one_idx = i;
            }
            else if (target_shape[i] == -1 && neg_one_idx != -1) {
                throw std::runtime_error("Cannot .reshape() with two or more unknown dimensions.");
            }
            else {
                new_shape[i] = target_shape[i];
                running_volume *= target_shape[i];
            }
        }

        int64_t unknown_dim;
        if (total_volume % running_volume != 0) {
            throw std::runtime_error("Invalid reshape parameters.");
        }
        else {
            unknown_dim = total_volume / running_volume;
        }

        if (neg_one_idx != -1) {
            new_shape[neg_one_idx] = unknown_dim;
        }

        new_strides[std::ssize(target_shape) - 1] = 1;
        for (int64_t i = std::ssize(target_shape) - 1; i > 0; --i) {
            new_strides[i - 1] = new_shape[i] * new_strides[i];
        }

        if (source.is_contiguous()) {
            return Tensor<T>(std::move(new_shape), std::move(new_strides), source.m_offset, source.m_state->m_storage, false);
        }
        else {
            throw std::runtime_error("lobotomized_reshape_view invoked on a non-contiguous tensor.");
        }
    }

    template <typename T>
    Tensor<T> lobotomized_permute_view(const Tensor<T>& source, const std::vector<int64_t>& axes) {
        const int64_t n_dim = std::ssize(source.m_shape);
        if (std::ssize(axes) != n_dim) {
            throw std::runtime_error("permute() axes list size must match shape list size.");
        }
        std::vector<int64_t> normalized_axes = normalize_axes_vector(axes, n_dim);
        std::vector<int64_t> new_shape = std::vector<int64_t>(n_dim);
        std::vector<int64_t> new_strides = std::vector<int64_t>(n_dim);
        std::vector<bool> seen_axes = std::vector<bool>(n_dim, false);
        for (int64_t target_ax = 0; target_ax < n_dim; ++target_ax) {
            int64_t src_ax = normalized_axes[target_ax];
            if (seen_axes[src_ax]) {
                throw std::runtime_error("Passed at least one axis twice inside .permute()");
            }
            seen_axes[src_ax] = true;

            new_shape[target_ax] = source.m_shape[src_ax];
            new_strides[target_ax] = source.m_strides[src_ax];
        }

        return Tensor<T>(std::move(new_shape), std::move(new_strides), source.m_offset, source.m_state->m_storage, false);
    }

    template <typename InT, typename OutT>
    Tensor<OutT> lobotomized_cast_alloc(const Tensor<InT>& source) {
        // source can have weird strides, but result is contiguous
        Tensor<OutT> result = Tensor<OutT>(source.m_shape);
        apply_in_place(result, source, [](OutT& a, InT b){a = static_cast<OutT>(b);});

        return result;
    }

    template <typename T>
    Tensor<T> lobotomized_concat_alloc(std::vector<Tensor<T>>& tensor_list, int64_t concat_dim, std::vector<int64_t> final_shape) {
        // target_shape must be known and concat_dim must be normalized already.
        Tensor<T> result = Tensor<T>(final_shape);
        int64_t current_offset = 0;

        for (const Tensor<T>& parent : tensor_list) {
            std::vector<int64_t> view_shape = parent.m_shape;

            Tensor<T> chunk_view(std::move(view_shape), result.m_strides, current_offset, result.m_state->m_storage, false);
            // by keeping strides it makes apply_in_place naturally skip indices for next tensor in list

            apply_in_place(chunk_view, parent, [](T& a, T b){ a = b; });

            current_offset += parent.m_shape[concat_dim] * result.m_strides[concat_dim];
            // if you have, say, 10, 5, 7 split into [10, 2, 7] and [10, 3, 7], increase offset, then you start writing at [0, 2, 0]. You never go back to [0, 0, 0] or [0, 1, 0]
            // or anything with the concat dim less than your current progress
        }

        return result;
    }
}