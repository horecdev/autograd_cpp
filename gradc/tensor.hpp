#pragma once
#include "node.hpp"
#include "graph_fwd.hpp" // IWYU pragma: keep

#include <cstdint>
#include <iostream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <optional>

namespace gradc {

    // random structs that are useful

    struct LazyTag{};
    inline constexpr LazyTag lazy = LazyTag();

    struct Placeholder{};
    inline constexpr Placeholder _ = Placeholder(); // inline doesnt confuse linker if its #included in 2+ files.

    enum IndexType {Single, Range, All};

    struct Slice {
        std::optional<int64_t> start;
        std::optional<int64_t> stop;
        std::optional<int64_t> step;

        Slice(std::optional<int64_t> start = std::nullopt, std::optional<int64_t> stop = std::nullopt, std::optional<int64_t> step = std::nullopt) 
            : start(start), stop(stop), step(step) {}
    };



    struct IndexDesc {
        IndexType m_type;
        int64_t m_single_val;
        Slice m_slice_val;

        IndexDesc(int64_t value) : m_type(IndexType::Single), m_single_val(value) {}
        IndexDesc(Placeholder) : m_type(IndexType::All) {}
        IndexDesc(Slice slice_value) : m_type(IndexType::Range), m_slice_val(std::move(slice_value)) {}
    };

    struct PrintOptions;

    struct ReductionMetadata{
        std::vector<int64_t> temp_shape;
        std::vector<int64_t> temp_strides;
        std::vector<int64_t> result_shape;
        std::vector<int64_t> result_strides;
        int64_t result_vol;
        int64_t reduced_vol;
    };

    template <typename T>
    class Node;

    template <typename T>
    constexpr bool is_supported_tensor_type_v = 
        std::is_same_v<T, int16_t> ||
        std::is_same_v<T, int32_t> ||
        std::is_same_v<T, int64_t> ||
        std::is_same_v<T, float> ||
        std::is_same_v<T, double> ||
        std::is_same_v<T, bool>;

    enum class DType {
        Float32,
        Float64,
        Int32,
        Int64,
        Bool,
        Unknown
    };

    template <typename T>
    constexpr DType type_to_dtype() { // before signature: allows the compiler to figure out at compilation
        if constexpr (std::is_same_v<T, float>) return DType::Float32; // before if statement: deletes the line if false, keeps if true
        else if constexpr (std::is_same_v<T, double>) return DType::Float64;
        else if constexpr (std::is_same_v<T, int32_t>) return DType::Int32;
        else if constexpr (std::is_same_v<T, int64_t>) return DType::Int64;
        else if constexpr (std::is_same_v<T, bool>) return DType::Bool;
        else return DType::Unknown;
    }

    template <typename T>
    struct Storage{
        std::vector<T> m_data;
        int64_t m_version;

        Storage() : m_data(std::vector<T>{}), m_version(0) {}

        Storage(std::vector<T>&& data) : m_data(std::move(data)), m_version(0) {}
        // If there is a new buffer - zero out the version.
        Storage(const Storage& other) : m_data(other.m_data), m_version(0) {}
        Storage(Storage&& other) noexcept : m_data(std::move(other.m_data)), m_version(0) {}
        
        Storage& operator=(const Storage& other) noexcept {
            if (this != &other) {
                m_data = other.m_data; // copy deep
                m_version = 0;
            }
            return *this;

        }
        Storage& operator=(Storage&& other) {
            if (this != &other) {
                m_data = std::move(other.m_data);
                m_version = 0;
            }
            return *this;
        } 

        ~Storage() {
            //std::cout << "Storage Destroyed" << std::endl;
        }
    };

    struct TensorStateBase {
        public:
            virtual std::vector<TensorStateBase*> get_dependencies() const = 0;

            virtual void backward() const = 0;

            virtual void clear_grad_if_non_leaf() = 0;
    };

    template <typename T>
    struct TensorState : public TensorStateBase {
        std::shared_ptr<Storage<T>> m_storage; // nodes can share just storage
        std::unique_ptr<Node<T>> m_creation_op; // it makes no sense to share m_op, but not storage.
        bool m_is_realized;
        std::optional<Tensor<T>> m_grad = std::nullopt; // every TensorState has grad. Even reshapes. 

        TensorState() : m_storage(std::make_shared<Storage<T>>()), m_creation_op(nullptr), m_is_realized(false) {}
        
        TensorState(std::vector<T>&& data) : m_storage(std::make_shared<Storage<T>>(std::move(data))), m_creation_op(nullptr), m_is_realized(false) {}

        TensorState(std::shared_ptr<Storage<T>> storage) : m_storage(std::move(storage)), m_creation_op(nullptr), m_is_realized(false) {} // copy tensor storage, set m_r_op to be nothing

        TensorState(std::shared_ptr<Storage<T>> storage, std::unique_ptr<Node<T>> realize_op) : m_storage(std::move(storage)), m_creation_op(std::move(realize_op)), m_is_realized(false) {}

        std::vector<TensorStateBase*> get_dependencies() const override {
            if (m_creation_op == nullptr) {
                return std::vector<TensorStateBase*>();
            }
            return m_creation_op->get_input_states();
        }

        void backward() const override {
            m_creation_op->backward(m_grad.value());
        }

        void clear_grad_if_non_leaf() {
            if (m_creation_op != nullptr) {
                m_grad = std::nullopt;
            }
        }
    };

    template <typename T>
    class Tensor {
        static_assert(is_supported_tensor_type_v<T>, "FATAL: Attempted creating a Tensor with unsupported data type.");

        template <typename U> friend class Tensor; // otherwise Tensor A cant look into T. B
        template <typename U> friend class Node;

        private:
            std::vector<int64_t> m_shape;
            std::vector<int64_t> m_strides;
            int64_t m_offset;
            std::shared_ptr<TensorState<T>> m_state;
            bool m_requires_grad;

            static bool validate_requires_grad(bool requires_grad) {
                if constexpr (!std::is_floating_point_v<T>) {
                    if (requires_grad) {
                        throw std::runtime_error("FATAL: Cannot require gradients on non-floating Tensors.");
                    }
                }
                return requires_grad;
            }

        public:
            void realize() {
                if (m_state->m_is_realized != true) {
                    Tensor computed_result = m_state->m_creation_op->realize();
                    if (m_state->m_storage != computed_result.m_state->m_storage) {
                        m_state->m_storage->m_data = std::move(computed_result.m_state->m_storage->m_data);
                    }
                }

                m_state->m_is_realized = true; // so we dont realize() twice (reflected across multiple aliases)
            }

            void backward();

            void accumulate_grad(const Tensor<T>& incoming_grad);

            void zero_grad();

            // LIFECYCLE 
            Tensor();
            explicit Tensor(T value);
            explicit Tensor(std::vector<int64_t> shape, T init_val = T());
            Tensor(std::vector<int64_t> shape, std::shared_ptr<Storage<T>> storage);
            Tensor(std::vector<int64_t> shape, bool requires_grad, LazyTag);
            Tensor(std::vector<int64_t> shape, std::vector<int64_t> strides, int64_t offset, std::shared_ptr<Storage<T>> storage, bool requires_grad);
            Tensor(std::vector<int64_t> shape, std::vector<int64_t> strides, int64_t offset, std::shared_ptr<TensorState<T>> state, bool requires_grad);
            ~Tensor();
            Tensor(const Tensor& source);
            Tensor(Tensor&& source);
            Tensor& operator=(const Tensor& source);
            Tensor& operator=(Tensor&& source);
            Tensor clone() const;

            // INDEXING
            template <typename... Args>
            Tensor operator[](Args... args) const;

            T item() const;

            // GETTERS

            const std::vector<int64_t>& shape() const {return m_shape;}

            const std::vector<int64_t>& strides() const {return m_strides;}

            int64_t offset() const {return m_offset;}

            bool requires_grad() const {return m_requires_grad;}

            TensorStateBase* _get_state_base() {
                return m_state.get();
            }

            std::optional<Tensor<T>> grad() const {
                return m_state->m_grad;
            }

            DType dtype() const {return type_to_dtype<T>();}

            const auto get_realize_op_ptr_type() const {
                if (m_state->m_creation_op == nullptr) {
                    return "nullptr";
                }
                return typeid(*m_state->m_creation_op).name();
            }

            int64_t volume() const {
                int64_t vol = 1;
                for (int64_t dim : m_shape) {
                    vol *= dim;
                }
                return vol;
            }

            // SETTERS
            void set_data(std::vector<T> data) {
                this->m_state->m_storage->m_data = data; // copy vector
            }
            Tensor& set_requires_grad(bool value) {
                m_requires_grad = validate_requires_grad(value);
                return *this;
            }

            // SHAPES
            bool is_contiguous() const;
            Tensor contiguous() const;
            Tensor transpose(const int64_t dim0, const int64_t dim1) const;
            Tensor permute(const std::vector<int64_t>& axes) const;
            Tensor reshape(const std::vector<int64_t>& target_shape) const;

            template <typename U> friend Tensor<T> lazy_concat(std::vector<Tensor<T>> &tensor_list, int64_t concat_dim);
            
            // MATH
            template <typename U, typename Func> friend void apply_in_place(Tensor<U>& left, const Tensor<U>& right, Func op);
            template <typename U, typename Func> friend Tensor<U> apply_out_of_place(const Tensor<U>& left, const Tensor<U>& right, const std::vector<int64_t>& target_shape, Func op);
            template <typename U, typename Func> friend Tensor<U> apply_reduction_operation(const Tensor<U>& source, const ReductionMetadata& reduction_metadata, U init_value, Func op);

            template <typename U, typename W> friend auto operator+(Tensor<U> left, Tensor<W> right); // we befriend whole family of functions named operator+. 
            template <typename U, typename W> friend auto operator*(Tensor<U> left, Tensor<W> right); // It operates on type U and U can be virtually anything

            template <typename U, typename W> friend Tensor<U>& operator+=(Tensor<U>& main, Tensor<W> other);
            template <typename U, typename W> friend Tensor<U>& operator*=(Tensor<U>& main, Tensor<W> other);

            // REDUCTIONS

            Tensor sum(const std::vector<int64_t>& axes, bool keepdims) const;
            template <typename OutT = std::conditional_t<std::is_integral_v<T>, float, T>> // if int: float. Otherwise T.
            Tensor<OutT> mean(const std::vector<int64_t>& axes, bool keepdims) const;

            // UTILS
            template <typename U> friend Tensor<U> lobotomized_broadcast_view(const Tensor<U>& source, const std::vector<int64_t>& target_shape);
            template <typename U> friend Tensor<U> lobotomized_contiguous_alloc(const Tensor<U>& source);
            template <typename U> friend Tensor<U> lobotomized_transpose_view(const Tensor<U>& source, int64_t dim0, int64_t dim1);
            template <typename U> friend Tensor<U> lobotomized_reshape_view(const Tensor<U>& source);
            template <typename U> friend Tensor<U> lobotomized_permute_view(const Tensor<T>& source, const std::vector<int64_t>& axes);
            template <typename InT, typename OutT> friend Tensor<OutT> lobotomized_cast_alloc(const Tensor<InT>& source);
            template <typename U> friend Tensor<U> create_lobotomized_slice_view(const Tensor<T>& source, const std::vector<IndexDesc>& descriptors);
            template <typename U> friend Tensor<U> lobotomized_concat_alloc(const std::vector<Tensor<U>>& tensor_list, int64_t concat_dim, std::vector<int64_t> final_shape);

           

            template <typename U> friend std::ostream& print_tensor(std::ostream& stream, const Tensor<U>& source, PrintOptions opts);
            template <typename U> friend void print_dim(std::ostream& stream, const Tensor<U>& source, const PrintOptions& opts, int64_t current_dim, int64_t base_offset, bool is_last);

            template <typename U> friend Tensor<U> unbroadcast_grad(const Tensor<U>& raw_grad, const Tensor<U>& parent);

            template <typename TargetT> Tensor<TargetT> cast() const;
    };      
} 

   