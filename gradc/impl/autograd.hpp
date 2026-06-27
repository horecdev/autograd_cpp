#include "../tensor.hpp"
#include "tensor_math.hpp"
#include "tensor_utils.hpp"

#include <unordered_set>
#include <vector>

namespace gradc {
    template <typename T>
    void Tensor<T>::accumulate_grad(const Tensor<T>& incoming_grad) {
        if (!m_requires_grad) {return;}

        if (!m_state->m_grad.has_value()) {
            Tensor<T> local_grad = Tensor<T>(m_shape, T(0));
            apply_in_place(local_grad, incoming_grad, [](T &a, T b) {a += b;});
            m_state->m_grad = std::move(local_grad);
        }
        else {
            apply_in_place(m_state->m_grad, incoming_grad, [](T &a, T b) {a += b;});
        }
    }

    // If Y1, Y2 depend on X:
    // Case 1: Reaches Y1 before it has ever seen X - first adds X, then Y1
    // Case 2: Reaches Y2 after it has seen X - Y2 is added AFTER X (cuz X was added before). Still holds. Order: X, Y1, Y2
    class AutogradEngine {
        public:
            static std::vector<TensorStateBase*> build_topo(TensorStateBase* root) {
                
            }
    };
}

