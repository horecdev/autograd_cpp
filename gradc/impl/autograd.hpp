#include "../tensor.hpp"
#include "tensor_math.hpp"
#include "tensor_utils.hpp"

#include <unordered_set>
#include <vector>
#include <algorithm>

namespace gradc {
    // If Y1, Y2 depend on X:
    // Case 1: Reaches Y1 before it has ever seen X - first adds X, then Y1
    // Case 2: Reaches Y2 after it has seen X - Y2 is added AFTER X (cuz X was added before). Still holds. Order: X, Y1, Y2
    class AutogradEngine {
        public:
            static void visit(TensorStateBase* current, std::unordered_set<TensorStateBase*>& visited, std::vector<TensorStateBase*>& topo_order) {
                if (visited.contains(current)) {
                    return;
                }

                visited.insert(current);

                for (TensorStateBase* new_root : current->get_dependencies()) {
                    visit(new_root, visited, topo_order);
                }

                topo_order.push_back(current);
            } 

            static std::vector<TensorStateBase*> build_topo(TensorStateBase* root) {
                std::unordered_set<TensorStateBase*> visited;
                std::vector<TensorStateBase*> topo_order;
                
                visit(root, visited, topo_order);

                std::reverse(topo_order.begin(), topo_order.end());
                return topo_order;
            }
    };


    template <typename T>
    void Tensor<T>::accumulate_grad(const Tensor<T>& incoming_grad) {
        if (!m_requires_grad) {return;}

        if (!m_state->m_grad.has_value()) {
            Tensor<T> local_grad = Tensor<T>(m_shape, T(0));
            apply_in_place(local_grad, incoming_grad, [](T &a, T b) {a += b;});
            m_state->m_grad = std::move(local_grad);
        }
        else {
            apply_in_place(m_state->m_grad.value(), incoming_grad, [](T &a, T b) {a += b;});
        }
    }

    template <typename T>
    void Tensor<T>::backward() {
        std::vector<TensorStateBase*> topo_order = AutogradEngine::build_topo(m_state);

        for (TensorStateBase* current : topo_order) {
            current->backward();
        }
    }
}

