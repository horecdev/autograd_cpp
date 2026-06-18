#include "tensor.hpp"

#include <unordered_set>
#include <vector>


// If Y1, Y2 depend on X:
// Case 1: Reaches Y1 before it has ever seen X - first adds X, then Y1
// Case 2: Reaches Y2 after it has seen X - Y2 is added AFTER X (cuz X was added before). Still holds. Order: X, Y1, Y2
class AutogradEngine {
    public:
        static std::vector<TensorState*> build_topo()
};