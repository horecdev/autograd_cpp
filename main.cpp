#include <iostream>
#include <vector>
#include <stdexcept> // IWYU pragma: keep
#include "gradc/gradc.hpp" // IWYU pragma: keep

using namespace gradc;

int main() {
    try {
        std::cout << "[1/4] Initializing base tensors..." << std::endl;
        Tensor<int32_t> a({2, 2});
        a.set_data({1, 2, 
                    3, 4});

        Tensor<float> f({2, 2});
        f.set_data({0.5f, 1.0f, 
                    1.5f, 2.0f});

        std::cout << "[2/4] Branching graph BEFORE in-place mutation..." << std::endl;
        // b captures 'a' before any mutations happen
        auto b = a + 10; 

        std::cout << "[3/4] Executing In-Place Operations..." << std::endl;
        
        // Test 1: Standard homogeneous in-place addition
        a += 5; 

        // Test 2: Mixed-type in-place addition (Float32 += Int32)
        // This is legal because common_type_t<float, int32_t> is float!
        // It triggers your internal: p_other = other.template cast<PromotedT>();
        f += a; 

        std::cout << "[4/4] Triggering graph realization..." << std::endl;
        // Realize the endpoints of our graph
        b.realize();
        a.realize();
        f.realize();

        std::cout << "\n--- IN-PLACE MUTATION RESULTS ---" << std::endl;

        std::cout << "[ Test 1: Active Tensor 'a' after += 5 ]" << std::endl;
        std::cout << "Expected: [[6, 7], [8, 9]]" << std::endl;
        std::cout << "Actual:   ";
        print_tensor(std::cout, a);
        std::cout << "\n" << std::endl;

        std::cout << "[ Test 2: Mixed-Type 'f' after f += a ]" << std::endl;
        std::cout << "Expected calculation: [[0.5, 1.0], [1.5, 2.0]] + [[6, 7], [8, 9]]" << std::endl;
        std::cout << "Expected: [[6.5, 8.0], [9.5, 11.0]]" << std::endl;
        std::cout << "Actual:   ";
        print_tensor(std::cout, f);
        std::cout << "\n" << std::endl;

        std::cout << "[ Test 3: The Graph Isolation Check ('b') ]" << std::endl;
        std::cout << "Context: 'b = a + 10' was declared BEFORE 'a += 5'." << std::endl;
        std::cout << "-> If your engine mutates storage eagerly, 'b' will read the mutated 'a' and show: [[16, 17], [18, 19]]" << std::endl;
        std::cout << "-> If your engine tracks graphs lazily or clones on write, 'b' will show original values: [[11, 12], [13, 14]]" << std::endl;
        std::cout << "Actual 'b': ";
        print_tensor(std::cout, b);
        std::cout << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\n!!! Crash during execution: " << e.what() << std::endl;
    }

    return 0;
}