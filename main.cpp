#include <iostream>
#include <vector>
#include <stdexcept> // IWYU pragma: keep
#include "gradc/gradc.hpp" // IWYU pragma: keep

using namespace gradc;

int main() {
    // 1. Init scalar tensors
    Tensor<float> a(2.0f);
    a.set_requires_grad(true);

    Tensor<float> b(3.0f);
    b.set_requires_grad(true);

    // 2. Math: c = (a * b) + a
    auto c = (a * b) + a;
    c.realize();

    // 3. Backward pass
    // Seed the root gradient with 1.0f
    c.accumulate_grad(Tensor<float>(1.0f)); 
    c.backward();

    // 4. Print expected vs actual
    // Expected dc/da = b + 1 = 3 + 1 = 4
    // Expected dc/db = a = 2
    std::cout << "Expected grad a: 4 | Actual: " << a.grad().value().item() << "\n";
    std::cout << "Expected grad b: 2 | Actual: " << b.grad().value().item() << "\n";

    return 0;
}