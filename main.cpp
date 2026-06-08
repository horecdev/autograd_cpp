#include <iostream>
#include <vector>
#include <stdexcept> // IWYU pragma: keep
#include "gradc/gradc.hpp" // IWYU pragma: keep

using namespace gradc;

int main() {

    Tensor<float> vec({5});
    Tensor<float> mat({3, 10, 10});
    auto transposed = mat.transpose(0, 1);

    print_tensor(std::cout, mat, {});

    return 0;
}