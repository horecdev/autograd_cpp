#include <iostream>
#include "gradc/tensor.hpp"

using namespace gradc;

int main() {
    Tensor<float> a({2, 2});
    Tensor<float> b({2, 2});
    std::cout << "Before add" << std::endl;
    a += b;
    std::cout << "The Engine Compiles." << std::endl;
    return 0;
}