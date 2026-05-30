#include <iostream>
#include <vector>
#include "gradc/tensor.hpp"

int main() {
    gradc::Tensor<float> tens(std::vector<size_t>{4, 3 ,2});
    tens.permute({2, 0, 1});
}