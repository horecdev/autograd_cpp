#include <iostream>
#include <vector>
#include "gradc/tensor.hpp"

template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& vector) {
    for (size_t i = 0; i < vector.size(); ++i) {
        if (i == vector.size() - 1) {
            stream << vector[i];
        }
        else {
            stream << vector[i] << " ";
        }
        
    }
    return stream;
}

int main() {
    gradc::Tensor<float> tens(std::vector<size_t>{4, 3 ,2});
    std::cout << "Shape: [" << tens.shape() << "] | Strides: [" << tens.strides() << "]" << std::endl;
}