#include <iostream>
#include "gradc/gradc.hpp"
#include "gradc/tensor.hpp"

using namespace gradc;

int main() {
    Tensor<float> a({2, 2});
    Tensor<float> b({2, 2});
    //Tensor<float> c({2});
    a.set_data({1, 2, 3, 4});
    b.set_data({4, 3, 2, 1});
    //c.set_data({10, 1});
    std::cout << "Before graph built" << std::endl;
    Tensor d = (a + b);// * c;
    std::cout << "After graph built" << std::endl;
    d.realize();
    std::cout << "After realize" << std::endl;
    std::cout << "The Engine Compiles." << std::endl;
    return 0;
}