#include <iostream>
#include <vector>
#include <stdexcept>
#include "gradc/gradc.hpp" // Ensure your headers are included

using namespace gradc;

int main() {

    try {
        Tensor<float> a({4, 4}); 

        Tensor<float> d = a.reshape({2, 8});

        std::cout << "HITTING REALIZE" << std::endl;

        d.realize();
        std::cout << "AFTER REALIZE" << std::endl;


    } catch (const std::exception& e) {
        std::cout << "\nCRITICAL ENGINE FAILURE: " << e.what() << std::endl;
    }

    return 0;
}