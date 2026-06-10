#include <iostream>
#include <vector>
#include <stdexcept> // IWYU pragma: keep
#include "gradc/gradc.hpp" // IWYU pragma: keep
#include "gradc/impl/tensor_utils.hpp"

using namespace gradc;

using namespace gradc;

void print_section(const std::string& name) {
    std::cout << "\n--- " << name << " ---\n";
}

int main() {
    PrintOptions opts;
    opts.show_metadata = true;

    print_section("1. Scalar Initialization & .item()");
    Tensor<float> s1(std::vector<size_t>{});
    s1.set_data({5.0f});
    std::cout << "s1.item() returns: " << s1.item() << "\n";
    print_tensor(std::cout, s1, opts);

    print_section("2. Scalar Out-of-Place Math (s3 = s1 + s2)");
    Tensor<float> s2(std::vector<size_t>{});
    s2.set_data({10.0f});
    Tensor<float> s3 = s1 + s2;
    s3.realize();
    print_tensor(std::cout, s3, opts);

    print_section("3. Scalar In-Place Math (s1 += s2)");
    s1 += s2;
    s1.realize();
    print_tensor(std::cout, s1, opts);

    print_section("4. Tensor + Scalar Broadcasting");
    Tensor<float> mat({2, 2});
    mat.set_data({1.0f, 2.0f, 3.0f, 4.0f});
    
    Tensor<float> bcast_add = mat + s2; // mat + 10.0
    bcast_add.realize();
    print_tensor(std::cout, bcast_add, opts);

    print_section("5. Tensor * Scalar Broadcasting");
    Tensor<float> bcast_mul = mat * s2; // mat * 10.0
    bcast_mul.realize();
    print_tensor(std::cout, bcast_mul, opts);

    print_section("6. Tensor In-Place Scalar Broadcasting (mat += s2)");
    mat += s2;
    mat.realize();
    print_tensor(std::cout, mat, opts);

    std::cout << "\n[ALL SCALAR TESTS EXECUTED]\n";
    return 0;
}