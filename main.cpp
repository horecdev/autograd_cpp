#include <iostream>
#include <vector>
#include <stdexcept> // IWYU pragma: keep
#include "gradc/gradc.hpp" // IWYU pragma: keep

using namespace gradc;

void print_section(const std::string& name) {
    std::cout << "\n======================================================\n";
    std::cout << " TEST: " << name << "\n";
    std::cout << "======================================================\n";
}

int main() {
    PrintOptions opts;
    opts.show_metadata = true;

    print_section("1. Basic Initialization & Printing");
    Tensor<float> a({2, 3});
    a.set_data({1, 2, 3, 4, 5, 6});
    std::cout << "Tensor A (2x3):\n";
    print_tensor(std::cout, a, opts);

    print_section("2. Reshape (Lazy & Realize)");
    Tensor<float> b = a.reshape({3, 2});
    std::cout << "Tensor B (Reshaped 3x2, BEFORE realize):\n";
    // Should print the correct shape/strides but might share data correctly anyway since reshape just alters metadata
    print_tensor(std::cout, b, opts); 
    
    b.realize();
    std::cout << "\nTensor B (AFTER realize):\n";
    print_tensor(std::cout, b, opts);

    print_section("3. Transpose & is_contiguous()");
    Tensor<float> c = a.transpose(0, 1);
    c.realize();
    std::cout << "Tensor C (Transposed 3x2):\n";
    print_tensor(std::cout, c, opts);
    std::cout << "Is A contiguous? " << (a.is_contiguous() ? "Yes" : "No") << "\n";
    std::cout << "Is C contiguous? " << (c.is_contiguous() ? "Yes" : "No") << "\n";

    print_section("4. Contiguous Node Enforcement");
    Tensor<float> c_contig = c.contiguous();
    std::cout << "Before realize, c_contig data is likely uninitialized or shared oddly.\n";
    c_contig.realize();
    std::cout << "Tensor c_contig (AFTER realize):\n";
    print_tensor(std::cout, c_contig, opts);
    std::cout << "Is c_contig contiguous? " << (c_contig.is_contiguous() ? "Yes" : "No") << "\n";

    print_section("5. Indexing & Slicing (Using Placeholder _ )");
    // a is 2x3: 
    // [1, 2, 3]
    // [4, 5, 6]
    Tensor<float> row = a[1, _]; // 2nd row
    row.realize();
    std::cout << "Sliced Row ( a[1, _] ) expects [4, 5, 6]:\n";
    print_tensor(std::cout, row, opts);

    Tensor<float> col = a[_, 2]; // 3rd column
    col.realize();
    std::cout << "\nSliced Col ( a[_, 2] ) expects [3, 6]:\n";
    print_tensor(std::cout, col, opts);

    print_section("6. Out-of-place Math & Lazy Evaluation");
    Tensor<float> d({2, 3});
    d.set_data({10, 20, 30, 40, 50, 60});
    
    Tensor<float> e = a + d;
    std::cout << "Tensor E = A + D (BEFORE realize - data should be zeros/garbage):\n";
    print_tensor(std::cout, e, opts);
    
    e.realize();
    std::cout << "\nTensor E = A + D (AFTER realize):\n";
    print_tensor(std::cout, e, opts);

    Tensor<float> m = a * d;
    m.realize();
    std::cout << "\nTensor M = A * D (AFTER realize):\n";
    print_tensor(std::cout, m, opts);

    print_section("7. Broadcasting Math");
    Tensor<float> vec({3});
    vec.set_data({100, 200, 300});
    std::cout << "Broadcasting Vector:\n";
    print_tensor(std::cout, vec, opts);

    Tensor<float> bcast_add = a + vec;
    bcast_add.realize();
    std::cout << "\nA + Vector (Broadcasting expects [101, 202, 303], [104, 205, 306]):\n";
    print_tensor(std::cout, bcast_add, opts);

    print_section("8. In-place Math");
    Tensor<float> inplace_target({2, 3});
    inplace_target.set_data({1, 1, 1, 1, 1, 1});
    
    std::cout << "Inplace Target Before (All 1s):\n";
    print_tensor(std::cout, inplace_target, opts);

    // Using += (adds A to inplace_target)
    inplace_target += a;
    inplace_target.realize();
    std::cout << "\nInplace Target AFTER ( += A ):\n";
    print_tensor(std::cout, inplace_target, opts);

    // Using *= (multiplies by 10)
    Tensor<float> multiplier({2, 3});
    multiplier.set_data({10, 10, 10, 10, 10, 10});
    inplace_target *= multiplier;
    inplace_target.realize();
    std::cout << "\nInplace Target AFTER ( *= 10 ):\n";
    print_tensor(std::cout, inplace_target, opts);

    print_section("9. Multi-Step Computation Graph Verification");
    // Graph: g1 = a + d, g2 = g1 * vec, g3 = g2.reshape({3, 2})
    Tensor<float> g1 = a + d;
    Tensor<float> g2 = g1 * vec;
    Tensor<float> g3 = g2.reshape({3, 2});

    std::cout << "Graph built. Nothing realized yet.\n";
    
    // Notice: Realizing g3 realizes g2, which realizes g1 (because of Node setup)
    g3.realize(); 
    
    std::cout << "G3 (AFTER realizing multi-step graph):\n";
    print_tensor(std::cout, g3, opts);

    std::cout << "\n======================================================\n";
    std::cout << " ALL TESTS COMPLETED.\n";
    std::cout << "======================================================\n";

    return 0;
}