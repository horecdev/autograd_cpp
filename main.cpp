#include <iostream>
#include <vector>
#include <stdexcept> // IWYU pragma: keep
#include "gradc/gradc.hpp" // IWYU pragma: keep

int main() {
    using namespace gradc;

    Tensor<int32_t> t_a({2, 3});
    t_a.set_data({1, 2, 3, 4, 5, 6});

    Tensor<float> t_b({1, 3});
    t_b.set_data({0.5f, 1.5f, 2.5f});
    t_b.set_requires_grad(true); 

    Tensor<int32_t> t_c({2, 1});
    t_c.set_data({10, 20});

    auto t_cast_a = t_a.cast<float>();
    
    auto t_scalar_1 = t_a + 100;
    auto t_scalar_2 = 50.5f + t_cast_a;
    t_a += 5;

    auto t_broadcast_1 = t_cast_a + t_b;
    auto t_broadcast_2 = t_broadcast_1 + t_c;
    auto t_broadcast_3 = t_c + t_b;

    auto t_sum_0 = t_broadcast_3.sum({0}, false);
    auto t_sum_1_keep = t_broadcast_3.sum({1}, true);
    auto t_sum_all = t_a.sum({0, 1}, false);

    auto t_mean_0 = t_broadcast_2.mean({0}, false);
    auto t_mean_all_promoted = t_a.mean<double>({0, 1}, true);

    auto t_chain = (t_broadcast_3 + t_mean_0) + 3.14159f;

    t_scalar_1.realize();
    t_scalar_2.realize();
    t_sum_0.realize();
    t_sum_1_keep.realize();
    t_sum_all.realize();
    t_mean_all_promoted.realize();
    t_chain.realize();

    return 0;
}