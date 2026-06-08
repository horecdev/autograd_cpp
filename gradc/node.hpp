#pragma once

namespace gradc {

    template <typename T>
    class Tensor;

    template <typename T>
    class Node {
        public:
            virtual Tensor<T> realize() = 0;
            virtual void backward() = 0;

            virtual ~Node() {}
    };
}