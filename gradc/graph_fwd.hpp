#pragma once

#include "graph.hpp"

namespace gradc {
    template <typename T> class AddNode;
    template <typename T> class MulNode;
    template <typename T> class CloneNode;
    template <typename T> class ContiguousNode;
    template <typename T> class TransposeNode;
    template <typename T> class ReshapeNode;
    template <typename T> class PermuteNode;
    template <typename T> class SliceNode;
    template <typename T> class SumNode;
    template <typename T> class MeanNode;
    template <typename InT, typename OutT> class CastNode;
    template <typename T> class ConcatNode;
}