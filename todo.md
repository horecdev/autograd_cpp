DONE Toposort
- 10 Backward passes
- SIMD fast paths
- AVX 32/64 byte alignment
- Compilator basically (Swap out AddNodes, InPlaceAddNodes. Remove redundant memory?)
- CUDA

Immediate: SliceNode finish - write the slice helper, put it in bracket operator, write the backwar pass (zero fill)