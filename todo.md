DONE Toposort
DONE 10 Backward passes
DONE Rename to view and alloc
DONE ConcatNode + bwd pass
DONE Slicing with ranges
DONE Implement dependencies to every node.
DONE Wipe grad memory after its no longer needed (current passed its grads down) (toposort)
DONE Write zero_grad to tensor class
DONE InPlaceOpps as disguise to normal opps (no nodes, just operator overload)
DONE Delete nodes on Inplace
DONE test out the bwd pass.
DONE InPlace mutation on REALZIE TIME if refcount tensorstate == 1 and refcount storage == 1. Its guaranteed to be an rvalue living only inside AddNode. Add to ALL applicable
DONE Check if requires grad to do calculation inside node backward() function
DONE Remove m_version from existence
DONE Unary nodes (ReLU) + framework
DONE Update Tensor, TensorState and Storage constructors to support eager size/device propagation
DONE Reflect changes inside lazy and eager operations to use the new constructors and pass correct parameters
- Add assertions that devices are identical
- Update all functions that actually operate on memory
- Fix the fill / write inefficiency (1GB writes for 500MB tensor for stuff with alloc)
- SIMD contiguous fast paths (lobotomized_contiguous_alloc, apply_in_place, apply_out_of_place, apply_unary)
- AVX 32/64 byte alignment (Storage prep for CUDA)
- Memory pools (same shape math ran thousands of times)
- CUDA
- Refactor the utils so its not bloated (namespaces)