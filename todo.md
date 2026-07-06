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
DONE Refactor storage to be 32-bit aligned for AVX, and hold an ENUM CPU/CUDA
DONE Update Tensor, TensorState constructors to support eager size/device propagation
DONE Reflect changes inside lazy and eager operations to use the new constructors and pass correct parameters
DONE Add assertions that devices are identical
DONE Fix the fill / write inefficiency (1GB writes for 500MB tensor for stuff with alloc) for CPU
DONE apply_unary_in_place + make contiguous use it
DONE BIG REFACTOR (hide backend from frontend)
- Update the out of, in, reduce etc. to write to uninitialized memory. think of the issue where memory is initialized and scalars.
- Write 6 dispatchers (out of, in, reduce, unary in, unary out of, layout) - everything that copies/creates new data gets a dispatcher to cuda/cpu
- Compile this big ass codebase and add missing dependencies + solve them
- SIMD contiguous fast paths (apply_in_place, apply_out_of_place, apply_unary)
- CUDA BRIDGE / UTILS .to() etc.
- Memory pools (same shape math ran thousands of times)
- CUDA