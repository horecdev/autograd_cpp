DONE Toposort
DONE 10 Backward passes
DONE Rename to view and alloc
DONE ConcatNode + bwd pass
DONE Slicing with ranges
DONE Implement dependencies to every node.
DONE Wipe grad memory after its no longer needed (current passed its grads down) (toposort)
- Write zero_grad to tensor class
- SIMD fast paths
- AVX 32/64 byte alignment (Storage prep for CUDA)
- InPlaceOpps as disguise to normal opps (no nodes, just operator overload)
- Delete nodes on Inplace
- InPlace mutation on REALZIE TIME if refcount tensorstate == 1 and refcount storage == 1. Its guaranteed to be an rvalue living only inside AddNode. Nowehere else with no shared storage. 
- Expose InPlaceOpps to user (responsibility is on them)
- Memory pools (same shape math ran thousands of times)
- CUDA
- Refactor the utils so its not bloated (namespaces)