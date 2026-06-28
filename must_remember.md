1) You dont take by const reference& in operator+ since you would have to copy rvalues (say (a + b) * c), you cannot just take its data. You have to copy.
2) Tensors can share Storage, Storage AND m_op (alias) but never just m_op (makes zero sense)
3) m_op is a pointer, because otherwise (since its a childclass) would have its bits truncated. Memory would be only allocated for the base Node.
In addition to copy a Node you would have to copy a Tensor and to copy a Tensor you copy a node... (tho we dont copy it anywhere)
4) All nodes except CastNode return Tensor<T> and have parents of type Tensor<T>
5) All grads are contiguous because they are initialized contiguous. To ever assign anything in it, it must obey its contiguous strides. 
Everytime you accumulate a tensor, say, [5, 10, 32] of grad, it means "elem [0, 0, 0] here corresponds to grad of elem [0, 0, 0]"
It can have most crazy strides, but because of how accumulate_grad() works, it is added with strided indices, making the master buffer contiguous.
6) Tensor operations that are just views (SliceNode, TransposeNode) do not need lazy initialization. They do not allocate new storage. Can just share their data
with parent (copy storage shared ptr). The storage is filled during realize().