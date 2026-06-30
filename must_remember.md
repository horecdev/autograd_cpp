1) You dont take by const reference& in operator+ since you would have to copy rvalues (say (a + b) * c), you cannot just take its data. You have to copy.
2) Tensors can share Storage, Storage AND m_op (alias) but never just m_op (makes zero sense)
3) m_op is a pointer, because otherwise (since its a childclass) would have its bits truncated. Memory would be only allocated for the base Node.
In addition to copy a Node you would have to copy a Tensor and to copy a Tensor you copy a node... (tho we dont copy it anywhere)
4) All nodes except CastNode return Tensor<T> and have parents of type Tensor<T>
5) All grads are contiguous because they are initialized contiguous. To ever assign anything in it, it must obey its contiguous strides. 
Everytime you accumulate a tensor, say, [5, 10, 32] of grad, it means "elem [0, 0, 0] here corresponds to grad of elem [0, 0, 0]"
It can have most crazy strides, but because of how accumulate_grad() works, it is added with strided indices, making the master buffer contiguous.
6) You cannot use lobotomize functions that allocate new storage during building a lazy graph (such as lobotomized_contiguous)
7) If a tensor and its incoming grad are both contiguous after a view (reshape etc) operation, you can just take parents strides (1 to 1 mapping and strides would be the same)
If it is not (say, Transpose) and m_parent has weird strides, swapping contiguous tensor's strides to match is suicide. Grad strides transposed != parent strides (pre-transpose) cuz its not contiguous.