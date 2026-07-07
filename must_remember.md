1) You dont take by const reference& in operator+ since you would have to copy rvalues (say (a + b) * c), you cannot just take its data. You have to copy.
2) Tensors can share Storage, Storage AND m_op (alias) but never just m_op (makes zero sense)
3) m_op is a pointer, because otherwise (since its a childclass) would have its bits truncated. Memory would be only allocated for the base Node.
In addition to copy a Node you would have to copy a Tensor and to copy a Tensor you copy a node... (tho we dont copy it anywhere thats why its unique_ptr)
4) All nodes except CastNode return Tensor<T> and have parents of type Tensor<T>
5) All grads are contiguous because they are initialized contiguous. To ever assign anything in it, it must obey its contiguous strides. 
Everytime you accumulate a tensor, say, [5, 10, 32] of grad, it means "elem [0, 0, 0] here corresponds to grad of elem [0, 0, 0]"
It can have most crazy strides, but because of how accumulate_grad() works, it is added with strided indices, making the master buffer contiguous.
6) You cannot use lobotomize functions that allocate new storage during building a lazy graph (such as lobotomized_contiguous_alloc)
7) Reshaping is changing where contiguous tensor's rows break. If a tensor and its incoming grad are both contiguous after a view (reshape etc) operation, you can just take parents strides (1 to 1 mapping and strides would be the same)
If it is not (say, Transpose) and m_parent has weird strides, swapping contiguous tensor's strides to match is suicide. Grad strides transposed != parent strides (pre-transpose) cuz its not contiguous.
8) in-place optimization on AddNode if is_exclusive: If you got (a + b) + c, then (a + b) is an rvalue tensor. Nobody holds a map to it,
unlike if it was temp = a + b and res = temp + c. Since m_left is only ever used to produce end result during realize, you can safely edit it because
m_left inside node is the only place in universe where it exists. Nobody else will know, and addnode doesnt need it for anything else THEREFORE nobody needs it.
9) m_storage of TensorState must be initialized with empty shared_ptr and not nullptr. If c = a + b and c has a nullptr to storage, then d = c.reshape happens, 
and d keeps the nullptr. ReshapeNode has a copy of c, also with a nullptr. 
When realize happens, c storage is filled with data. ReshapeNode has none, d has none, but should. Everything blows up.
If there is a shared_ptr, its inside c, ReshapeNode, and d. Change immediately gets reflected.
10) Every apply function supposes memory is already allocated. It can be initialized for InPlace, or uninitialized for everytihng other (including Reduce)
11) Frontend asserts every single m_parent, m_left, m_right etc. is on the same device as result and tensor that will get result
12) Since frontend asserts that, accumulate_grad means result accumulates to m_parent, so grads are also on the right device.