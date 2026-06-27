1) You dont take by const reference& in operator+ since you would have to copy rvalues (say (a + b) * c), you cannot just take its data. You have to copy.
2) Tensors can share Storage, Storage AND m_op (alias) but never just m_op (makes zero sense)
3) m_op is a pointer, because otherwise (since its a childclass) would have its bits truncated. Memory would be only allocated for the base Node.
In addition to copy a Node you would have to copy a Tensor and to copy a Tensor you copy a node... (tho we dont copy it anywhere)