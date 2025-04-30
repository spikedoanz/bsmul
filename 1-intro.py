"""
a matrix A : (n,n) with det = 0 can be 
decomposed into v ⊗ w where v,w : (n)

a matmul A @ B can thus be decomposed into
v ⊗ w @ x ⊗ y
which simplifies to
(w.T @ x)          v ⊗ y
^^^^^^^^^          ^^^^^
dot product        a matrix (n,n)
→ scalar : (1)

which brings the complexity
compute : O(n^3)    ->  O(n + n^2)
space   : O(2n^2)   ->  O(4n)
"""

from tinygrad import Tensor

einsum = Tensor.einsum

N = 1000

v = Tensor.randn(N)
w = Tensor.randn(N)
x = Tensor.randn(N)
y = Tensor.randn(N)

A = einsum("i,j -> i j", v, w)
B = einsum("i,j -> i j", x, y)

vanilla = A @ B
bsmul   = (w @ x) * einsum("i,j -> i j", v, y)

print(all(vanilla.isclose(bsmul).flatten().tolist()))
