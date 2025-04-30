"""
a matrix A : (n,n) with det = 0 can be 
decomposed into ∑_i v_i ⊗ w_i where v,w : (n)

a matmul A @ B can thus be decomposed into
∑_i ∑_j v_i ⊗ w_i @ x_j ⊗ y_j
which simplifies to
∑           ((w.T @ x)   *     v ⊗ y)
element     ^^^^^^^^^          ^^^^^
wise        dot product        a matrix (n,n)
addition

which has the work complexity:
compute : O(n^3)    ->  O((r+s)n + rsn)
space   : O(n^2)    ->  O(r*n + s*n)

and depth complexity:
        : O(logn)   ->  O(log(max(r,s)))
"""

from tinygrad import Tensor

einsum = Tensor.einsum

def bsmul(V: Tensor, W: Tensor, X: Tensor, Y: Tensor) -> Tensor:
    """
    Compute  (Σ_i  v_i ⊗ w_i)  @  (Σ_j  x_j ⊗ y_j)
            └─── A (rank-r) ┘     └── B (rank-s) ┘
    using O((r+s)n + rsn) work instead of O(n³).

    Shapes
    -------
    V, W : (r, n)   →   A = Σ_i V[i] ⊗ W[i]
    X, Y : (s, n)   →   B = Σ_j X[j] ⊗ Y[j]
    r, s : can be interpreted as rank

    Returns
    -------
    C : (n, n)      =   A @ B
    """

    # 1) All pairwise inner products   d_{ij} = w_i · x_j     (shape (r, s))
    # 2) Weighted outer-sums           ∑_{i,j} d_{ij} v_i ⊗ y_j
    D = einsum("ri,s i -> r s", W, X)
    return einsum("rs, r i, s j -> i j", D, V, Y)

if __name__ == "__main__":
  n, r, s = 100, 2, 2
  atol = 1e-4 # this is numerically **very** different from vanilla matmul
  V, W = Tensor.randn(r, n), Tensor.randn(r, n)
  X, Y = Tensor.randn(s, n), Tensor.randn(s, n)

  # low-rank product
  C_low = bsmul(V, W, X, Y)

  # “vanilla” dense product
  A = einsum("ri, rj -> i j", V, W)           # build dense A (n×n)
  B = einsum("si, sj -> i j", X, Y)           # build dense B (n×n)
  C_dense = A @ B

  print(C_low.isclose(C_dense, atol=atol).all().item())  # → True
