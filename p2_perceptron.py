"""
an mlp  (W @ x + b).relu() 
can thus be factored into
        ((v ⊗ w @ x) + b).relu()
which using the trick from 1-intro, is
        ((w @ x) ⊗ v + b).relu()
>   i don't think the relu can be factored in, 
    since the sum creates partial dependencies.
    -------------------------------------------
    but this has huge gains for memory, and kernel
    fusion, since this has memory O(n) the whole way
    through.

>   Question: how does this extend to batched inputs?
>   Question: how does this extend to MoEs?
"""

import math
from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.helpers import getenv, colored, trange
from tinygrad.nn.datasets import mnist

from p1_intro import bsmul

einsum = Tensor.einsum

class Linear:
  def __init__(self, in_features:int, out_features:int, bias=True):
    bound = 1 / math.sqrt(in_features)
    self.weight = Tensor.uniform(
      out_features, 
      in_features, 
      low=-bound, high=bound)
    self.bias = Tensor.uniform(
      out_features, 
      low=-bound, high=bound
    ) if bias else None

  def __call__(self, x:Tensor) -> Tensor: 
    return x.linear(self.weight.transpose(), self.bias)

class BSLinear:
  def __init__(self, dim: int, bias=True):
    bound = 1 / math.sqrt(dim)
    self.v = Tensor.uniform(dim, low=-bound, high=bound)
    self.w = Tensor.uniform(dim, low=-bound, high=bound)
    self.bias = Tensor.uniform(dim, low=-bound, high=bound) if bias else None

  def __call__(self, x:Tensor) -> Tensor: 
    return einsum("i,j -> i j", (x @ self.w), self.v) + self.bias

class Model:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
      lambda x: x.rearrange("... 1 h w -> ... (h w)"),  
      Linear(28*28, 28*28), 
      Tensor.relu,
      Linear(28*28, 10)
    ]

  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)

class BSModel:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
      lambda x: x.rearrange("... 1 h w -> ... (h w)"),  
      BSLinear(28*28), 
      Tensor.relu,
      Linear(28*28, 10)
    ]

  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)

# -- Training ---------------------------------------------------------------

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = mnist(fashion=getenv("FASHION"))

  model = BSModel() if getenv("BS") > 0 else Model()
  opt = nn.optim.Adam(nn.state.get_parameters(model))

  @TinyJit
  @Tensor.train()
  def train_step() -> Tensor:
    opt.zero_grad()
    samples = Tensor.randint(getenv("BS", 512), high=X_train.shape[0])
    out = model(X_train[samples])
    loss = out.sparse_categorical_crossentropy(Y_train[samples]).backward()
    opt.step()
    return loss

  @TinyJit
  @Tensor.test()
  def get_test_acc() -> Tensor: 
      return (model(X_test).argmax(axis=1) == Y_test).mean()*100

  test_acc = float('nan')
  for i in (t:=trange(getenv("STEPS", 70))):
    GlobalCounters.reset()   # NOTE: this makes it nice for DEBUG=2 timing
    loss = train_step()
    if i%10 == 9: test_acc = get_test_acc().item()
    t.set_description(
      f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%"
    )

  # verify eval acc
  if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
    if test_acc >= target and test_acc != 100.0: 
      print(colored(f"{test_acc=} >= {target}", "green"))
    else: raise ValueError(colored(f"{test_acc=} < {target}", "red"))
