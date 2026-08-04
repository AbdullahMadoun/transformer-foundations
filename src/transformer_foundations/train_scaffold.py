import torch
from einops import reduce
from typing import Optional
from collections.abc import Callable,Iterable
import math
import typing, os
import numpy as np 


def cross_entropy_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor: 
    """
    -log [softmax(oi)] = -log [e^oi / (sum of all e^oi)] = -(oi + log(sum of all e^oi) 

    """
    # Get oi
    input_max = torch.max(input, dim =-1, keepdim = True).values
    log_sm = input - input_max - torch.log(torch.sum(torch.exp(input - input_max) , dim = -1,  keepdim=True))
    target_logits = torch.gather(log_sm, dim = -1, index = target.unsqueeze(-1) )

    return reduce(-target_logits, "...  -> ", "mean")
class AdamW(torch.optim.Optimizer): 
    def __init__(self, params, lr = 0.001, betas =  (0.9, 0.95), eps = 1e-5 , weight_decay = 0.01):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults )
    def step(self, closure: Optional[Callable] = None): 
        loss = None if closure is None else closure() 
        for group in self.param_groups: 
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            B1, B2 = group["betas"]
            for p in group["params"]:
                g = p.grad.data
                if g is None: 
                    continue
                state = self.state[p]
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                t = state.get("t", 1)
                
                at = lr * math.sqrt(1 - B2 ** t) / (1-B1**t)
              
                p.data -= lr* weight_decay* p.data
                m = B1* m + (1- B1) *g 
                v= B2* v + (1- B2) *g**2

                p.data -= at* m/ (torch.sqrt(v) + eps)
                state["m"] = m 
                state["v"] = v
                state["t"] = t+1
        return loss 
def learning_rate_schedule(t, lr_min, lr_max, Tw, Tc):
    if t < Tw: 
        lr = t/ Tw * lr_max
    elif t < Tc: 
        lr = lr_min + 1/2* (1 + math.cos(((t- Tw)/(Tc - Tw))*math.pi))*(lr_max - lr_min)
    else: 
        lr = lr_min 
    return lr
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads: 
        return
    total_norm_sq = torch.stack([(g ** 2).sum() for g in grads]).sum()
    l2 = torch.sqrt(total_norm_sq)
    if l2 < max_l2_norm: 
        return 
    scaling = max_l2_norm / (l2  + 1e-6)
    for g in grads: 
          g.mul_(scaling)
def data_loading(x, batch_size, context_length, device ):
    rng = np.random.default_rng() 
    bound = len(x) - context_length 
    start_indices = rng.integers(low = 0, high = bound, size = batch_size  )
    inputs = []
    targets = []
    for i in start_indices: 
        chunk = x[i: i+ context_length+1]
        inputs.append(chunk[:-1])
        targets.append(chunk[1:])

    inputs = torch.from_numpy(np.array(inputs) ).to(device = device, dtype = torch.long)
    targets = torch.from_numpy(np.array(targets) ).to(device = device, dtype = torch.long)

    return (inputs, targets)
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]): 

    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration" : iteration
    }
    torch.save(state_dict, out)
def load_checkpoint(src: str| os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer): 
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    return state_dict["iteration"]
    