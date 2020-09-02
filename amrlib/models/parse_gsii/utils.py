import math
import torch
import numpy as np


def move_to_device(maybe_tensor, device):
    if torch.is_tensor(maybe_tensor):
        return maybe_tensor.to(device)
    elif isinstance(maybe_tensor, np.ndarray):
        return torch.from_numpy(maybe_tensor).to(device).contiguous()
    elif isinstance(maybe_tensor, dict):
        return {
            key: move_to_device(value, device)
            for key, value in maybe_tensor.items()
        }
    elif isinstance(maybe_tensor, list):
        return [move_to_device(x, device) for x in maybe_tensor]
    elif isinstance(maybe_tensor, tuple):
        return tuple([move_to_device(x, device) for x in maybe_tensor])
    return maybe_tensor

def compute_f_by_tensor(input, target, mask):
    input = input.view(-1).tolist()
    target = target.view(-1).tolist()
    mask = mask.view(-1).tolist()
    tp, fp, tn, fn = 0., 0., 0., 0.
    for i, t, m in zip(input, target, mask):
        if m == 1:
            continue
        else:
            if i == 1:
                if t == 1:
                    tp +=1
                else:
                    fp +=1
            else:
                if t == 1:
                    fn +=1
                else:
                    tn +=1
    if tp == 0:
        return 0., 0., 0.

    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F = 2*P*R/(P+R)
    return P, R, F

def gelu_fast(x):
    if not hasattr(gelu_fast, "_a"):
        gelu_fast._a = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + torch.tanh(gelu_fast._a * (x + 0.044715 * torch.pow(x, 3))))

def gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def label_smoothed_nll_loss(log_probs, target, eps):
    #log_probs: N x C
    #target: N
    nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
    if eps == 0.:
        return nll_loss
    smooth_loss = -log_probs.sum(dim=-1)
    eps_i = eps / log_probs.size(-1)
    loss = (1. - eps) * nll_loss + eps_i * smooth_loss
    return loss
