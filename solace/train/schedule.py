"""Learning-rate schedule and exponential moving average of the weights."""
import numpy as np
import torch


def warmup_cosine(warmup, steps):
    """Linear warm-up, then cosine decay to 1 % of the base rate."""
    return lambda i: min(1.0, (i + 1) / warmup) * (0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * min(i, steps) / steps)))


class EMA:
    """Exponential moving average of every floating-point entry of the state dict (decay ramps up as (1 + n) / (10 + n))."""

    def __init__(self, model, decay=0.999):
        self.decay, self.n = decay, 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                self.shadow[k].copy_(v)

    def state(self):
        return {k: v.clone() for k, v in self.shadow.items()}


def pin(model, values):
    """Hold some weights at fixed values during training (Experiment A): set them and stop their gradients.
    values: {state-dict key: value}."""
    params = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in values.items():
            params[k].fill_(v)
            params[k].requires_grad_(False)
    return model
