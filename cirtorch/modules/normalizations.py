import torch
import torch.nn as nn

# --------------------------------------
# Normalization layers
# --------------------------------------


class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)


class PowerLaw(nn.Module):

    def __init__(self, eps=1e-6):
        super(PowerLaw, self).__init__()
        self.eps = eps

    def forward(self, x):
        x = x + self.eps
        return x.abs().sqrt().mul(x.sign())


NORMALIZATION_LAYERS = {
    "L2N": L2N,
    "PowerLaw": PowerLaw
}

