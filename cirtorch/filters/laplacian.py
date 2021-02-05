import torch
import torch.nn as nn

from .kernels import get_laplacian_kernel2d, normalize_kernel2d
from .filter import filter2D


def laplacian(input, kernel_size, border_type='reflect', normalized=True):
    """
        Creates an operator that returns a tensor using a Laplacian filter.
        The operator smooths the given tensor with a laplacian kernel by convolving
        it to each channel. It supports batched operation.
    """
    kernel = torch.unsqueeze(get_laplacian_kernel2d(kernel_size), dim=0)

    if normalized:
        kernel = normalize_kernel2d(kernel)

    return filter2D(input, kernel, border_type)


class Laplacian(nn.Module):
    """
        Creates an operator that returns a tensor using a Laplacian filter.
        The operator smooths the given tensor with a laplacian kernel by convolving
        it to each channel. It supports batched operation.
    """

    def __init__(self, kernel_size, border_type='reflect', normalized=True):
        super(Laplacian, self).__init__()
        self.kernel_size = kernel_size
        self.border_type = border_type
        self.normalized = normalized

    def forward(self, input):
        return laplacian(input, self.kernel_size, self.border_type, self.normalized)
