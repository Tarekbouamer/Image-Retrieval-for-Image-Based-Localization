import torch
import torch.nn as nn

from .filter import filter2D
from .kernels import get_gaussian_kernel2d


def gaussian_blur2d(input, kernel_size, sigma, border_type='reflect'):
    """
        Creates an operator that blurs a tensor using a Gaussian filter.
        The operator smooths the given tensor with a gaussian kernel by convolving
        it to each channel. It supports batched operation.
    """
    kernel = torch.unsqueeze(get_gaussian_kernel2d(kernel_size, sigma), dim=0)

    return filter2D(input, kernel, border_type)


class GaussianBlur2d(nn.Module):
    """
        Creates an operator that blurs a tensor using a Gaussian filter.
        The operator smooths the given tensor with a gaussian kernel by convolving
        it to each channel. It supports batched operation.
    """

    def __init__(self, kernel_size, sigma, border_type='reflect'):
        super(GaussianBlur2d, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.border_type = border_type

    def forward(self, input):
        return gaussian_blur2d(input, self.kernel_size, self.sigma, self.border_type)
