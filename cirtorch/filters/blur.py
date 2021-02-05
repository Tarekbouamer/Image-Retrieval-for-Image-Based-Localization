import torch.nn as nn

from .filter import filter2D

from .kernels import (
    get_box_kernel2d,
    normalize_kernel2d
)


def box_blur(input, kernel_size, border_type='reflect', normalized=True):
    """
        Blurs an image using the box filter.
        The function smooths an image using the kernel:
    """
    kernel = get_box_kernel2d(kernel_size)

    if normalized:
        kernel = normalize_kernel2d(kernel)

    return filter2D(input, kernel, border_type)


class BoxBlur(nn.Module):
    """
        Blurs an image using the box filter.
        The function smooths an image using the kernel
    """

    def __init__(self, kernel_size, border_type='reflect', normalized=True):
        super(BoxBlur, self).__init__()

        self.kernel_size = kernel_size
        self.border_type = border_type
        self.normalized = normalized

    def forward(self, input):
        return box_blur(input, self.kernel_size, self.border_type, self.normalized)
