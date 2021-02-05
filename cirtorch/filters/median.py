import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels import get_binary_kernel2d


def _compute_zero_padding(kernel_size):
    """
        Utility function that computes zero padding tuple.
    """

    computed = [(k - 1) // 2 for k in kernel_size]

    return computed[0], computed[1]


def median_blur(input, kernel_size):
    """
        Blurs an image using the median filter.
    """
    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    padding = _compute_zero_padding(kernel_size)

    # prepare kernel
    kernel = get_binary_kernel2d(kernel_size).to(input)
    b, c, h, w = input.shape

    # map the local window to single vector
    features = F.conv2d(input.reshape(b * c, 1, h, w), kernel, padding=padding, stride=1)
    features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

    # compute the median along the feature axis
    median = torch.median(features, dim=2)[0]

    return median


class MedianBlur(nn.Module):
    """
        Blurs an image using the median filter.
    """

    def __init__(self, kernel_size):
        super(MedianBlur, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, input):
        return median_blur(input, self.kernel_size)
