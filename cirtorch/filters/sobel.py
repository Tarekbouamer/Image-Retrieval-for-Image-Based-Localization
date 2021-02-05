import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels import (
    get_spatial_gradient_kernel2d,
    get_spatial_gradient_kernel3d,
    normalize_kernel2d
)


def spatial_gradient(input, mode='sobel', order=1, normalized=True):
    """
        Computes the first order image derivative in both x and y using a Sobel operator.
    """

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))
    # allocate kernel
    kernel = get_spatial_gradient_kernel2d(mode, order)

    if normalized:
        kernel = normalize_kernel2d(kernel)

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel.to(input).detach()
    tmp_kernel = tmp_kernel.unsqueeze(1).unsqueeze(1)

    # convolve input tensor with sobel kernel
    kernel_flip = tmp_kernel.flip(-3)

    # Pad with "replicate for spatial dims, but with zeros for channel
    spatial_pad = [
        kernel.size(1) // 2,
        kernel.size(1) // 2,
        kernel.size(2) // 2,
        kernel.size(2) // 2
    ]
    out_channels = 3 if order == 2 else 2

    padded_inp = F.pad(input.reshape(b * c, 1, h, w), spatial_pad, 'replicate')[:, :, None]

    return F.conv3d(padded_inp, kernel_flip, padding=0).view(b, c, out_channels, h, w)


def spatial_gradient3d(input, mode='diff', order=1):
    """
        Computes the first and second order volume derivative in x, y and d using a diff operator.
    """
    if not len(input.shape) == 5:
        raise ValueError("Invalid input shape, we expect BxCxDxHxW. Got: {}"
                         .format(input.shape))
    # allocate kernel
    kernel = get_spatial_gradient_kernel3d(mode, order)

    # prepare kernel
    b, c, d, h, w = input.shape
    tmp_kernel = kernel.to(input).detach()
    tmp_kernel = tmp_kernel.repeat(c, 1, 1, 1, 1)

    # convolve input tensor with grad kernel
    kernel_flip = tmp_kernel.flip(-3)

    # Pad with "replicate for spatial dims, but with zeros for channel
    spatial_pad = [
        kernel.size(2) // 2,
        kernel.size(2) // 2,
        kernel.size(3) // 2,
        kernel.size(3) // 2,
        kernel.size(4) // 2,
        kernel.size(4) // 2
    ]
    out_ch = 6 if order == 2 else 3

    return F.conv3d(F.pad(
        input, spatial_pad, 'replicate'), kernel_flip, padding=0, groups=c).view(b, c, out_ch, d, h, w)


def sobel(input, normalized=True, eps=1e-6):
    """
        Computes the Sobel operator and returns the magnitude per channel.
    """

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    # comput the x/y gradients
    edges = spatial_gradient(input, normalized=normalized)

    # unpack the edges
    gx = edges[:, :, 0]
    gy = edges[:, :, 1]

    # compute gradient maginitude
    magnitude = torch.sqrt(gx * gx + gy * gy + eps)

    return magnitude


class SpatialGradient(nn.Module):
    """
        Computes the first order image derivative in both x and y using a Sobel operator.
    """

    def __init__(self, mode='sobel', order=1, normalized=True):
        super(SpatialGradient, self).__init__()

        self.normalized = normalized
        self.order = order
        self.mode = mode

    def forward(self, input):
        return spatial_gradient(input, self.mode, self.order, self.normalized)


class SpatialGradient3d(nn.Module):
    """
        Computes the first and second order volume derivative in x, y and d using a diff operator.
    """

    def __init__(self, mode='diff', order=1):

        super(SpatialGradient3d, self).__init__()
        self.order = order
        self.mode = mode
        self.kernel = get_spatial_gradient_kernel3d(mode, order)

    def forward(self, input):
        return spatial_gradient3d(input, self.mode, self.order)


class Sobel(nn.Module):
    """
        Computes the Sobel operator and returns the magnitude per channel.
    """

    def __init__(self, normalized=True, eps=1e-6):
        super(Sobel, self).__init__()
        self.normalized = normalized
        self.eps = eps

    def forward(self, input):
        return sobel(input, self.normalized, self.eps)
