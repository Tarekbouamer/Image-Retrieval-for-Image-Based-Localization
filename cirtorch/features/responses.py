import torch
import torch.nn as nn

from ..filters.sobel import spatial_gradient
from ..filters.gaussian import gaussian_blur2d


def harris_response(input, k=0.04, grads_mode='sobel', sigmas=None):
    """
        Computes the Harris cornerness function. Function does not do any normalization or nms.
    """
    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    if sigmas is not None:
        if not torch.is_tensor(sigmas):
            raise TypeError("sigmas type is not a torch.Tensor. Got {}"
                            .format(type(sigmas)))
        if (not len(sigmas.shape) == 1) or (sigmas.size(0) != input.size(0)):
            raise ValueError("Invalid sigmas shape, we expect B == input.size(0). Got: {}".format(sigmas.shape))

    gradients = spatial_gradient(input, grads_mode)

    dx = gradients[:, :, 0]
    dy = gradients[:, :, 1]

    # compute the structure tensor M elements
    def g(x):
        return gaussian_blur2d(x, (7, 7), (1., 1.))

    dx2 = g(dx ** 2)
    dy2 = g(dy ** 2)
    dxy = g(dx * dy)
    det_m = dx2 * dy2 - dxy * dxy
    trace_m = dx2 + dy2

    # compute the response map
    scores = det_m - k * (trace_m ** 2)

    if sigmas is not None:
        scores = scores * sigmas.pow(4).view(-1, 1, 1, 1)

    return scores


def gftt_response(input, grads_mode='sobel', sigmas=None):
    """
        Computes the Shi-Tomasi cornerness function. Function does not do any normalization or nms.
    """

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))
    gradients = spatial_gradient(input, grads_mode)
    dx = gradients[:, :, 0]
    dy = gradients[:, :, 1]

    # compute the structure tensor M elements
    def g(x):
        return gaussian_blur2d(x, (7, 7), (1., 1.))

    dx2 = g(dx ** 2)
    dy2 = g(dy ** 2)
    dxy = g(dx * dy)

    det_m = dx2 * dy2 - dxy * dxy
    trace_m = dx2 + dy2

    e1 = 0.5 * (trace_m + torch.sqrt((trace_m ** 2 - 4 * det_m).abs()))
    e2 = 0.5 * (trace_m - torch.sqrt((trace_m ** 2 - 4 * det_m).abs()))

    scores = torch.min(e1, e2)
    if sigmas is not None:
        scores = scores * sigmas.pow(4).view(-1, 1, 1, 1)

    return scores


def hessian_response(input, grads_mode='sobel', sigmas=None):
    """
        Computes the absolute of determinant of the Hessian matrix. Function does not do any normalization or nms.
    """
    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    if sigmas is not None:
        if (not len(sigmas.shape) == 1) or (sigmas.size(0) != input.size(0)):
            raise ValueError("Invalid sigmas shape, we expect B == input.size(0). Got: {}"
                             .format(sigmas.shape))

    gradients = spatial_gradient(input, grads_mode, 2)
    dxx = gradients[:, :, 0]
    dxy = gradients[:, :, 1]
    dyy = gradients[:, :, 2]

    scores = dxx * dyy - dxy ** 2

    if sigmas is not None:
        scores = scores * sigmas.pow(4).view(-1, 1, 1, 1)

    return scores


def dog_response(input):
    """
        Computes the Difference-of-Gaussian response given the Gaussian 5d input:
    """

    if not len(input.shape) == 5:
        raise ValueError("Invalid input shape, we expect BxCxDxHxW. Got: {}"
                         .format(input.shape))

    return input[:, :, 1:] - input[:, :, :-1]


class BlobDoG(nn.Module):
    """
        nn.Module that calculates Difference-of-Gaussians blobs
        See :func:`feature.dog_response` for details.
    """

    def __init__(self):
        super(BlobDoG, self).__init__()
        return

    def forward(self, input, sigmas=None):
        return dog_response(input)


class CornerHarris(nn.Module):
    """
        nn.Module that calculates Harris corners
        See :func:`feature.harris_response` for details.
    """

    def __init__(self, k, grads_mode='sobel'):
        super(CornerHarris, self).__init__()

        if type(k) is float:
            self.register_buffer('k', torch.tensor(k))

        else:
            self.register_buffer('k', k)  # type: ignore

        self.grads_mode: str = grads_mode

        return

    def forward(self, input, sigmas=None):

        return harris_response(input, self.k, self.grads_mode, sigmas)


class CornerGFTT(nn.Module):
    """
        nn.Module that calculates Shi-Tomasi corners
        See :func:`.feature.gfft_response` for details.
    """

    def __init__(self, grads_mode='sobel'):
        super(CornerGFTT, self).__init__()
        self.grads_mode: str = grads_mode
        return

    def forward(self, input, sigmas=None):
        return gftt_response(input, self.grads_mode, sigmas)


class BlobHessian(nn.Module):
    """
        nn.Module that calculates Hessian blobs
        See :func:`.feature.hessian_response` for details.
    """

    def __init__(self, grads_mode='sobel') -> None:
        super(BlobHessian, self).__init__()
        self.grads_mode: str = grads_mode
        return

    def forward(self, input, sigmas=None):
        return hessian_response(input, self.grads_mode, sigmas)