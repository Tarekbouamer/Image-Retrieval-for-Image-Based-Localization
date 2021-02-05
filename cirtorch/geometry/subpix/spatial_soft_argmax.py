import torch
import torch.nn as nn
import torch.nn.functional as F

from .dsnt import spatial_softmax2d, spatial_expectation2d
from cirtorch.utils.grid import create_meshgrid, create_meshgrid3d
from ..conversions import normalize_pixel_coordinates, normalize_pixel_coordinates3d
from cirtorch.features.nms import nms3d
from cirtorch.filters.sobel import  spatial_gradient3d


def _get_window_grid_kernel2d(h, w, device=torch.device('cpu')):
    """
        Helper function, which generates a kernel to with window coordinates, residual to window center.
    """
    window_grid2d = create_meshgrid(h, w, False, device=device)
    window_grid2d = normalize_pixel_coordinates(window_grid2d, h, w)
    conv_kernel = window_grid2d.permute(3, 0, 1, 2)
    return conv_kernel


def _get_center_kernel2d(h, w, device=torch.device('cpu')):
    """
        Helper function, which generates a kernel to return center coordinates,
        when applied with F.conv2d to 2d coordinates grid.
    """
    center_kernel = torch.zeros(2, 2, h, w, device=device)

    #  If the size is odd, we have one pixel for center, if even - 2
    if h % 2 != 0:
        h_i1 = h // 2
        h_i2 = (h // 2) + 1
    else:
        h_i1 = (h // 2) - 1
        h_i2 = (h // 2) + 1

    if w % 2 != 0:
        w_i1 = w // 2
        w_i2 = (w // 2) + 1
    else:
        w_i1 = (w // 2) - 1
        w_i2 = (w // 2) + 1

    center_kernel[(0, 1), (0, 1), h_i1: h_i2, w_i1: w_i2] = 1.0 / float(((h_i2 - h_i1) * (w_i2 - w_i1)))

    return center_kernel


def _get_center_kernel3d(d, h, w, device=torch.device('cpu')):
    """
        Helper function, which generates a kernel to return center coordinates,
        when applied with F.conv2d to 3d coordinates grid.
    """

    center_kernel = torch.zeros(3, 3, d, h, w, device=device)

    #  If the size is odd, we have one pixel for center, if even - 2
    if h % 2 != 0:
        h_i1 = h // 2
        h_i2 = (h // 2) + 1
    else:
        h_i1 = (h // 2) - 1
        h_i2 = (h // 2) + 1

    if w % 2 != 0:
        w_i1 = w // 2
        w_i2 = (w // 2) + 1
    else:
        w_i1 = (w // 2) - 1
        w_i2 = (w // 2) + 1
    if d % 2 != 0:
        d_i1 = d // 2
        d_i2 = (d // 2) + 1

    else:
        d_i1 = (d // 2) - 1
        d_i2 = (d // 2) + 1

    center_num = float((h_i2 - h_i1) * (w_i2 - w_i1) * (d_i2 - d_i1))

    center_kernel[(0, 1, 2), (0, 1, 2), d_i1: d_i2, h_i1: h_i2, w_i1: w_i2] = 1.0 / center_num

    return center_kernel


def _get_window_grid_kernel3d(d, h, w, device=torch.device('cpu')):
    """
        Helper function, which generates a kernel to return coordinates,
        residual to window center.
    """

    grid2d = create_meshgrid(h, w, True, device=device)

    if d > 1:
        z = torch.linspace(-1, 1, d, device=device).view(d, 1, 1, 1)

    else:
        z = torch.zeros(1, 1, 1, 1, device=device)

    grid3d = torch.cat([z.repeat(1, h, w, 1).contiguous(), grid2d.repeat(d, 1, 1, 1)], dim=3)

    conv_kernel = grid3d.permute(3, 0, 1, 2).unsqueeze(1)

    return conv_kernel


class ConvSoftArgmax2d(nn.Module):
    """
        Module that calculates soft argmax 2d per window.
        `geometry.subpix.conv_soft_argmax2d` for details.
    """

    def __init__(self, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1),
                 temperature = torch.tensor(1.0), normalized_coordinates = True,
                 eps = 1e-8, output_value = False):

        super(ConvSoftArgmax2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.temperature = temperature
        self.normalized_coordinates = normalized_coordinates
        self.eps = eps
        self.output_value = output_value

    def forward(self, x):

        return conv_soft_argmax2d(x,
                                  self.kernel_size,
                                  self.stride,
                                  self.padding,
                                  self.temperature,
                                  self.normalized_coordinates,
                                  self.eps,
                                  self.output_value)


class ConvSoftArgmax3d(nn.Module):
    """
        Module that calculates soft argmax 3d per window.
        See `geometry.subpix.conv_soft_argmax3d` for details.
    """

    def __init__(self, kernel_size  = (3, 3, 3),  stride = (1, 1, 1), padding = (1, 1, 1),
                 temperature = torch.tensor(1.0), normalized_coordinates = False,
                 eps = 1e-8, output_value = True, strict_maxima_bonus = 0.0):

        super(ConvSoftArgmax3d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.temperature = temperature
        self.normalized_coordinates = normalized_coordinates
        self.eps = eps
        self.output_value = output_value
        self.strict_maxima_bonus = strict_maxima_bonus

    def forward(self, x):

        return conv_soft_argmax3d(x,
                                  self.kernel_size,
                                  self.stride,
                                  self.padding,
                                  self.temperature,
                                  self.normalized_coordinates,
                                  self.eps,
                                  self.output_value,
                                  self.strict_maxima_bonus)


def conv_soft_argmax2d(input, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), temperature=torch.tensor(1.0),
                       normalized_coordinates=True, eps=1e-8, output_value=False):
    """
        Function that computes the convolutional spatial Soft-Argmax 2D over the windows
        of a given input heatmap. Function has two outputs: argmax coordinates and the softmaxpooled heatmap values
        themselves. On each window, the function computed is
    """

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    if temperature <= 0:
        raise ValueError("Temperature should be positive float or tensor. Got: {}"
                         .format(temperature))

    b, c, h, w = input.shape
    kx, ky = kernel_size
    device = input.device
    dtype = input.dtype
    input = input.view(b * c, 1, h, w)

    center_kernel = _get_center_kernel2d(kx, ky, device).to(dtype)
    window_kernel = _get_window_grid_kernel2d(kx, ky, device).to(dtype)

    # applies exponential normalization trick
    x_max = F.adaptive_max_pool2d(input, (1, 1))

    # max is detached to prevent undesired backprop loops in the graph
    x_exp = ((input - x_max.detach()) / temperature).exp()

    # Not available yet in version 1.0, so let's do manually
    pool_coef  = float(kx * ky)

    # softmax denominator
    den = pool_coef * F.avg_pool2d(x_exp, kernel_size, stride=stride, padding=padding) + eps

    x_softmaxpool = pool_coef * F.avg_pool2d(x_exp * input, kernel_size, stride=stride, padding=padding) / den
    x_softmaxpool = x_softmaxpool.view(b, c, x_softmaxpool.size(2), x_softmaxpool.size(3))

    # We need to output also coordinates
    # Pooled window center coordinates
    grid_global = create_meshgrid(h, w, False, device).to(dtype).permute(0, 3, 1, 2)

    grid_global_pooled = F.conv2d(grid_global, center_kernel, stride=stride, padding=padding)

    # Coordinates of maxima residual to window center
    # prepare kernel
    coords_max = F.conv2d(x_exp, window_kernel, stride=stride, padding=padding)

    coords_max = coords_max / den.expand_as(coords_max)
    coords_max = coords_max + grid_global_pooled.expand_as(coords_max)

    if normalized_coordinates:
        coords_max = normalize_pixel_coordinates(coords_max.permute(0, 2, 3, 1), h, w)
        coords_max = coords_max.permute(0, 3, 1, 2)

    # Back B*C -> (b, c)
    coords_max = coords_max.view(b, c, 2, coords_max.size(2), coords_max.size(3))

    if output_value:
        return coords_max, x_softmaxpool

    return coords_max


def conv_soft_argmax3d(input, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                       temperature=torch.tensor(1.0), normalized_coordinates=False, eps=1e-8,
                       output_value=True, strict_maxima_bonus=0.0):
    """
        Function that computes the convolutional spatial Soft-Argmax 3D over the windows
        of a given input heatmap. Function has two outputs: argmax coordinates and the softmaxpooled heatmap values
        themselves. On each window, the function computed is:
    """
    if not len(input.shape) == 5:
        raise ValueError("Invalid input shape, we expect BxCxDxHxW. Got: {}"
                         .format(input.shape))

    if temperature <= 0:
        raise ValueError("Temperature should be positive float or tensor. Got: {}"
                         .format(temperature))

    b, c, d, h, w = input.shape
    kx, ky, kz = kernel_size
    device = input.device
    dtype = input.dtype
    input = input.view(b * c, 1, d, h, w)

    center_kernel = _get_center_kernel3d(kx, ky, kz, device).to(dtype)
    window_kernel = _get_window_grid_kernel3d(kx, ky, kz, device).to(dtype)

    # applies exponential normalization trick
    x_max = F.adaptive_max_pool3d(input, (1, 1, 1))

    # max is detached to prevent undesired backprop loops in the graph
    x_exp = ((input - x_max.detach()) / temperature).exp()

    pool_coef = float(kx * ky * kz)

    # softmax denominator
    den = pool_coef * F.avg_pool3d(x_exp.view_as(input), kernel_size, stride=stride, padding=padding) + eps

    # We need to output also coordinates
    # Pooled window center coordinates
    grid_global = create_meshgrid3d(d, h, w, False, device=device).to(dtype).permute(0, 4, 1, 2, 3)

    grid_global_pooled = F.conv3d(grid_global, center_kernel, stride=stride, padding=padding)

    # Coordinates of maxima residual to window center
    # prepare kernel
    coords_max = F.conv3d(x_exp, window_kernel, stride=stride, padding=padding)

    coords_max = coords_max / den.expand_as(coords_max)
    coords_max = coords_max + grid_global_pooled.expand_as(coords_max)

    if normalized_coordinates:
        coords_max = normalize_pixel_coordinates3d(coords_max.permute(0, 2, 3, 4, 1), d, h, w)
        coords_max = coords_max.permute(0, 4, 1, 2, 3)

    # Back B*C -> (b, c)
    coords_max = coords_max.view(b, c, 3, coords_max.size(2), coords_max.size(3), coords_max.size(4))

    if not output_value:
        return coords_max

    x_softmaxpool = pool_coef * F.avg_pool3d(x_exp.view(input.size()) * input, kernel_size,
                                             stride=stride,  padding=padding) / den

    if strict_maxima_bonus > 0:

        in_levels = input.size(2)

        out_levels = x_softmaxpool.size(2)

        skip_levels = (in_levels - out_levels) // 2

        strict_maxima= F.avg_pool3d(nms3d(input, kernel_size), 1, stride, 0)

        strict_maxima = strict_maxima[:, :, skip_levels:out_levels - skip_levels]

        x_softmaxpool *= 1.0 + strict_maxima_bonus * strict_maxima

    x_softmaxpool = x_softmaxpool.view(b, c, x_softmaxpool.size(2), x_softmaxpool.size(3),  x_softmaxpool.size(4))

    return coords_max, x_softmaxpool


def spatial_soft_argmax2d(input, temperature=torch.tensor(1.0),  normalized_coordinates=True, eps=1e-8):
    """
        Function that computes the Spatial Soft-Argmax 2D of a given input heatmap.
        Returns the index of the maximum 2d coordinates of the give map.
        The output order is x-coord and y-coord.
    """
    input_soft = spatial_softmax2d(input, temperature)
    output = spatial_expectation2d(input_soft, normalized_coordinates)
    return output


class SpatialSoftArgmax2d(nn.Module):
    """
        Module that computes the Spatial Soft-Argmax 2D of a given heatmap.
        See :func:`geometry.subpix.spatial_soft_argmax2d` for details.
    """

    def __init__(self, temperature=torch.tensor(1.0), normalized_coordinates=True, eps=1e-8):
        super(SpatialSoftArgmax2d, self).__init__()
        self.temperature = temperature
        self.normalized_coordinates = normalized_coordinates
        self.eps = eps

    def forward(self, input):

        return spatial_soft_argmax2d(input, self.temperature, self.normalized_coordinates, self.eps)


def conv_quad_interp3d(input: torch.Tensor, strict_maxima_bonus: float = 10.0, eps: float = 1e-7):
    """
        Function that computes the single iteration of quadratic interpolation of of the extremum (max or min) location
        and value per each 3x3x3 window which contains strict extremum, similar to one done is SIFT
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))
    if not len(input.shape) == 5:
        raise ValueError("Invalid input shape, we expect BxCxDxHxW. Got: {}"
                         .format(input.shape))

    B, CH, D, H, W = input.shape

    grid_global = create_meshgrid3d(D, H, W, False, device=input.device).permute(0, 4, 1, 2, 3)
    grid_global = grid_global.to(input.dtype)

    # to determine the location we are solving system of linear equations Ax = b, where b is 1st order gradient
    # and A is Hessian matrix

    b = spatial_gradient3d(input, order=1, mode='diff')  #
    b = b.permute(0, 1, 3, 4, 5, 2).reshape(-1, 3, 1)
    A = spatial_gradient3d(input, order=2, mode='diff')
    A = A.permute(0, 1, 3, 4, 5, 2).reshape(-1, 6)

    dxx = A[..., 0]
    dyy = A[..., 1]
    dss = A[..., 2]

    dxy = 0.25 * A[..., 3]  # normalization to match OpenCV implementation
    dys = 0.25 * A[..., 4]  # normalization to match OpenCV implementation
    dxs = 0.25 * A[..., 5]  # normalization to match OpenCV implementation

    Hes = torch.stack([dxx, dxy, dxs,
                       dxy, dyy, dys,
                       dxs, dys, dss], dim=-1).view(-1, 3, 3)

    # The following is needed to avoid singular cases
    Hes += torch.rand(Hes[0].size(), device=Hes.device).abs()[None] * eps

    nms_mask = nms3d(input, (3, 3, 3), True)

    x_solved = torch.zeros_like(b)
    x_solved_masked, _ = torch.solve(b[nms_mask.view(-1)], Hes[nms_mask.view(-1)])
    x_solved.masked_scatter_(nms_mask.view(-1, 1, 1), x_solved_masked)

    dx = -x_solved

    # Ignore ones, which are far from window center
    mask1 = (dx.abs().max(dim=1, keepdim=True)[0] > 0.7)
    dx.masked_fill_(mask1.expand_as(dx), 0)

    dy = 0.5 * torch.bmm(b.permute(0, 2, 1), dx)

    y_max = input + dy.view(B, CH, D, H, W)

    if strict_maxima_bonus > 0:
        y_max += strict_maxima_bonus * nms_mask.to(input.dtype)

    dx_res = dx.flip(1).reshape(B, CH, D, H, W, 3).permute(0, 1, 5, 2, 3, 4)
    coords_max = grid_global.repeat(B, 1, 1, 1, 1).unsqueeze(1)
    coords_max = coords_max + dx_res

    return coords_max, y_max


class ConvQuadInterp3d(nn.Module):
    """
        Module that calculates soft argmax 3d per window
        See :func:`geometry.subpix.conv_quad_interp3d` for details.
    """

    def __init__(self, strict_maxima_bonus=10.0, eps=1e-7):
        super(ConvQuadInterp3d, self).__init__()
        self.strict_maxima_bonus = strict_maxima_bonus
        self.eps = eps
        return

    def forward(self, x):
        return conv_quad_interp3d(x, self.strict_maxima_bonus, self.eps)
