import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cirtorch.filters.filter import filter2D
from cirtorch.filters.gaussian import gaussian_blur2d

__all__ = [

    "ScalePyramid",
    "pyrdown"
]


def _get_pyramid_gaussian_kernel():
    """
        Utility function that return a pre-computed gaussian kernel.
    """
    return torch.tensor([[
        [1.,    4.,     6.,     4.,     1.],
        [4.,    16.,    24.,    16.,    4.],
        [6.,    24.,    36.,    24.,    6.],
        [4.,    16.,    24.,    16.,    4.],
        [1.,    4.,     6.,     4.,     1.]
    ]]) / 256.


class ScalePyramid(nn.Module):
    """
        Creates an scale pyramid of image, usually used for local feature detection.
        Images are consequently smoothed with Gaussian blur and downscaled.

    Arguments:
        n_levels (int): number of the levels in octave.
        init_sigma (float): initial blur level.
        min_size (int): the minimum size of the octave in pixels. Default is 5
        double_image (bool): add 2x upscaled image as 1st level of pyramid. OpenCV SIFT does this. Default is False
    Returns:
        Tuple(List(Tensors), List(Tensors), List(Tensors)):
        1st output: images
        2nd output: sigmas (coefficients for scale conversion)
        3rd output: pixelDists (coefficients for coordinate conversion)

    """

    def __init__(self, levels=3, sigma=1.6, min_size=15, double_image=False):

        super(ScalePyramid, self).__init__()

        self.levels = levels
        self.sigma = sigma
        self.min_size = min_size
        self.double_image = double_image

        self.extra_levels = 3
        self.border = min_size // 2 - 1
        self.sigma_step = 2 ** (1. / float(self.levels))
        return

    def get_kernel_size(self, sigma):
        #  matches OpenCV, but may cause padding problem for small images
        #  PyTorch does not allow to pad more than original size.
        #  Therefore there is a hack in forward function

        ksize = int(2.0 * 4.0 * sigma + 1.0)

        if ksize % 2 == 0:
            ksize += 1

        return ksize

    def get_first_level(self, input):
        pixel_distance = 1.0
        cur_sigma = 0.5

        # Same as in OpenCV up to interpolation difference
        if self.double_image:
            x = F.interpolate(input, scale_factor=2.0, mode='bilinear', align_corners=False)
            pixel_distance = 0.5
            cur_sigma *= 2.0
        else:
            x = input

        if self.sigma > cur_sigma:

            sigma = max(math.sqrt(self.sigma**2 - cur_sigma**2), 0.01)
            ksize = self.get_kernel_size(sigma)

            cur_level = gaussian_blur2d(x, (ksize, ksize), (sigma, sigma))
            cur_sigma = self.sigma
        else:
            cur_level = x

        return cur_level, cur_sigma, pixel_distance

    def forward(self, x):

        batch_size, Ch, H, W = x.size()

        cur_level, cur_sigma, pixel_distance = self.get_first_level(x)

        sigmas = [cur_sigma * torch.ones(batch_size, self.levels + self.extra_levels).to(x.device).to(x.dtype)]

        pixel_dists = [pixel_distance * torch.ones(batch_size, self.levels + self.extra_levels).to(x.device).to(x.dtype)]

        pyr = [[cur_level]]

        oct_idx = 0

        while True:
            cur_level = pyr[-1][0]

            for level_idx in range(1, self.levels + self.extra_levels):

                sigma = cur_sigma * math.sqrt(self.sigma_step**2 - 1.0)

                ksize = self.get_kernel_size(sigma)

                # Hack, because PyTorch does not allow to pad more than original size.
                # But for the huge sigmas, one needs huge kernel and padding...

                ksize = min(ksize, min(cur_level.size(2), cur_level.size(3)))

                if ksize % 2 == 0:
                    ksize += 1

                cur_level = gaussian_blur2d(cur_level, (ksize, ksize), (sigma, sigma))

                cur_sigma *= self.sigma_step
                pyr[-1].append(cur_level)

                sigmas[-1][:, level_idx] = cur_sigma

                pixel_dists[-1][:, level_idx] = pixel_distance

            nextOctaveFirstLevel = F.interpolate(pyr[-1][-self.extra_levels], scale_factor=0.5, mode='nearest')

            # Nearest matches OpenCV SIFT
            pixel_distance *= 2.0
            cur_sigma = self.init_sigma

            if (min(nextOctaveFirstLevel.size(2),
                    nextOctaveFirstLevel.size(3)) <= self.min_size):
                break

            pyr.append([nextOctaveFirstLevel])

            sigmas.append(cur_sigma * torch.ones(batch_size, self.levels + self.extra_levels).to(x.device))

            pixel_dists.append(pixel_distance * torch.ones(batch_size, self.levels + self.extra_levels).to(x.device))
            oct_idx += 1

        for i in range(len(pyr)):
            pyr[i] = torch.stack(pyr[i], dim=2)

        return pyr, sigmas, pixel_dists


def pyrdown(input, border_type='reflect', align_corners=False):
    """
        Blurs a tensor and downsamples it.
    """
    return PyrDown(border_type, align_corners)(input)


class PyrDown(nn.Module):
    """
        Blurs a tensor and downsamples it.
    """

    def __init__(self, border_type='reflect', align_corners=False):
        super(PyrDown, self).__init__()

        self.border_type = border_type
        self.kernel = _get_pyramid_gaussian_kernel()
        self.align_corners = align_corners

    def forward(self, input):

        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))

        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))

        # blur image
        x_blur = filter2D(input, self.kernel, self.border_type)

        # downsample.
        out = F.interpolate(x_blur, scale_factor=0.5, mode='bilinear', align_corners=self.align_corners)

        return out