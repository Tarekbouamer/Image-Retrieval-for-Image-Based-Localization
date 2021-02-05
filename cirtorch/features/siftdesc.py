import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from ..filters.gaussian import get_gaussian_kernel2d
from ..filters.sobel import spatial_gradient

from ..geometry.conversions import pi


def get_sift_pooling_kernel(ksize=25):
    """
        Returns a weighted pooling kernel for SIFT descriptor
    """
    ks_2 = float(ksize) / 2.0
    xc2 = ks_2 - (torch.arange(ksize).float() + 0.5 - ks_2).abs()
    kernel = torch.ger(xc2, xc2) / (ks_2**2)
    return kernel


def get_sift_bin_ksize_stride_pad(patch_size, num_spatial_bins):
    """
        Returns a tuple with SIFT parameters, given the patch size and number of spatial bins.
    """

    ksize = 2 * int(patch_size / (num_spatial_bins + 1))
    stride = patch_size // num_spatial_bins
    pad = ksize // 4
    out_size = (patch_size + 2 * pad - (ksize - 1) - 1) // stride + 1

    if out_size != num_spatial_bins:
        raise ValueError(f"Patch size {patch_size} is incompatible with \
            requested number of spatial bins {num_spatial_bins} \
            for SIFT descriptor. Usually it happens when patch size is too small\
            for num_spatial_bins specified")

    return ksize, stride, pad


class SIFTDescriptor(nn.Module):
    """
        Module, which computes SIFT descriptors of given patches
    """
    def __init__(self, patch_size=41, num_ang_bins=8, num_spatial_bins=4, rootsift=True, clipval=0.2):
        super(SIFTDescriptor, self).__init__()

        self.eps = 1e-10
        self.num_ang_bins = num_ang_bins
        self.num_spatial_bins = num_spatial_bins
        self.clipval = clipval
        self.rootsift = rootsift
        self.patch_size = patch_size

        ks = self.patch_size
        sigma = float(ks) / math.sqrt(2.0)

        self.gk = get_gaussian_kernel2d((ks, ks), (sigma, sigma), True)

        (self.bin_ksize, self.bin_stride, self.pad) = get_sift_bin_ksize_stride_pad(patch_size, num_spatial_bins)

        nw = get_sift_pooling_kernel(ksize=self.bin_ksize).float()

        self.pk = nn.Conv2d(1, 1, kernel_size=(nw.size(0), nw.size(1)),
                            stride=(self.bin_stride, self.bin_stride),
                            padding=(self.pad, self.pad),
                            bias=False)

        self.pk.weight.data.copy_(nw.reshape(1, 1, nw.size(0), nw.size(1)))

        return

    def get_pooling_kernel(self):
        return self.pk.weight.detach()

    def get_weighting_kernel(self):
        return self.gk.detach()

    def forward(self, input):

        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect Bx1xHxW. Got: {}"
                             .format(input.shape))

        B, CH, W, H = input.size()

        if (W != self.patch_size) or (H != self.patch_size) or (CH != 1):
            raise TypeError(
                "input shape should be must be [Bx1x{}x{}]. "
                "Got {}".format(self.patch_size, self.patch_size, input.size()))

        self.pk = self.pk.to(input.dtype).to(input.device)

        grads = spatial_gradient(input, 'diff')

        # unpack the edges
        gx = grads[:, :, 0]
        gy = grads[:, :, 1]

        mag = torch.sqrt(gx * gx + gy * gy + self.eps)
        ori = torch.atan2(gy, gx + self.eps) + 2.0 * pi
        mag = mag * self.gk.expand_as(mag).type_as(mag).to(mag.device)

        o_big = float(self.num_ang_bins) * ori / (2.0 * pi)

        bo0_big_ = torch.floor(o_big)
        wo1_big_ = (o_big - bo0_big_)
        bo0_big = bo0_big_ % self.num_ang_bins
        bo1_big = (bo0_big + 1) % self.num_ang_bins
        wo0_big = (1.0 - wo1_big_) * mag  # type: ignore
        wo1_big = wo1_big_ * mag

        ang_bins = []

        for i in range(0, self.num_ang_bins):
            out = self.pk((bo0_big == i).to(input.dtype) * wo0_big +  # noqa
                          (bo1_big == i).to(input.dtype) * wo1_big)
            ang_bins.append(out)

        ang_bins = torch.cat(ang_bins, dim=1)
        ang_bins = ang_bins.view(B, -1)
        ang_bins = F.normalize(ang_bins, p=2)

        ang_bins = torch.clamp(ang_bins, 0., float(self.clipval))
        ang_bins = F.normalize(ang_bins, p=2)

        if self.rootsift:
            ang_bins = torch.sqrt(F.normalize(ang_bins, p=1) + self.eps)

        return ang_bins


def sift_describe(input, patch_size=41, num_ang_bins=8, num_spatial_bins=4, rootsift=True, clipval=0.2):
    """
        Computes the sift descriptor.
    """

    return SIFTDescriptor(patch_size, num_ang_bins, num_spatial_bins, rootsift, clipval)(input)
