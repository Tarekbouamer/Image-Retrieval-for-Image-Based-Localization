from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cirtorch.utils.grid import create_meshgrid, create_meshgrid3d
from ..linalg import transform_points

__all__ = [
    "HomographyWarper",
    "homography_warp",
    "homography_warp3d",
    "warp_grid",
    "warp_grid3d",
    "normalize_homography",
    "normalize_homography3d",
    "normal_transform_pixel",
    "normal_transform_pixel3d",
]


def warp_grid(grid, src_homo_dst):
    """
        Compute the grid to warp the coordinates grid by the homography/ies.
    """
    batch_size = src_homo_dst.size(0)
    _, height, width, _ = grid.size()

    # expand grid to match the input batch size
    grid = grid.expand(batch_size, -1, -1, -1)  # NxHxWx2
    if len(src_homo_dst.shape) == 3:  # local homography case
        src_homo_dst = src_homo_dst.view(batch_size, 1, 3, 3)  # Nx1x3x3

    # perform the actual grid transformation,
    # the grid is copied to input device and casted to the same type

    flow = transform_points(src_homo_dst, grid.to(src_homo_dst))  # NxHxWx2
    return flow.view(batch_size, height, width, 2)  # NxHxWx2


def warp_grid3d(grid, src_homo_dst):
    """
        Compute the grid to warp the coordinates grid by the homography/ies.
    """
    batch_size = src_homo_dst.size(0)
    _, depth, height, width, _ = grid.size()

    # expand grid to match the input batch size
    grid = grid.expand(batch_size, -1, -1, -1, -1)  # NxDxHxWx3

    if len(src_homo_dst.shape) == 3:  # local homography case
        src_homo_dst = src_homo_dst.view(batch_size, 1, 4, 4)  # Nx1x3x3

    # perform the actual grid transformation,
    # the grid is copied to input device and casted to the same type
    flow = transform_points(src_homo_dst, grid.to(src_homo_dst))  # NxDxHxWx3

    return flow.view(batch_size, depth, height, width, 3)  # NxDxHxWx3


# functional api
def homography_warp(patch_src, src_homo_dst, dsize,
                    mode='bilinear', padding_mode='zeros', align_corners=False, normalized_coordinates=True):
    """
        Warp image patchs or tensors by normalized 2D homographies. See `HomographyWarper` for details.
    """

    if not src_homo_dst.device == patch_src.device:
        raise TypeError("Patch and homography must be on the same device. \
                         Got patch.device: {} src_H_dst.device: {}.".format(
            patch_src.device, src_homo_dst.device))

    height, width = dsize

    grid = create_meshgrid(height, width, normalized_coordinates=normalized_coordinates)

    warped_grid = warp_grid(grid, src_homo_dst)

    return F.grid_sample(patch_src, warped_grid,
                         mode=mode,
                         padding_mode=padding_mode,
                         align_corners=align_corners)


def homography_warp3d(patch_src, src_homo_dst, dsize,
                      mode='bilinear', padding_mode='zeros', align_corners=False, normalized_coordinates=True):
    """
        Warp image patchs or tensors by normalized 3D homographies.
    """
    if not src_homo_dst.device == patch_src.device:
        raise TypeError("Patch and homography must be on the same device. \
                         Got patch.device: {} src_H_dst.device: {}.".format(
            patch_src.device, src_homo_dst.device))

    depth, height, width = dsize

    grid = create_meshgrid3d(depth, height, width, normalized_coordinates=normalized_coordinates, device=patch_src.device)

    warped_grid = warp_grid3d(grid, src_homo_dst)

    return F.grid_sample(patch_src, warped_grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


# layer api
class HomographyWarper(nn.Module):
    """
        Warp tensors by homographies.
    """

    def __init__(self, height, width, mode='bilinear', padding_mode='zeros', normalized_coordinates=True, align_corners=False):
        super(HomographyWarper, self).__init__()
        self.width = width
        self.height = height

        self.mode = mode
        self.padding_mode = padding_mode
        self.normalized_coordinates = normalized_coordinates
        self.align_corners = align_corners

        # create base grid to compute the flow
        self.grid = create_meshgrid(height, width, normalized_coordinates=normalized_coordinates)

        # initialize the warped destination grid
        self._warped_grid = None

    def precompute_warp_grid(self, src_homo_dst: torch.Tensor) -> None:
        """
            Compute and store internaly the transformations of the points.
            Useful when the same homography/homographies are reused.
         """
        self._warped_grid = warp_grid(self.grid, src_homo_dst)

    def forward(self, patch_src, src_homo_dst=None):
        """
            Warp a tensor from source into reference frame.
        """
        _warped_grid = self._warped_grid

        if src_homo_dst is not None:
            warped_patch = homography_warp(patch_src, src_homo_dst, (self.height, self.width),
                                           mode=self.mode,
                                           padding_mode=self.padding_mode,
                                           align_corners=self.align_corners,
                                           normalized_coordinates=self.normalized_coordinates)
        elif _warped_grid is not None:
            if not _warped_grid.device == patch_src.device:
                raise TypeError("Patch and warped grid must be on the same device. \
                                 Got patch.device: {} warped_grid.device: {}. Whether \
                                 recall precompute_warp_grid() with the correct device \
                                 for the homograhy or change the patch device."
                                .format(patch_src.device, _warped_grid.device))

            warped_patch = F.grid_sample(patch_src, _warped_grid, mode=self.mode, padding_mode=self.padding_mode,
                                         align_corners=self.align_corners)
        else:
            raise RuntimeError("Unknown warping. If homographies are not provided \
                                they must be preset using the method: \
                                precompute_warp_grid().")

        return warped_patch


def normal_transform_pixel(height, width, eps=1e-14, device=None, dtype=None):
    """
        Compute the normalization matrix from image size in pixels to [-1, 1].
    """
    tr_mat = torch.tensor([[1.0, 0.0, -1.0],
                           [0.0, 1.0, -1.0],
                           [0.0, 0.0, 1.0]], device=device, dtype=dtype)  # 3x3

    # prevent divide by zero bugs
    width_denom = eps if width == 1 else width - 1.0
    height_denom = eps if height == 1 else height - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom

    return tr_mat.unsqueeze(0)  # 1x3x3


def normal_transform_pixel3d(depth, height, width, eps=1e-14, device=None, dtype=None):
    """
        Compute the normalization matrix from image size in pixels to [-1, 1].
    """
    tr_mat = torch.tensor([[1.0, 0.0, 0.0, -1.0],
                           [0.0, 1.0, 0.0, -1.0],
                           [0.0, 0.0, 1.0, -1.0],
                           [0.0, 0.0, 0.0, 1.0]], device=device, dtype=dtype)  # 4x4

    # prevent divide by zero bugs
    width_denom = eps if width == 1 else width - 1.0
    height_denom = eps if height == 1 else height - 1.0
    depth_denom = eps if depth == 1 else depth - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom
    tr_mat[2, 2] = tr_mat[2, 2] * 2.0 / depth_denom

    return tr_mat.unsqueeze(0)  # 1x4x4


def normalize_homography(dst_pix_trans_src_pix, dsize_src, dsize_dst):
    """
        Normalize a given homography in pixels to [-1, 1].
    """

    if not (len(dst_pix_trans_src_pix.shape) == 3 or dst_pix_trans_src_pix.shape[-2:] == (3, 3)):
        raise ValueError("Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {}"
                         .format(dst_pix_trans_src_pix.shape))

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix = normal_transform_pixel(src_h, src_w).to(dst_pix_trans_src_pix)

    src_pix_trans_src_norm = torch.inverse(src_norm_trans_src_pix)

    dst_norm_trans_dst_pix = normal_transform_pixel(dst_h, dst_w).to(dst_pix_trans_src_pix)

    # compute chain transformations
    dst_norm_trans_src_norm = (dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm))
    return dst_norm_trans_src_norm


def normalize_homography3d(dst_pix_trans_src_pix, dsize_src, dsize_dst):
    """
        Normalize a given homography in pixels to [-1, 1].
    """

    if not (len(dst_pix_trans_src_pix.shape) == 3 or dst_pix_trans_src_pix.shape[-2:] == (4, 4)):
        raise ValueError("Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {}"
                         .format(dst_pix_trans_src_pix.shape))

    # source and destination sizes
    src_d, src_h, src_w = dsize_src
    dst_d, dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix = normal_transform_pixel3d(src_d, src_h, src_w).to(dst_pix_trans_src_pix)

    src_pix_trans_src_norm = torch.inverse(src_norm_trans_src_pix)

    dst_norm_trans_dst_pix = normal_transform_pixel3d(dst_d, dst_h, dst_w).to(dst_pix_trans_src_pix)

    # compute chain transformations
    dst_norm_trans_src_norm = (dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm))

    return dst_norm_trans_src_norm
