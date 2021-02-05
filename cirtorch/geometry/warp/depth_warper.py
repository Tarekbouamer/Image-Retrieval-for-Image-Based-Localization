from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..linalg import transform_points, inverse_transformation, compose_transformations
from ..conversions import convert_points_to_homogeneous, normalize_pixel_coordinates
from ..camera import PinholeCamera
from ..camera.PinholeCamera import cam2pixel, pixel2cam

from cirtorch.utils.grid import create_meshgrid


__all__ = [
    "depth_warp",
    "DepthWarper",
]


class DepthWarper(nn.Module):
    """
        Warps a patch by depth.
    """

    def __init__(self, pinhole_dst, height, width, mode='bilinear', padding_mode='zeros', align_corners=True):
        super(DepthWarper, self).__init__()

        # constructor members
        self.width = width
        self.height = height

        self.mode = mode
        self.padding_mode = padding_mode
        self.eps = 1e-6
        self.align_corners = align_corners

        # state members
        self._pinhole_dst = pinhole_dst
        self._pinhole_src = None
        self._dst_proj_src = None

        self.grid = self._create_meshgrid(height, width)

    @staticmethod
    def _create_meshgrid(height, width):

        grid = create_meshgrid(height, width, normalized_coordinates=False)  # 1xHxWx2

        return convert_points_to_homogeneous(grid)  # append ones to last dim

    def compute_projection_matrix( self, pinhole_src):
        """
            Computes the projection matrix from the source to destination frame.
        """

        if not isinstance(self._pinhole_dst, PinholeCamera):
            raise TypeError("Member self._pinhole_dst expected to be of class " 
                            "PinholeCamera. Got {}" .format(type(self._pinhole_dst)))

        if not isinstance(pinhole_src, PinholeCamera):
            raise TypeError("Argument pinhole_src expected to be of class "
                            "PinholeCamera. Got {}".format(type(pinhole_src)))

        # compute the relative pose between the other and the reference

        # camera frames.
        dst_trans_src = compose_transformations(self._pinhole_dst.extrinsics, inverse_transformation(pinhole_src.extrinsics))

        # the reference.
        dst_proj_src = torch.matmul(self._pinhole_dst.intrinsics, dst_trans_src)

        # update class members
        self._pinhole_src = pinhole_src
        self._dst_proj_src = dst_proj_src

        return self

    def _compute_projection(self, x, y, invd):

        point = torch.tensor([[[x], [y], [1.0], [invd]]], device=self._dst_proj_src.device, dtype=self._dst_proj_src.dtype)

        flow = torch.matmul(self._dst_proj_src, point)

        z = 1. / flow[:, 2]
        x = (flow[:, 0] * z)
        y = (flow[:, 1] * z)

        return torch.cat([x, y], 1)

    def compute_subpixel_step(self):
        """
            This computes the required inverse depth step to achieve sub pixel accurate sampling of
            the depth cost volume, per camera. Szeliski, Richard, and Daniel Scharstein.
            "Symmetric sub-pixel stereo matching." European Conference on Computer
            Vision. Springer Berlin Heidelberg, 2002.
        """

        delta_d = 0.01
        xy_m1 = self._compute_projection(self.width / 2, self.height / 2, 1.0 - delta_d)
        xy_p1 = self._compute_projection(self.width / 2, self.height / 2, 1.0 + delta_d)

        dx = torch.norm((xy_p1 - xy_m1), 2, dim=-1) / 2.0

        dxdd = dx / (delta_d)  # pixel*(1/meter)
        # half pixel sampling, we're interested in the min for all cameras

        return torch.min(0.5 / dxdd)

    def warp_grid(self, depth_src):
        """
            Computes a grid for warping a given the depth from the reference pinhole camera.
            The function `compute_projection_matrix` has to be called beforehand in
            order to have precomputed the relative projection matrices encoding the
            relative pose and the intrinsics between the reference and a non
            reference camera.
        """

        if self._dst_proj_src is None or self._pinhole_src is None:
            raise ValueError("Please, call compute_projection_matrix.")

        if len(depth_src.shape) != 4:
            raise ValueError("Input depth_src has to be in the shape of "
                             "Bx1xHxW. Got {}".format(depth_src.shape))

        # unpack depth attributes
        batch_size, _, height, width = depth_src.shape
        device = depth_src.device
        dtype = depth_src.dtype

        # expand the base coordinate grid according to the input batch size
        pixel_coords = self.grid.to(device=device, dtype=dtype).expand(batch_size, -1, -1, -1)  # BxHxWx3

        # reproject the pixel coordinates to the camera frame
        cam_coords_src = pixel2cam(depth_src,
                                   self._pinhole_src.intrinsics_inverse().to(device=device, dtype=dtype),
                                   pixel_coords)  # BxHxWx3

        # reproject the camera coordinates to the pixel
        pixel_coords_src = cam2pixel(cam_coords_src,
                                     self._dst_proj_src.to(device=device, dtype=dtype))  # (B*N)xHxWx2

        # normalize between -1 and 1 the coordinates
        pixel_coords_src_norm = normalize_pixel_coordinates(pixel_coords_src,
                                                            self.height,
                                                            self.width)
        return pixel_coords_src_norm

    def forward(self, depth_src, patch_dst):
        """
            Warps a tensor from destination frame to reference given the depth in the reference frame.
        """
        return F.grid_sample(patch_dst,
                             self.warp_grid(depth_src),  # type: ignore
                             mode=self.mode,
                             padding_mode=self.padding_mode,
                             align_corners=self.align_corners)


# functional api

def depth_warp(pinhole_dst, pinhole_src, depth_src, patch_dst, height, width, align_corners=True):
    """
        Function that warps a tensor from destination frame to reference given the depth in the reference frame.
    """
    warper = DepthWarper(pinhole_dst, height, width, align_corners=align_corners)

    warper.compute_projection_matrix(pinhole_src)

    return warper(depth_src, patch_dst)
