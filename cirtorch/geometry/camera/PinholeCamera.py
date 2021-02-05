from typing import Iterable, Optional, Union
import warnings

import torch
import torch.nn as nn

from ..linalg import transform_points, inverse_transformation


class PinholeCamera:
    """
        Class that represents a Pinhole Camera model.
    """

    def __init__(self, intrinsics, extrinsics, height, width):

        # verify batch size and shapes
        self._check_valid([intrinsics, extrinsics, height, width])
        self._check_valid_params(intrinsics, "intrinsics")
        self._check_valid_params(extrinsics, "extrinsics")
        self._check_valid_shape(height, "height")
        self._check_valid_shape(width, "width")

        # set class attributes
        self.height = height
        self.width = width
        self._intrinsics = intrinsics
        self._extrinsics = extrinsics

    @staticmethod
    def _check_valid(data_iter):

        if not all(data.shape[0] for data in data_iter):
            raise ValueError("Arguments shapes must match")

        return True

    @staticmethod
    def _check_valid_params(data, data_name):
        if len(data.shape) not in (3, 4,) and data.shape[-2:] != (4, 4):  # Shouldn't this be an OR logic than AND?
            raise ValueError("Argument {0} shape must be in the following shape"
                             " Bx4x4 or BxNx4x4. Got {1}".format(data_name, data.shape))
        return True

    @staticmethod
    def _check_valid_shape(data, data_name):
        if not len(data.shape) == 1:
            raise ValueError("Argument {0} shape must be in the following shape"
                             " B. Got {1}".format(data_name, data.shape))
        return True

    @property
    def intrinsics(self):
        """
            The full 4x4 intrinsics matrix.
        """
        assert self._check_valid_params(self._intrinsics, "intrinsics")
        return self._intrinsics

    @property
    def extrinsics(self):
        """
            The full 4x4 extrinsics matrix.
        """
        assert self._check_valid_params(self._extrinsics, "extrinsics")
        return self._extrinsics

    @property
    def batch_size(self):
        """
            Returns the batch size of the storage.
        """
        return self.intrinsics.shape[0]

    @property
    def fx(self):
        """
            Returns the focal lenght in the x-direction.
        """
        return self.intrinsics[..., 0, 0]

    @property
    def fy(self):
        """
            Returns the focal length in the y-direction.
        """
        return self.intrinsics[..., 1, 1]

    @property
    def cx(self):
        """
            Returns the x-coordinate of the principal point.
        """
        return self.intrinsics[..., 0, 2]

    @property
    def cy(self):
        """
            Returns the y-coordinate of the principal point.
        """
        return self.intrinsics[..., 1, 2]

    @property
    def tx(self):
        """
            Returns the x-coordinate of the translation vector.
        """
        return self.extrinsics[..., 0, -1]

    @tx.setter
    def tx(self, value):
        """
            Set the x-coordinate of the translation vector with the given value.
        """
        self.extrinsics[..., 0, -1] = value
        return self

    @property
    def ty(self):
        """
            Returns the y-coordinate of the translation vector.
        """
        return self.extrinsics[..., 1, -1]

    @ty.setter
    def ty(self, value):
        """
            Set the y-coordinate of the translation vector with the given value.
        """
        self.extrinsics[..., 1, -1] = value
        return self

    @property
    def tz(self):
        """
            Returns the z-coordinate of the translation vector.
        """
        return self.extrinsics[..., 2, -1]

    @tz.setter
    def tz(self, value):
        """
            Set the y-coordinate of the translation vector with the given value.
        """
        self.extrinsics[..., 2, -1] = value
        return self

    @property
    def rt_matrix(self):
        """
            Returns the 3x4 rotation-translation matrix.
        """
        return self.extrinsics[..., :3, :4]

    @property
    def camera_matrix(self):
        """
            Returns the 3x3 camera matrix containing the intrinsics.
        """
        return self.intrinsics[..., :3, :3]

    @property
    def rotation_matrix(self):
        """
            Returns the 3x3 rotation matrix from the extrinsics.
        """
        return self.extrinsics[..., :3, :3]

    @property
    def translation_vector(self):
        """
            Returns the translation vector from the extrinsics.
        """
        return self.extrinsics[..., :3, -1:]

    def clone(self):
        """
            Returns a deep copy of the current object instance.
        """
        height = self.height.clone()
        width = self.width.clone()
        intrinsics = self.intrinsics.clone()
        extrinsics= self.extrinsics.clone()

        return PinholeCamera(intrinsics, extrinsics, height, width)

    def intrinsics_inverse(self):
        """
            Returns the inverse of the 4x4 instrisics matrix.
        """
        return self.intrinsics.inverse()

    def scale(self, scale_factor):
        """
            Scales the pinhole model.
        """

        # scale the intrinsic parameters
        intrinsics = self.intrinsics.clone()

        intrinsics[..., 0, 0] *= scale_factor
        intrinsics[..., 1, 1] *= scale_factor
        intrinsics[..., 0, 2] *= scale_factor
        intrinsics[..., 1, 2] *= scale_factor

        # scale the image height/width
        height = scale_factor * self.height.clone()
        width = scale_factor * self.width.clone()

        return PinholeCamera(intrinsics, self.extrinsics, height, width)

    def scale_(self, scale_factor):
        """
            Scales the pinhole model in-place.
        """
        # scale the intrinsic parameters
        self.intrinsics[..., 0, 0] *= scale_factor
        self.intrinsics[..., 1, 1] *= scale_factor
        self.intrinsics[..., 0, 2] *= scale_factor
        self.intrinsics[..., 1, 2] *= scale_factor

        # scale the image height/width
        self.height *= scale_factor
        self.width *= scale_factor

        return self

    # NOTE: just for test. Decide if we keep it.
    @classmethod
    def from_parameters(self, fx, fy, cx, cy, height, width, tx, ty, tz, batch_size=1, device=None, dtype=None):

        # create the camera matrix
        intrinsics = torch.zeros(batch_size, 4, 4, device=device, dtype=dtype)

        intrinsics[..., 0, 0] += fx
        intrinsics[..., 1, 1] += fy
        intrinsics[..., 0, 2] += cx
        intrinsics[..., 1, 2] += cy
        intrinsics[..., 2, 2] += 1.0
        intrinsics[..., 3, 3] += 1.0

        # create the pose matrix
        extrinsics = torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)
        extrinsics[..., 0, -1] += tx
        extrinsics[..., 1, -1] += ty
        extrinsics[..., 2, -1] += tz

        # create image hegith and width
        height_tmp = torch.zeros(batch_size, device=device, dtype=dtype)
        height_tmp[..., 0] += height

        width_tmp = torch.zeros(batch_size, device=device, dtype=dtype)
        width_tmp[..., 0] += width

        return self(intrinsics, extrinsics, height_tmp, width_tmp)


class PinholeCamerasList(PinholeCamera):
    """
        Class that represents a list of pinhole cameras. The class inherits from `PinholeCamera` meaning that
        it will keep the same class properties but with an extra dimension.
    """

    def __init__(self, pinholes_list):

        self._initialize_parameters(pinholes_list)

    def _initialize_parameters(self, pinholes):
        """
            Initialises the class attributes given a cameras list.
        """

        if not isinstance(pinholes, (list, tuple,)):
            raise TypeError("pinhole must of type list or tuple. Got {}".format(type(pinholes)))

        height, width = [], []
        intrinsics, extrinsics = [], []

        for pinhole in pinholes:
            if not isinstance(pinhole, PinholeCamera):
                raise TypeError("Argument pinhole must be from type " "PinholeCamera. Got {}".format(type(pinhole)))

            height.append(pinhole.height)
            width.append(pinhole.width)
            intrinsics.append(pinhole.intrinsics)
            extrinsics.append(pinhole.extrinsics)

        # contatenate and set members. We will assume BxNx4x4
        self.height = torch.stack(height, dim=1)
        self.width = torch.stack(width, dim=1)
        self._intrinsics = torch.stack(intrinsics, dim=1)
        self._extrinsics = torch.stack(extrinsics, dim=1)

        return self

    @property
    def num_cameras(self):
        """
            Returns the number of pinholes cameras per batch.
        """

        num_cameras = -1

        if self.intrinsics is not None:
            num_cameras = int(self.intrinsics.shape[1])
        return num_cameras

    def get_pinhole(self, idx):
        """
            Returns a PinholeCamera object with parameters such as Bx4x4.
        """

        height = self.height[..., idx]
        width = self.width[..., idx]
        intrinsics = self.intrinsics[:, idx]
        extrinsics = self.extrinsics[:, idx]

        return PinholeCamera(intrinsics, extrinsics, height, width)


# TODO: not sure about keeping this, move it to geometry section


def pinhole_matrix(pinholes, eps=1e-6):
    """
        Function that returns the pinhole matrix from a pinhole model
    """

    assert len(pinholes.shape) == 2 and pinholes.shape[1] == 12, pinholes.shape

    # unpack pinhole values
    fx, fy, cx, cy = torch.chunk(pinholes[..., :4], 4, dim=1)  # Nx1

    # create output container
    k = torch.eye(4, device=pinholes.device, dtype=pinholes.dtype) + eps
    k = k.view(1, 4, 4).repeat(pinholes.shape[0], 1, 1)  # Nx4x4

    # fill output with pinhole values
    k[..., 0, 0:1] = fx
    k[..., 0, 2:3] = cx
    k[..., 1, 1:2] = fy
    k[..., 1, 2:3] = cy
    return k


def inverse_pinhole_matrix(pinhole, eps=1e-6):
    """
        Returns the inverted pinhole matrix from a pinhole model
    """

    assert len(pinhole.shape) == 2 and pinhole.shape[1] == 12, pinhole.shape

    # unpack pinhole values
    fx, fy, cx, cy = torch.chunk(pinhole[..., :4], 4, dim=1)  # Nx1

    # create output container
    k = torch.eye(4, device=pinhole.device, dtype=pinhole.dtype)
    k = k.view(1, 4, 4).repeat(pinhole.shape[0], 1, 1)  # Nx4x4

    # fill output with inverse values
    k[..., 0, 0:1] = 1. / (fx + eps)
    k[..., 1, 1:2] = 1. / (fy + eps)
    k[..., 0, 2:3] = -1. * cx / (fx + eps)
    k[..., 1, 2:3] = -1. * cy / (fy + eps)

    return k


def scale_pinhole(pinholes, scale):
    """
        Scales the pinhole matrix for each pinhole model.
    """

    assert len(pinholes.shape) == 2 and pinholes.shape[1] == 12, pinholes.shape
    assert len(scale.shape) == 1, scale.shape

    pinholes_scaled = pinholes.clone()
    pinholes_scaled[..., :6] = pinholes[..., :6] * scale.unsqueeze(-1)

    return pinholes_scaled


# based on:
# https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py#L26


def pixel2cam(depth, intrinsics_inv, pixel_coords):
    """
        Transform coordinates in the pixel frame to the camera frame.
    """
    if not len(depth.shape) == 4 and depth.shape[1] == 1:
        raise ValueError("Input depth has to be in the shape of "
                         "Bx1xHxW. Got {}".format(depth.shape))
    if not len(intrinsics_inv.shape) == 3:
        raise ValueError("Input intrinsics_inv has to be in the shape of "
                         "Bx4x4. Got {}".format(intrinsics_inv.shape))
    if not len(pixel_coords.shape) == 4 and pixel_coords.shape[3] == 3:
        raise ValueError("Input pixel_coords has to be in the shape of "
                         "BxHxWx3. Got {}".format(intrinsics_inv.shape))

    cam_coords = transform_points(intrinsics_inv[:, None], pixel_coords)

    return cam_coords * depth.permute(0, 2, 3, 1)


# based on
# https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py#L43

def cam2pixel(cam_coords_src, dst_proj_src, eps=1e-6):
    """
        Transform coordinates in the camera frame to the pixel frame.
    """
    if not len(cam_coords_src.shape) == 4 and cam_coords_src.shape[3] == 3:
        raise ValueError("Input cam_coords_src has to be in the shape of "
                         "BxHxWx3. Got {}".format(cam_coords_src.shape))
    if not len(dst_proj_src.shape) == 3 and dst_proj_src.shape[-2:] == (4, 4):
        raise ValueError("Input dst_proj_src has to be in the shape of "
                         "Bx4x4. Got {}".format(dst_proj_src.shape))

    b, h, w, _ = cam_coords_src.shape

    # apply projection matrix to points
    point_coords = transform_points(dst_proj_src[:, None], cam_coords_src)

    x_coord = point_coords[..., 0]
    y_coord = point_coords[..., 1]
    z_coord = point_coords[..., 2]

    # compute pixel coordinates
    u_coord = x_coord / (z_coord + eps)
    v_coord = y_coord / (z_coord + eps)

    # stack and return the coordinates, that's the actual flow
    pixel_coords_dst = torch.stack([u_coord, v_coord], dim=-1)

    return pixel_coords_dst  # (B*N)xHxWx2
