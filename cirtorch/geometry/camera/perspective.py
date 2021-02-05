import torch
import torch.nn.functional as F

from ..conversions import convert_points_to_homogeneous, convert_points_from_homogeneous


def project_points(point_3d, camera_matrix):
    """
        Projects a 3d point onto the 2d camera plane.
    """
    if not (point_3d.device == camera_matrix.device):
        raise ValueError("Input tensors must be all in the same device.")

    if not point_3d.shape[-1] == 3:
        raise ValueError("Input points_3d must be in the shape of (*, 3)."
                         " Got {}".format(point_3d.shape))

    if not camera_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            "Input camera_matrix must be in the shape of (*, 3, 3).")

    # projection eq. [u, v, w]' = K * [x y z 1]'
    # u = fx * X / Z + cx
    # v = fy * Y / Z + cy

    # project back using depth dividing in a safe way
    xy_coords = convert_points_from_homogeneous(point_3d)

    x_coord = xy_coords[..., 0]
    y_coord = xy_coords[..., 1]

    # unpack intrinsics
    fx = camera_matrix[..., 0, 0]
    fy = camera_matrix[..., 1, 1]
    cx = camera_matrix[..., 0, 2]
    cy = camera_matrix[..., 1, 2]

    # apply intrinsics ans return
    u_coord = x_coord * fx + cx
    v_coord = y_coord * fy + cy

    return torch.stack([u_coord, v_coord], dim=-1)


def unproject_points(point_2d, depth, camera_matrix, normalize=False):
    """
        Unprojects a 2d point in 3d.
        Transform coordinates in the pixel frame to the camera frame.
    """

    if not (point_2d.device == depth.device == camera_matrix.device):
        raise ValueError("Input tensors must be all in the same device.")

    if not point_2d.shape[-1] == 2:
        raise ValueError("Input points_2d must be in the shape of (*, 2)."
                         " Got {}".format(point_2d.shape))

    if not depth.shape[-1] == 1:
        raise ValueError("Input depth must be in the shape of (*, 1)."
                         " Got {}".format(depth.shape))

    if not camera_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            "Input camera_matrix must be in the shape of (*, 3, 3).")

    # projection eq. K_inv * [u v 1]'
    # x = (u - cx) * Z / fx
    # y = (v - cy) * Z / fy

    # unpack coordinates
    u_coord = point_2d[..., 0]
    v_coord = point_2d[..., 1]

    # unpack intrinsics
    fx = camera_matrix[..., 0, 0]
    fy = camera_matrix[..., 1, 1]
    cx = camera_matrix[..., 0, 2]
    cy = camera_matrix[..., 1, 2]

    # projective
    x_coord = (u_coord - cx) / fx
    y_coord = (v_coord - cy) / fy

    xyz = torch.stack([x_coord, y_coord], dim=-1)

    xyz = convert_points_to_homogeneous(xyz)

    if normalize:
        xyz = F.normalize(xyz, dim=-1, p=2)

    return xyz * depth