from typing import Tuple, Optional

import torch
import torch.nn.functional as F

from ..warp.homography_warper import normalize_homography, homography_warp

from ..conversions import \
    (deg2rad,
     normalize_pixel_coordinates,
     convert_affinematrix_to_homography,
     convert_affinematrix_to_homography3d)

from .projwarp import get_projective_transform

__all__ = [
    "warp_perspective",
    "warp_affine",
    "get_perspective_transform",
    "get_rotation_matrix2d",
    "remap",
    "invert_affine_transform",
    "angle_to_rotation_matrix",
    "get_affine_matrix2d",
    "get_affine_matrix3d",
    "get_shear_matrix2d",
    "get_shear_matrix3d"
]


def transform_warp_impl(src, dst_pix_trans_src_pix, dsize_src, dsize_dst, grid_mode, padding_mode, align_corners):
    """
        Compute the transform in normalized coordinates and perform the warping.
    """
    dst_norm_trans_src_norm = normalize_homography(dst_pix_trans_src_pix, dsize_src, dsize_dst)

    src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm)

    return homography_warp(src, src_norm_trans_dst_norm, dsize_dst, grid_mode, padding_mode, align_corners, True)


def warp_perspective(src, M, dsize, mode='bilinear', border_mode='zeros', align_corners=False):
    """
        Applies a perspective transformation to an image.
        The function warp_perspective transforms the source image using the specified matrix:
    """

    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}"
                         .format(src.shape))

    if not (len(M.shape) == 3 and M.shape[-2:] == (3, 3)):
        raise ValueError("Input M must be a Bx3x3 tensor. Got {}"
                         .format(M.shape))

    # launches the warper
    h, w = src.shape[-2:]
    return transform_warp_impl(src, M, (h, w), dsize, mode, border_mode, align_corners)


def warp_affine(src, M, dest_size, mode='bilinear', padding_mode='zeros', align_corners=False):
    """
        Applies an affine transformation to a tensor.
        The function warp_affine transforms the source tensor using the specified matrix:
    """

    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}"
                         .format(src.shape))

    if not (len(M.shape) == 3 or M.shape[-2:] == (2, 3)):
        raise ValueError("Input M must be a Bx2x3 tensor. Got {}"
                         .format(M.shape))

    B, C, H, W = src.size()
    src_size = (H, W)


    dst_norm_trans_src_norm = normalize_homography(M, src_size, dest_size)

    src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm)

    grid = F.affine_grid(src_norm_trans_dst_norm[:, :2, :],
                         [B, C, dest_size[0], dest_size[1]],
                         align_corners=align_corners)

    return F.grid_sample(src, grid,
                         align_corners=align_corners,
                         mode=mode,
                         padding_mode=padding_mode)


def get_perspective_transform(src, dst):
    """
        Calculates a perspective transform from four pairs of the corresponding points.
        The function calculates the matrix of a perspective transform so that:
    """
    if not src.shape[-2:] == (4, 2):
        raise ValueError("Inputs must be a Bx4x2 tensor. Got {}"
                         .format(src.shape))
    if not src.shape == dst.shape:
        raise ValueError("Inputs must have the same shape. Got {}"
                         .format(dst.shape))
    if not (src.shape[0] == dst.shape[0]):
        raise ValueError("Inputs must have same batch size dimension. Expect {} but got {}"
                         .format(src.shape, dst.shape))

    # we build matrix A by using only 4 point correspondence. The linear
    # system is solved with the least square method, so here
    # we could even pass more correspondence

    p = []
    for i in [0, 1, 2, 3]:
        p.append(_build_perspective_param(src[:, i], dst[:, i], 'x'))
        p.append(_build_perspective_param(src[:, i], dst[:, i], 'y'))

    # A is Bx8x8
    A = torch.stack(p, dim=1)

    # b is a Bx8x1
    b = torch.stack([
        dst[:, 0:1, 0], dst[:, 0:1, 1],
        dst[:, 1:2, 0], dst[:, 1:2, 1],
        dst[:, 2:3, 0], dst[:, 2:3, 1],
        dst[:, 3:4, 0], dst[:, 3:4, 1],
    ], dim=1)

    # solve the system Ax = b
    X, LU = torch.solve(b, A)

    # create variable to return
    batch_size = src.shape[0]
    M = torch.ones(batch_size, 9, device=src.device, dtype=src.dtype)
    M[..., :8] = torch.squeeze(X, dim=-1)

    return M.view(-1, 3, 3)  # Bx3x3


def _build_perspective_param(p, q, axis):
    ones = torch.ones_like(p)[..., 0:1]
    zeros = torch.zeros_like(p)[..., 0:1]

    if axis == 'x':
        return torch.cat(
            [p[:, 0:1], p[:, 1:2], ones, zeros, zeros, zeros,
             -p[:, 0:1] * q[:, 0:1], -p[:, 1:2] * q[:, 0:1]
             ], dim=1)

    if axis == 'y':
        return torch.cat(
            [zeros, zeros, zeros, p[:, 0:1], p[:, 1:2], ones,
             -p[:, 0:1] * q[:, 1:2], -p[:, 1:2] * q[:, 1:2]], dim=1)

    raise NotImplementedError(f"perspective params for axis `{axis}` is not implemented.")


def angle_to_rotation_matrix(angle):
    """
        Create a rotation matrix out of angles in degrees.

    """
    ang_rad = deg2rad(angle)
    
    cos_a = torch.cos(ang_rad)
    sin_a = torch.sin(ang_rad)

    return torch.stack([cos_a, sin_a, -sin_a, cos_a], dim=-1).view(*angle.shape, 2, 2)


def get_rotation_matrix2d(center, angle, scale):
    """
        Calculates an affine matrix of 2D rotation.The function calculates the following matrix:
    """

    if not (len(center.shape) == 2 and center.shape[1] == 2):
        raise ValueError("Input center must be a Bx2 tensor. Got {}"
                         .format(center.shape))
    if not len(angle.shape) == 1:
        raise ValueError("Input angle must be a B tensor. Got {}"
                         .format(angle.shape))
    if not (len(scale.shape) == 2 and scale.shape[1] == 2):
        raise ValueError("Input scale must be a Bx2 tensor. Got {}"
                         .format(scale.shape))
    if not (center.shape[0] == angle.shape[0] == scale.shape[0]):
        raise ValueError("Inputs must have same batch size dimension. Got center {}, angle {} and scale {}"
                         .format(center.shape, angle.shape, scale.shape))
    if not (center.device == angle.device == scale.device) or not (center.dtype == angle.dtype == scale.dtype):
        raise ValueError("Inputs must have same device Got center ({}, {}), angle ({}, {}) and scale ({}, {})"
                         .format(center.device, center.dtype, angle.device, angle.dtype, scale.device, scale.dtype))

    # convert angle and apply scale
    rotation_matrix = angle_to_rotation_matrix(angle)

    scaling_matrix = torch.zeros((2, 2), device=rotation_matrix.device, dtype=rotation_matrix.dtype)\
        .fill_diagonal_(1).repeat(rotation_matrix.size(0), 1, 1)

    scaling_matrix = scaling_matrix * scale.unsqueeze(dim=2).repeat(1, 1, 2)
    scaled_rotation = rotation_matrix @ scaling_matrix

    alpha = scaled_rotation[:, 0, 0]
    beta = scaled_rotation[:, 0, 1]

    # unpack the center to x, y coordinates
    x = center[..., 0]
    y = center[..., 1]

    # create output tensor
    batch_size = center.shape[0]

    one = torch.tensor(1., device=center.device, dtype=center.dtype)

    M = torch.zeros(batch_size, 2, 3, device=center.device, dtype=center.dtype)
    
    M[..., 0:2, 0:2] = scaled_rotation
    M[..., 0, 2] = (one - alpha) * x - beta * y
    M[..., 1, 2] = beta * x + (one - alpha) * y

    return M


def remap(tensor, map_x, map_y, align_corners=False):
    """
        Applies a generic geometrical transformation to a tensor.
    """

    if not tensor.shape[-2:] == map_x.shape[-2:] == map_y.shape[-2:]:
        raise ValueError("Inputs last two dimensions must match.")

    batch_size, _, height, width = tensor.shape

    # grid_sample need the grid between -1/1
    map_xy = torch.stack([map_x, map_y], dim=-1)
    map_xy_norm = normalize_pixel_coordinates(map_xy, height, width)

    # simulate broadcasting since grid_sample does not support it
    map_xy_norm = map_xy_norm.expand(batch_size, -1, -1, -1)

    # warp ans return
    tensor_warped = F.grid_sample(tensor, map_xy_norm, align_corners=align_corners)

    return tensor_warped


def invert_affine_transform(matrix):
    """
        Inverts an affine transformation.
        The function computes an inverse affine transformation represented by 2×3 matrix:
    """
    if not (len(matrix.shape) == 3 and matrix.shape[-2:] == (2, 3)):
        raise ValueError("Input matrix must be a Bx2x3 tensor. Got {}"
                         .format(matrix.shape))

    matrix_tmp: torch.Tensor = convert_affinematrix_to_homography(matrix)
    matrix_inv: torch.Tensor = torch.inverse(matrix_tmp)

    return matrix_inv[..., :2, :3]


def get_affine_matrix2d(translations, center, scale, angle, sx=None, sy=None):
    """
        Composes affine matrix from the components.
    """
    transform = get_rotation_matrix2d(center, -angle, scale)
    transform[..., 2] += translations  # tx/ty

    # pad transform to get Bx3x3
    transform_h = convert_affinematrix_to_homography(transform)

    if any([s is not None for s in [sx, sy]]):
        shear_mat = get_shear_matrix2d(center, sx, sy)
        transform_h = transform_h @ shear_mat

    return transform_h


def get_shear_matrix2d(center, sx=None, sy=None):
    """
        Composes shear matrix Bx4x4 from the components.
    """

    sx = torch.tensor([0.]).repeat(center.size(0)) if sx is None else sx
    sy = torch.tensor([0.]).repeat(center.size(0)) if sy is None else sy

    x, y = torch.split(center, 1, dim=-1)
    x, y = x.view(-1), y.view(-1)

    sx_tan = torch.tan(sx)
    sy_tan = torch.tan(sy)

    ones = torch.ones_like(sx)

    shear_mat = torch.stack([
        ones,               -sx_tan,                        sx_tan * y,
        -sy_tan,    ones + sx_tan * sy_tan,         sy_tan * (sx_tan * y + x)
    ], dim=-1).view(-1, 2, 3)

    shear_mat = convert_affinematrix_to_homography(shear_mat)

    return shear_mat


def get_affine_matrix3d(translations, center, scale, angles, sxy=None, sxz=None, syx=None, syz=None, szx=None, szy=None):
    """
        Composes 3d affine matrix from the components.
    """

    transform = get_projective_transform(center, -angles, scale)
    transform[..., 3] += translations  # tx/ty/tz

    # pad transform to get Bx3x3
    transform_h = convert_affinematrix_to_homography3d(transform)

    if any([s is not None for s in [sxy, sxz, syx, syz, szx, szy]]):
        shear_mat = get_shear_matrix3d(center, sxy, sxz, syx, syz, szx, szy)
        transform_h = transform_h @ shear_mat

    return transform_h


def get_shear_matrix3d(center, sxy=None, sxz=None, syx=None, syz=None, szx=None, szy=None):
    """
        Composes shear matrix Bx4x4 from the components.
    """
    sxy = torch.tensor([0.]).repeat(center.size(0)) if sxy is None else sxy
    sxz = torch.tensor([0.]).repeat(center.size(0)) if sxz is None else sxz
    syx = torch.tensor([0.]).repeat(center.size(0)) if syx is None else syx
    syz = torch.tensor([0.]).repeat(center.size(0)) if syz is None else syz
    szx = torch.tensor([0.]).repeat(center.size(0)) if szx is None else szx
    szy = torch.tensor([0.]).repeat(center.size(0)) if szy is None else szy

    x, y, z = torch.split(center, 1, dim=-1)
    x, y, z = x.view(-1), y.view(-1), z.view(-1)

    # Prepare parameters
    sxy_tan = torch.tan(sxy)  # type: ignore
    sxz_tan = torch.tan(sxz)  # type: ignore
    syx_tan = torch.tan(syx)  # type: ignore
    syz_tan = torch.tan(syz)  # type: ignore
    szx_tan = torch.tan(szx)  # type: ignore
    szy_tan = torch.tan(szy)  # type: ignore

    # compute translation matrix
    m00, m10, m20, m01, m11, m21, m02, m12, m22 = _compute_shear_matrix_3d(
        sxy_tan, sxz_tan, syx_tan, syz_tan, szx_tan, szy_tan)

    m03 = m01 * y + m02 * z
    m13 = m10 * x + m11 * y + m12 * z - y
    m23 = m20 * x + m21 * y + m22 * z - z

    # shear matrix is implemented with negative values
    sxy_tan, sxz_tan, syx_tan, syz_tan, szx_tan, szy_tan = \
        - sxy_tan, - sxz_tan, - syx_tan, - syz_tan, - szx_tan, - szy_tan

    m00, m10, m20, m01, m11, m21, m02, m12, m22 = _compute_shear_matrix_3d(
        sxy_tan, sxz_tan, syx_tan, syz_tan, szx_tan, szy_tan)

    shear_mat = torch.stack([
        m00, m01, m02, m03,
        m10, m11, m12, m13,
        m20, m21, m22, m23
    ], dim=-1).view(-1, 3, 4)

    shear_mat = convert_affinematrix_to_homography3d(shear_mat)

    return shear_mat


def _compute_shear_matrix_3d(sxy_tan, sxz_tan, syx_tan, syz_tan, szx_tan, szy_tan):
    zeros = torch.zeros_like(sxy_tan)
    ones = torch.ones_like(sxy_tan)

    m00, m10, m20 = ones, sxy_tan, sxz_tan
    m01, m11, m21 = syx_tan, sxy_tan * syx_tan + ones, sxz_tan * syx_tan + syz_tan
    m02 = syx_tan * szy_tan + szx_tan
    m12 = sxy_tan * szx_tan + szy_tan * m11
    m22 = sxz_tan * szx_tan + szy_tan * m21 + ones

    return m00, m10, m20, m01, m11, m21, m02, m12, m22