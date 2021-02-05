import torch
from ..conversions import convert_affinematrix_to_homography3d, deg2rad, angle_axis_to_rotation_matrix
from ..linalg import eye_like
from ..warp.homography_warper import normalize_homography3d, homography_warp3d


__all__ = [
    "warp_affine3d",
    "get_projective_transform",
    "projection_from_Rt",
    "get_perspective_transform3d",
    "warp_perspective3d"
]


def warp_affine3d(src, M, dsize, mode='bilinear', padding_mode='zeros', align_corners=True):
    """
        Applies a projective transformation a to 3d tensor.
    """
    assert len(src.shape) == 5, src.shape
    assert len(M.shape) == 3 and M.shape[-2:] == (3, 4), M.shape
    assert len(dsize) == 3, dsize

    B, C, D, H, W = src.size()

    size_src = (D, H, W)
    size_out = dsize

    M_4x4 = convert_affinematrix_to_homography3d(M)  # Bx4x4

    # we need to normalize the transformation since grid sample needs -1/1 coordinates
    dst_norm_trans_src_norm = normalize_homography3d(M_4x4, size_src, size_out)    # Bx4x4

    src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm)
    P_norm = src_norm_trans_dst_norm[:, :3]  # Bx3x4

    # compute meshgrid and apply to input
    dsize_out = [B, C] + list(size_out)
    grid = torch.nn.functional.affine_grid(P_norm, dsize_out, align_corners=align_corners)

    return torch.nn.functional.grid_sample(src, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode)


def projection_from_Rt(rmat, tvec):
    """
        Compute the projection matrix from Rotation and translation.
    """
    assert len(rmat.shape) >= 2 and rmat.shape[-2:] == (3, 3), rmat.shape
    assert len(tvec.shape) >= 2 and tvec.shape[-2:] == (3, 1), tvec.shape

    return torch.cat([rmat, tvec], dim=-1)  # Bx3x4


def get_projective_transform(center, angles, scales):
    """
        Calculates the projection matrix for a 3D rotation.
    """
    assert len(center.shape) == 2 and center.shape[-1] == 3, center.shape
    assert len(angles.shape) == 2 and angles.shape[-1] == 3, angles.shape
    assert center.device == angles.device, (center.device, angles.device)
    assert center.dtype == angles.dtype, (center.dtype, angles.dtype)

    # create rotation matrix
    angle_axis_rad = deg2rad(angles)
    rmat = angle_axis_to_rotation_matrix(angle_axis_rad)  # Bx3x3
    scaling_matrix = eye_like(3, rmat)
    scaling_matrix = scaling_matrix * scales.unsqueeze(dim=1)
    rmat = rmat @ scaling_matrix.to(rmat)

    # define matrix to move forth and back to origin
    from_origin_mat = torch.eye(4)[None].repeat(rmat.shape[0], 1, 1).type_as(center)  # Bx4x4
    from_origin_mat[..., :3, -1] += center

    # TODO: check this too
    to_origin_mat = from_origin_mat.clone()
    to_origin_mat = from_origin_mat.inverse()

    # append tranlation with zeros
    proj_mat = projection_from_Rt(rmat, torch.zeros_like(center)[..., None])  # Bx3x4

    # chain 4x4 transforms
    proj_mat = convert_affinematrix_to_homography3d(proj_mat)  # Bx4x4
    proj_mat = (from_origin_mat @ proj_mat @ to_origin_mat)

    return proj_mat[..., :3, :]  # Bx3x4


def get_perspective_transform3d(src, dst):
    """
        Calculate a 3d perspective transform from four pairs of the corresponding points.
    """

    if not src.shape[-2:] == (8, 3):
        raise ValueError("Inputs must be a Bx8x3 tensor. Got {}"
                         .format(src.shape))
    if not src.shape == dst.shape:
        raise ValueError("Inputs must have the same shape. Got {}"
                         .format(dst.shape))
    if not (src.shape[0] == dst.shape[0]):
        raise ValueError("Inputs must have same batch size dimension. Expect {} but got {}"
                         .format(src.shape, dst.shape))
    assert src.device == dst.device and src.dtype == dst.dtype, (
        f"Expect `src` and `dst` to be in the same device (Got {src.dtype}, {dst.dtype}) "
        f"with the same dtype (Got {src.dtype}, {dst.dtype})." )

    # we build matrix A by using only 4 point correspondence. The linear
    # system is solved with the least square method, so here
    # we could even pass more correspondence
    p = []

    # 000, 100, 110, 101, 011
    for i in [0, 1, 2, 5, 7]:
        p.append(_build_perspective_param3d(src[:, i], dst[:, i], 'x'))
        p.append(_build_perspective_param3d(src[:, i], dst[:, i], 'y'))
        p.append(_build_perspective_param3d(src[:, i], dst[:, i], 'z'))

    # A is Bx15x15
    A = torch.stack(p, dim=1)

    # b is a Bx15x1
    b = torch.stack([
        dst[:, 0:1, 0], dst[:, 0:1, 1], dst[:, 0:1, 2],
        dst[:, 1:2, 0], dst[:, 1:2, 1], dst[:, 1:2, 2],
        dst[:, 2:3, 0], dst[:, 2:3, 1], dst[:, 2:3, 2],
        # dst[:, 3:4, 0], dst[:, 3:4, 1], dst[:, 3:4, 2],
        # dst[:, 4:5, 0], dst[:, 4:5, 1], dst[:, 4:5, 2],
        dst[:, 5:6, 0], dst[:, 5:6, 1], dst[:, 5:6, 2],
        # dst[:, 6:7, 0], dst[:, 6:7, 1], dst[:, 6:7, 2],
        dst[:, 7:8, 0], dst[:, 7:8, 1], dst[:, 7:8, 2],
    ], dim=1)

    # solve the system Ax = b
    X, LU = torch.solve(b, A)

    # create variable to return
    batch_size = src.shape[0]
    M = torch.ones(batch_size, 16, device=src.device, dtype=src.dtype)
    M[..., :15] = torch.squeeze(X, dim=-1)

    return M.view(-1, 4, 4)  # Bx4x4


def _build_perspective_param3d(p, q, axis):
    ones = torch.ones_like(p)[..., 0:1]
    zeros = torch.zeros_like(p)[..., 0:1]

    if axis == 'x':
        return torch.cat([
            p[:, 0:1], p[:, 1:2], p[:, 2:3], ones,
            zeros, zeros, zeros, zeros,
            zeros, zeros, zeros, zeros,
            -p[:, 0:1] * q[:, 0:1], -p[:, 1:2] * q[:, 0:1], -p[:, 2:3] * q[:, 0:1]
        ], dim=1)

    if axis == 'y':
        return torch.cat([
            zeros, zeros, zeros, zeros,
            p[:, 0:1], p[:, 1:2], p[:, 2:3], ones,
            zeros, zeros, zeros, zeros,
            -p[:, 0:1] * q[:, 1:2], -p[:, 1:2] * q[:, 1:2], -p[:, 2:3] * q[:, 1:2]
        ], dim=1)

    if axis == 'z':
        return torch.cat([
            zeros, zeros, zeros, zeros,
            zeros, zeros, zeros, zeros,
            p[:, 0:1], p[:, 1:2], p[:, 2:3], ones,
            -p[:, 0:1] * q[:, 2:3], -p[:, 1:2] * q[:, 2:3], -p[:, 2:3] * q[:, 2:3]
        ], dim=1)

    raise NotImplementedError(f"perspective params for axis `{axis}` is not implemented.")


def warp_perspective3d(src, M, dsize, mode='bilinear', border_mode='zeros', align_corners=False):
    """
        Applies a perspective transformation to an image.
        The function warp_perspective transforms the source image using the specified matrix:
    """

    if not len(src.shape) == 5:
        raise ValueError("Input src must be a BxCxDxHxW tensor. Got {}"
                         .format(src.shape))

    if not (len(M.shape) == 3 or M.shape[-2:] == (4, 4)):
        raise ValueError("Input M must be a Bx4x4 tensor. Got {}"
                         .format(M.shape))

    # launches the warper
    d, h, w = src.shape[-3:]
    return transform_warp_impl3d(src, M, (d, h, w), dsize, mode, border_mode, align_corners)


def transform_warp_impl3d(src, dst_pix_trans_src_pix, dsize_src, dsize_dst, grid_mode, padding_mode, align_corners):
    """Compute the transform in normalized cooridnates and perform the warping.
    """
    dst_norm_trans_src_norm = normalize_homography3d(dst_pix_trans_src_pix, dsize_src, dsize_dst)

    src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm)

    return homography_warp3d(src, src_norm_trans_dst_norm, dsize_dst, grid_mode, padding_mode, align_corners, True)