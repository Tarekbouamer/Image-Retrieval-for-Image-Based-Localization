import torch
import torch.nn.functional as F


__all__ = [
    "rad2deg",
    "deg2rad",
    "pol2cart",
    "cart2pol",
    "convert_points_from_homogeneous",
    "convert_points_to_homogeneous",
    "convert_affinematrix_to_homography",
    "convert_affinematrix_to_homography3d",
    "angle_axis_to_rotation_matrix",
    "angle_axis_to_quaternion",
    "rotation_matrix_to_angle_axis",
    "rotation_matrix_to_quaternion",
    "quaternion_to_angle_axis",
    "quaternion_to_rotation_matrix",
    "quaternion_log_to_exp",
    "quaternion_exp_to_log",
    "denormalize_pixel_coordinates",
    "normalize_pixel_coordinates",
    "normalize_quaternion",
    "denormalize_pixel_coordinates3d",
    "normalize_pixel_coordinates3d",
]

pi = torch.tensor(3.14159265358979323846)


def rad2deg(tensor):
    """
        Function that converts angles from radians to degrees.

    """
    return 180. * tensor / pi.to(tensor.device).type(tensor.dtype)


def deg2rad(tensor):
    """
            Function that converts angles from degrees to radians.
    """
    return tensor * pi.to(tensor.device).type(tensor.dtype) / 180.


def pol2cart(rho, phi):
    """
        Function that converts polar coordinates to cartesian coordinates.
    """

    x = rho * torch.cos(phi)
    y = rho * torch.sin(phi)
    return x, y


def cart2pol(x, y, eps=1e-8):
    """
        Function that converts cartesian coordinates to polar coordinates.
    """

    rho = torch.sqrt(x**2 + y**2 + eps)
    phi = torch.atan2(y, x)
    return rho, phi


def convert_points_from_homogeneous(points, eps=1e-8):
    """"
        Function that converts points from homogeneous to Euclidean space.

    """

    if len(points.shape) < 2:
        raise ValueError("Input must be at least a 2D tensor. Got {}".format(
            points.shape))

    # we check for points at infinity
    z_vec = points[..., -1:]

    mask = torch.abs(z_vec) > eps
    scale = torch.ones_like(z_vec).masked_scatter_(mask, torch.tensor(1.0).to(points.device) / z_vec[mask])

    return scale * points[..., :-1]


def convert_points_to_homogeneous(points):
    """
        Function that converts points from Euclidean to homogeneous space.

    """

    if len(points.shape) < 2:
        raise ValueError("Input must be at least a 2D tensor. Got {}".format(
            points.shape))

    return torch.nn.functional.pad(points, [0, 1], "constant", 1.0)


def convert_affinematrix_to_homography_impl(A):
    H = torch.nn.functional.pad(A, [0, 0, 0, 1], "constant", value=0.)
    H[..., -1, -1] += 1.0
    return H


def convert_affinematrix_to_homography(A):
    """
        Function that converts batch of affine matrices from [Bx2x3] to [Bx3x3].
    """

    if not (len(A.shape) == 3 and A.shape[-2:] == (2, 3)):
        raise ValueError("Input matrix must be a Bx2x3 tensor. Got {}"
                         .format(A.shape))

    return convert_affinematrix_to_homography_impl(A)


def convert_affinematrix_to_homography3d(A):
    """
        Function that converts batch of affine matrices from [Bx3x4] to [Bx4x4].
    """

    if not (len(A.shape) == 3 and A.shape[-2:] == (3, 4)):
        raise ValueError("Input matrix must be a Bx3x4 tensor. Got {}"
                         .format(A.shape))

    return convert_affinematrix_to_homography_impl(A)


def angle_axis_to_rotation_matrix(angle_axis):
    """
        Convert 3d vector of axis-angle rotation to 3x3 rotation matrix
    """

    if not angle_axis.shape[-1] == 3:
        raise ValueError(
            "Input size must be a (*, 3) tensor. Got {}".format(
                angle_axis.shape))

    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.

        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)

        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)

        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)

        rotation_matrix = torch.cat([r00, r01, r02,
                                     r10, r11, r12,
                                     r20, r21, r22], dim=1)

        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)

        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat([k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)

        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(3).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 3, 3).repeat(batch_size, 1, 1)

    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor

    return rotation_matrix  # Nx3x3


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
        Convert 3x3 rotation matrix to Rodrigues vector.
    """

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            "Input size must be a (*, 3, 3) tensor. Got {}".format(
                rotation_matrix.shape))

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)

    return quaternion_to_angle_axis(quaternion)


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-8):
    """
        Convert 3x3 rotation matrix to 4d quaternion vector.
        The quaternion vector has components in (x, y, z, w) format.
    """

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            "Input size must be a (*, 3, 3) tensor. Got {}".format(rotation_matrix.shape))

    def safe_zero_division(numerator, denominator):

        eps = torch.finfo(numerator.dtype).tiny

        return numerator / torch.clamp(denominator, min=eps)

    rotation_matrix_vec = rotation_matrix.view(*rotation_matrix.shape[:-2], 9)

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(rotation_matrix_vec, chunks=9, dim=-1)

    trace = m00 + m11 + m22

    def trace_positive_cond():
        sq = torch.sqrt(trace + 1.0) * 2.  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        return torch.cat([qx, qy, qz, qw], dim=-1)

    def cond_1():
        sq = torch.sqrt(1.0 + m00 - m11 - m22 + eps) * 2.  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        return torch.cat([qx, qy, qz, qw], dim=-1)

    def cond_2():
        sq = torch.sqrt(1.0 + m11 - m00 - m22 + eps) * 2.  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        return torch.cat([qx, qy, qz, qw], dim=-1)

    def cond_3():
        sq = torch.sqrt(1.0 + m22 - m00 - m11 + eps) * 2.  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        return torch.cat([qx, qy, qz, qw], dim=-1)

    where_2 = torch.where(m11 > m22, cond_2(), cond_3())
    where_1 = torch.where((m00 > m11) & (m00 > m22), cond_1(), where_2)

    quaternion = torch.where(trace > 0., trace_positive_cond(), where_1)

    return quaternion


def normalize_quaternion(quaternion, eps=1e-12):
    """
        Normalizes a quaternion. The quaternion should be in (x, y, z, w) format.
    """

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(quaternion.shape))

    return F.normalize(quaternion, p=2, dim=-1, eps=eps)


def quaternion_to_rotation_matrix(quaternion):
    """
        Converts a quaternion to a rotation matrix. The quaternion should be in (x, y, z, w) format.
    """

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(quaternion.shape))

    # normalize the input quaternion
    quaternion_norm = normalize_quaternion(quaternion)

    # unpack the normalized quaternion components
    x, y, z, w = torch.chunk(quaternion_norm, chunks=4, dim=-1)

    # compute the actual conversion
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z

    twx = tx * w
    twy = ty * w
    twz = tz * w

    txx = tx * x
    txy = ty * x
    txz = tz * x

    tyy = ty * y
    tyz = tz * y
    tzz = tz * z

    one = torch.tensor(1.)

    matrix = torch.stack([
        one - (tyy + tzz), txy - twz, txz + twy,
        txy + twz, one - (txx + tzz), tyz - twx,
        txz - twy, tyz + twx, one - (txx + tyy)], dim=-1).view(-1, 3, 3)

    if len(quaternion.shape) == 1:
        matrix = torch.squeeze(matrix, dim=0)

    return matrix


def quaternion_to_angle_axis(quaternion):
    """
        Convert quaternion vector to angle axis of rotation. The quaternion should be in (x, y, z, w) format.
    """

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape Nx4 or 4. Got {}".format(
                quaternion.shape))

    # unpack input and compute conversion
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]

    sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta = torch.sqrt(sin_squared_theta)
    cos_theta = quaternion[..., 0]

    two_theta = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos = two_theta / sin_theta
    k_neg = 2.0 * torch.ones_like(sin_theta)

    k = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k

    return angle_axis


def quaternion_log_to_exp(quaternion, eps=1e-8):
    """
        Applies exponential map to log quaternion. The quaternion should be in (x, y, z, w) format.
    """

    if not quaternion.shape[-1] == 3:
        raise ValueError(
            "Input must be a tensor of shape (*, 3). Got {}".format(
                quaternion.shape))

    # compute quaternion norm
    norm_q = torch.norm(quaternion, p=2, dim=-1, keepdim=True).clamp(min=eps)

    # compute scalar and vector
    quaternion_vector = quaternion * torch.sin(norm_q) / norm_q
    quaternion_scalar = torch.cos(norm_q)

    # compose quaternion and return
    quaternion_exp = torch.cat([quaternion_vector, quaternion_scalar], dim=-1)

    return quaternion_exp


def quaternion_exp_to_log(quaternion, eps=1e-8):
    """
        Applies the log map to a quaternion. The quaternion should be in (x, y, z, w) format.
    """

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(quaternion.shape))

    # unpack quaternion vector and scalar
    quaternion_vector = quaternion[..., 0:3]
    quaternion_scalar = quaternion[..., 3:4]

    # compute quaternion norm
    norm_q = torch.norm(quaternion_vector, p=2, dim=-1, keepdim=True).clamp(min=eps)

    # apply log map
    quaternion_log = quaternion_vector * torch.acos(torch.clamp(quaternion_scalar, min=-1.0, max=1.0)) / norm_q

    return quaternion_log


def angle_axis_to_quaternion(angle_axis):
    """
        Convert an angle axis to a quaternion. The quaternion vector has components in (x, y, z, w) format.
    """

    if not angle_axis.shape[-1] == 3:
        raise ValueError(
            "Input must be a tensor of shape Nx3 or 3. Got {}".format(angle_axis.shape))

    # unpack input and compute conversion
    a0 = angle_axis[..., 0:1]
    a1 = angle_axis[..., 1:2]
    a2 = angle_axis[..., 2:3]

    theta_squared = a0 * a0 + a1 * a1 + a2 * a2

    theta = torch.sqrt(theta_squared)
    half_theta = theta * 0.5

    mask = theta_squared > 0.0
    ones = torch.ones_like(half_theta)

    k_neg = 0.5 * ones
    k_pos = torch.sin(half_theta) / theta
    k = torch.where(mask, k_pos, k_neg)
    w = torch.where(mask, torch.cos(half_theta), ones)

    quaternion = torch.zeros_like(angle_axis)
    quaternion[..., 0:1] += a0 * k
    quaternion[..., 1:2] += a1 * k
    quaternion[..., 2:3] += a2 * k

    return torch.cat([w, quaternion], dim=-1)


def normalize_pixel_coordinates(pixel_coordinates, height, width, eps=1e-8):
    """
        Normalize pixel coordinates between -1 and 1. Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1)
    """

    if pixel_coordinates.shape[-1] != 2:
        raise ValueError("Input pixel_coordinates must be of shape (*, 2). "
                         "Got {}".format(pixel_coordinates.shape))

    # compute normalization factor
    hw = torch.stack([
        torch.tensor(width, device=pixel_coordinates.device, dtype=pixel_coordinates.dtype),
        torch.tensor(height, device=pixel_coordinates.device, dtype=pixel_coordinates.dtype)
    ])

    factor = torch.tensor(2., device=pixel_coordinates.device, dtype=pixel_coordinates.dtype) / (hw - 1).clamp(eps)

    return factor * pixel_coordinates - 1


def denormalize_pixel_coordinates(pixel_coordinates, height, width, eps=1e-8):
    """
        Denormalize pixel coordinates. The input is assumed to be -1 if on extreme left, 1 if on extreme right (x = w-1)
    """

    if pixel_coordinates.shape[-1] != 2:
        raise ValueError("Input pixel_coordinates must be of shape (*, 2). "
                         "Got {}".format(pixel_coordinates.shape))

    # compute normalization factor
    hw = torch.stack([
        torch.tensor(width), torch.tensor(height)
    ]).to(pixel_coordinates.device).to(pixel_coordinates.dtype)

    factor = torch.tensor(2.) / (hw - 1).clamp(eps)

    return torch.tensor(1.) / factor * (pixel_coordinates + 1)


def normalize_pixel_coordinates3d(pixel_coordinates, depth, height, width, eps=1e-8):
    """
        Normalize pixel coordinates between -1 and 1. Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1)
    """

    if pixel_coordinates.shape[-1] != 3:
        raise ValueError("Input pixel_coordinates must be of shape (*, 3). "
                         "Got {}".format(pixel_coordinates.shape))

    # compute normalization factor
    dhw = torch.stack([
        torch.tensor(depth), torch.tensor(width), torch.tensor(height)
    ]).to(pixel_coordinates.device).to(pixel_coordinates.dtype)

    factor = torch.tensor(2.) / (dhw - 1).clamp(eps)

    return factor * pixel_coordinates - 1


def denormalize_pixel_coordinates3d(pixel_coordinates, depth, height, width, eps=1e-8):
    """
        Denormalize pixel coordinates. The input is assumed to be -1 if on extreme left, 1 if on extreme right (x = w-1)
    """

    if pixel_coordinates.shape[-1] != 3:
        raise ValueError("Input pixel_coordinates must be of shape (*, 3). "
                         "Got {}".format(pixel_coordinates.shape))

    # compute normalization factor
    dhw = torch.stack([
        torch.tensor(depth), torch.tensor(width), torch.tensor(height)
    ]).to(pixel_coordinates.device).to(pixel_coordinates.dtype)

    factor = torch.tensor(2.) / (dhw - 1).clamp(eps)

    return torch.tensor(1.) / factor * (pixel_coordinates + 1)
