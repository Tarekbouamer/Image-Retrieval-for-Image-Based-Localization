from typing import Optional

import torch
from .conversions import convert_points_to_homogeneous, convert_points_from_homogeneous

__all__ = [
    "cross_product_matrix",
    "eye_like",
    "vec_like",

    "compose_transformations",
    "relative_transformation",
    "inverse_transformation",
    "transform_points",
    "transform_boxes",
    "perspective_transform_lafs",
]


def cross_product_matrix(x):
    """
        Returns the cross_product_matrix symmetric matrix of a vector.
    """

    assert len(x.shape) == 2 and x.shape[1] == 3, x.shape

    # get vector compononens
    x0 = x[..., 0]
    x1 = x[..., 1]
    x2 = x[..., 2]

    # construct the matrix, reshape to 3x3 and return
    zeros = torch.zeros_like(x0)
    cross_product_matrix_flat = torch.stack([
        zeros,  -x2,    x1,
        x2,     zeros, -x0,
        -x1,    x0,     zeros], dim=-1)

    return cross_product_matrix_flat.view(-1, 3, 3)


def eye_like(n, input):
    """
        Returns a 2-D tensor with ones on the diagonal and zeros elsewhere with same size as the input.
    """
    assert n > 0, (type(n), n)

    assert len(input.shape) >= 1, input.shape

    identity = torch.eye(n, device=input.device, dtype=input.dtype)

    return identity[None].repeat(input.shape[0], 1, 1)


def vec_like(n, tensor):
    """
        Returns a 2-D tensor with a vector containing zeros with same size as the input.
    """
    assert n > 0, (type(n), n)
    assert len(tensor.shape) >= 1, tensor.shape

    vec = torch.zeros(n, 1, device=tensor.device, dtype=tensor.dtype)

    return vec[None].repeat(tensor.shape[0], 1, 1)


def compose_transformations(trans_01, trans_12):
    """
        Functions that composes two homogeneous transformations.
    """

    if not trans_01.dim() in (2, 3) and trans_01.shape[-2:] == (4, 4):
        raise ValueError("Input trans_01 must be a of the shape Nx4x4 or 4x4."
                         " Got {}".format(trans_01.shape))

    if not trans_12.dim() in (2, 3) and trans_12.shape[-2:] == (4, 4):
        raise ValueError("Input trans_12 must be a of the shape Nx4x4 or 4x4."
                         " Got {}".format(trans_12.shape))

    if not trans_01.dim() == trans_12.dim():
        raise ValueError("Input number of dims must match. Got {} and {}"
                         .format(trans_01.dim(), trans_12.dim()))
    # unpack input data
    rmat_01 = trans_01[..., :3, :3]  # Nx3x3
    rmat_12 = trans_12[..., :3, :3]  # Nx3x3

    tvec_01 = trans_01[..., :3, -1:]  # Nx3x1
    tvec_12 = trans_12[..., :3, -1:]  # Nx3x1

    # compute the actual transforms composition
    rmat_02 = torch.matmul(rmat_01, rmat_12)
    tvec_02 = torch.matmul(rmat_01, tvec_12) + tvec_01

    # pack output tensor
    trans_02 = torch.zeros_like(trans_01)

    trans_02[..., :3, 0:3] += rmat_02
    trans_02[..., :3, -1:] += tvec_02
    trans_02[..., -1, -1:] += 1.0

    return trans_02


def inverse_transformation(trans_12):
    """
        Function that inverts a 4x4 homogeneous transformation
    """

    if not trans_12.dim() in (2, 3) and trans_12.shape[-2:] == (4, 4):
        raise ValueError("Input size must be a Nx4x4 or 4x4. Got {}"
                         .format(trans_12.shape))

    # unpack input tensor
    rmat_12 = trans_12[..., :3, 0:3]  # Nx3x3
    tvec_12 = trans_12[..., :3, 3:4]  # Nx3x1

    # compute the actual inverse
    rmat_21 = torch.transpose(rmat_12, -1, -2)
    tvec_21 = torch.matmul(-rmat_21, tvec_12)

    # pack to output tensor
    trans_21 = torch.zeros_like(trans_12)
    trans_21[..., :3, 0:3] += rmat_21
    trans_21[..., :3, -1:] += tvec_21
    trans_21[..., -1, -1:] += 1.0

    return trans_21


def relative_transformation(trans_01, trans_02):
    """
        Function that computes the relative homogenous transformation from a
    """
    if not trans_01.dim() in (2, 3) and trans_01.shape[-2:] == (4, 4):
        raise ValueError("Input must be a of the shape Nx4x4 or 4x4."
                         " Got {}".format(trans_01.shape))
    if not trans_02.dim() in (2, 3) and trans_02.shape[-2:] == (4, 4):
        raise ValueError("Input must be a of the shape Nx4x4 or 4x4."
                         " Got {}".format(trans_02.shape))
    if not trans_01.dim() == trans_02.dim():
        raise ValueError("Input number of dims must match. Got {} and {}"
                         .format(trans_01.dim(), trans_02.dim()))

    trans_10 = inverse_transformation(trans_01)
    trans_12 = compose_transformations(trans_10, trans_02)

    return trans_12


def transform_points(trans_01, points_1):
    """
        Function that applies transformations to a set of points.
    """

    if not (trans_01.device == points_1.device and trans_01.dtype == points_1.dtype):
        raise TypeError(
            "Tensor must be in the same device and dtype. "
            f"Got trans_01 with ({trans_01.dtype}, {points_1.dtype}) and "
            f"points_1 with ({points_1.dtype}, {points_1.dtype})")

    if not trans_01.shape[0] == points_1.shape[0] and trans_01.shape[0] != 1:
        raise ValueError("Input batch size must be the same for both tensors or 1")

    if not trans_01.shape[-1] == (points_1.shape[-1] + 1):
        raise ValueError("Last input dimensions must differ by one unit")

    # We reshape to BxNxD in case we get more dimensions, e.g., MxBxNxD
    shape_inp = list(points_1.shape)
    points_1 = points_1.reshape(-1, points_1.shape[-2], points_1.shape[-1])
    trans_01 = trans_01.reshape(-1, trans_01.shape[-2], trans_01.shape[-1])

    # We expand trans_01 to match the dimensions needed for bmm
    trans_01 = torch.repeat_interleave(trans_01, repeats=points_1.shape[0] // trans_01.shape[0], dim=0)

    # to homogeneous
    points_1_h = convert_points_to_homogeneous(points_1)  # BxNxD+1

    # transform coordinates
    points_0_h = torch.bmm(points_1_h, trans_01.permute(0, 2, 1))
    points_0_h = torch.squeeze(points_0_h, dim=-1)

    # to euclidean
    points_0 = convert_points_from_homogeneous(points_0_h)  # BxNxD

    # reshape to the input shape
    shape_inp[-2] = points_0.shape[-2]
    shape_inp[-1] = points_0.shape[-1]
    points_0 = points_0.reshape(shape_inp)

    return points_0


def transform_boxes(trans_mat, boxes, mode="xyxy"):
    """
        Function that applies a transformation matrix to a box or batch of boxes. Boxes must
        be a tensor of the shape (N, 4) or a batch of boxes (B, N, 4) and trans_mat must be a (3, 3)
        transformation matrix or a batch of transformation matrices (B, 3, 3)
    """

    if not isinstance(mode, str):
        raise TypeError(f"Mode must be a string. Got {type(mode)}")

    if mode not in ("xyxy", "xywh"):
        raise ValueError(f"Mode must be one of 'xyxy', 'xywh'. Got {mode}")

    # convert boxes to format xyxy
    if mode == "xywh":
        boxes[..., -2] = boxes[..., 0] + boxes[..., -2]  # x + w
        boxes[..., -1] = boxes[..., 1] + boxes[..., -1]  # y + h

    transformed_boxes: torch.Tensor = kornia.transform_points(trans_mat, boxes.view(boxes.shape[0], -1, 2))
    transformed_boxes = transformed_boxes.view_as(boxes)

    if mode == 'xywh':
        transformed_boxes[..., 2] = transformed_boxes[..., 2] - transformed_boxes[..., 0]
        transformed_boxes[..., 3] = transformed_boxes[..., 3] - transformed_boxes[..., 1]

    return transformed_boxes


def perspective_transform_lafs(trans_01, lafs_1):
    """
        Function that applies perspective transformations to a set of local affine frames (LAFs).

    """
    kornia.feature.laf.raise_error_if_laf_is_not_valid(lafs_1)

    if not trans_01.device == lafs_1.device:
        raise TypeError("Tensor must be in the same device")

    if not trans_01.shape[0] == lafs_1.shape[0]:
        raise ValueError("Input batch size must be the same for both tensors")

    if (not (trans_01.shape[-1] == 3)) or (not (trans_01.shape[-2] == 3)):
        raise ValueError("Transformation should be homography")

    bs, n, _, _ = lafs_1.size()

    # First, we convert LAF to points
    threepts_1 = kornia.feature.laf.laf_to_three_points(lafs_1)
    points_1 = threepts_1.permute(0, 1, 3, 2).reshape(bs, n * 3, 2)

    # First, transform the points
    points_0 = transform_points(trans_01, points_1)

    # Back to LAF format
    threepts_0 = points_0.view(bs, n, 3, 2).permute(0, 1, 3, 2)
    return kornia.feature.laf.laf_from_three_points(threepts_0)