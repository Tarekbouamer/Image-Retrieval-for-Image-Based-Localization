import torch

from .imgwarp import warp_affine, get_rotation_matrix2d, get_affine_matrix2d
from .projwarp import warp_affine3d, get_projective_transform

from cirtorch.utils.helper import _extract_device_dtype

from ..conversions import convert_affinematrix_to_homography

__all__ = [
    "affine",
    "scale",
    "rotate",
    "rotate3d",
    "translate",
    "shear",
    "resize",
    "rescale"
]

# TODO: move functions that starts with _ to mose tensor operations


def _compute_tensor_center(size, device, dtype):
    """
        Computes the center of tensor plane for (H, W), (C, H, W) and (B, C, H, W).
    """

    height, width = size

    center_x = float(width - 1) / 2
    center_y = float(height - 1) / 2

    center = torch.tensor([center_x, center_y], device=device, dtype=dtype)

    return center


def _compute_tensor_center3d(tensor):
    """
        Computes the center of tensor plane for (D, H, W), (C, D, H, W) and (B, C, D, H, W).
    """

    assert 3 <= len(tensor.shape) <= 5, f"Must be a 3D tensor as DHW, CDHW and BCDHW. Got {tensor.shape}."
    depth, height, width = tensor.shape[-3:]

    center_x = float(width - 1) / 2
    center_y = float(height - 1) / 2
    center_z = float(depth - 1) / 2

    center = torch.tensor([center_x, center_y, center_z], device=tensor.device, dtype=tensor.dtype)

    return center


def _compute_rotation_matrix(angle, center):
    """
        Computes a pure affine rotation matrix.
    """

    scale = torch.ones_like(center)
    
    matrix = get_rotation_matrix2d(center, angle, scale)

    # pad transform to get Bx3x3
    matrix = convert_affinematrix_to_homography(matrix)

    return matrix


def _compute_rotation_matrix3d(yaw, pitch, roll, center):
    """
        Computes a pure affine rotation matrix.
    """
    if len(yaw.shape) == len(pitch.shape) == len(roll.shape) == 0:
        yaw = yaw.unsqueeze(dim=0)
        pitch = pitch.unsqueeze(dim=0)
        roll = roll.unsqueeze(dim=0)

    if len(yaw.shape) == len(pitch.shape) == len(roll.shape) == 1:
        yaw = yaw.unsqueeze(dim=1)
        pitch = pitch.unsqueeze(dim=1)
        roll = roll.unsqueeze(dim=1)

    assert len(yaw.shape) == len(pitch.shape) == len(roll.shape) == 2, \
        f"Expected yaw, pitch, roll to be (B, 1). Got {yaw.shape}, {pitch.shape}, {roll.shape}."

    angles = torch.cat([yaw, pitch, roll], dim=1)
    scales = torch.ones_like(yaw)
    matrix = get_projective_transform(center, angles, scales)

    return matrix


def _compute_translation_matrix(translation):
    """
        Computes affine matrix for translation.
    """
    matrix = torch.eye(3, device=translation.device, dtype=translation.dtype)
    matrix = matrix.repeat(translation.shape[0], 1, 1)

    dx, dy = torch.chunk(translation, chunks=2, dim=-1)

    matrix[..., 0, 2:3] += dx
    matrix[..., 1, 2:3] += dy

    return matrix


def _compute_scaling_matrix(scale, center):
    """
        Computes affine matrix for scaling.
    """

    angle = torch.zeros(scale.shape[:1], device=scale.device, dtype=scale.dtype)
    matrix = get_rotation_matrix2d(center, angle, scale)

    return matrix


def _compute_shear_matrix(shear):
    """
        Computes affine matrix for shearing.
    """

    matrix = torch.eye(3, device=shear.device, dtype=shear.dtype)
    matrix = matrix.repeat(shear.shape[0], 1, 1)

    shx, shy = torch.chunk(shear, chunks=2, dim=-1)
    matrix[..., 0, 1:2] += shx
    matrix[..., 1, 0:1] += shy

    return matrix


# based on:
# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L166

def affine(tensor, matrix, mode='bilinear', align_corners=False):
    """
        Apply an affine transformation to the image.
    """

    # warping needs data in the shape of BCHW
    is_unbatched = tensor.ndimension() == 3

    if is_unbatched:
        tensor = torch.unsqueeze(tensor, dim=0)

    # we enforce broadcasting since by default grid_sample it does not
    # give support for that

    matrix = matrix.expand(tensor.shape[0], -1, -1)

    # warp the input tensor
    height = tensor.shape[-2]
    width = tensor.shape[-1]

    warped = warp_affine(tensor, matrix, (height, width),
                         mode=mode,
                         align_corners=align_corners)

    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped, dim=0)

    return warped


def affine3d(tensor, matrix, mode='bilinear', align_corners=False):
    """
        Apply an affine transformation to the 3d volume.
    """

    # warping needs data in the shape of BCDHW
    is_unbatched: bool = tensor.ndimension() == 4

    if is_unbatched:
        tensor = torch.unsqueeze(tensor, dim=0)

    # we enforce broadcasting since by default grid_sample it does not
    # give support for that
    matrix = matrix.expand(tensor.shape[0], -1, -1)

    # warp the input tensor
    depth = tensor.shape[-3]
    height = tensor.shape[-2]
    width = tensor.shape[-1]

    warped = warp_affine3d(tensor, matrix, (depth, height, width),
                           mode=mode,
                           align_corners=align_corners)

    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped, dim=0)

    return warped


# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L185


def rotate(tensor, M_rot, mode='bilinear', align_corners=False):
    """
        Rotate the image anti-clockwise about the centre.
    """
    if len(tensor.shape) not in (3, 4,):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. "
                         "Got: {}".format(tensor.shape))
    
    # warp using the affine transform
    return affine(tensor, M_rot , mode, align_corners)


def rotate3d(tensor, yaw, pitch, roll, center=None, mode='bilinear', align_corners=False):
    """
        Rotate the image anti-clockwise about the centre.
    """

    if center is not None and not torch.is_tensor(center):
        raise TypeError("Input center type is not a torch.Tensor. Got {}"
                        .format(type(center)))
    if len(tensor.shape) not in (4, 5,):
        raise ValueError("Invalid tensor shape, we expect CxDxHxW or BxCxDxHxW. "
                         "Got: {}".format(tensor.shape))

    # compute the rotation center
    if center is None:
        center = _compute_tensor_center3d(tensor)

    # compute the rotation matrix
    yaw = yaw.expand(tensor.shape[0])
    pitch = pitch.expand(tensor.shape[0])
    roll = roll.expand(tensor.shape[0])
    center = center.expand(tensor.shape[0], -1)
    rotation_matrix = _compute_rotation_matrix3d(yaw, pitch, roll, center)

    # warp using the affine transform
    return affine3d(tensor, rotation_matrix[..., :3, :4],
                    mode=mode,
                    align_corners=align_corners)


def translate(tensor, translation, align_corners=False):
    """
        Translate the tensor in pixel units.
    """

    if len(tensor.shape) not in (3, 4,):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. "
                         "Got: {}".format(tensor.shape))

    # compute the translation matrix
    translation_matrix = _compute_translation_matrix(translation)

    # warp using the affine transform
    return affine(tensor, translation_matrix[..., :2, :3], align_corners=align_corners)


def scale(tensor, scale_factor, center=None, align_corners=False):
    """
        Scales the input image.
    """

    if len(scale_factor.shape) == 1:
        # convert isotropic scaling to x-y direction
        scale_factor = scale_factor.repeat(1, 2)

    # compute the tensor center
    if center is None:
        center = _compute_tensor_center(tensor)

    # compute the rotation matrix
    center = center.expand(tensor.shape[0], -1)
    scale_factor = scale_factor.expand(tensor.shape[0], 2)

    scaling_matrix= _compute_scaling_matrix(scale_factor, center)

    # warp using the affine transform
    return affine(tensor, scaling_matrix[..., :2, :3], align_corners=align_corners)


def shear(tensor, shear, align_corners=False):
    """
        Shear the tensor.
    """

    if len(tensor.shape) not in (3, 4,):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. "
                         "Got: {}".format(tensor.shape))

    # compute the translation matrix
    shear_matrix = _compute_shear_matrix(shear)

    # warp using the affine transform
    return affine(tensor, shear_matrix[..., :2, :3], align_corners=align_corners)


def _side_to_image_size(side_size, aspect_ratio, side="short"):

    if side not in ("short", "long", "vert", "horz"):
        raise ValueError(f"side can be one of 'short', 'long', 'vert', and 'horz'. Got '{side}'")

    if side == "vert":
        return side_size, int(side_size * aspect_ratio)
    elif side == "horz":
        return int(side_size / aspect_ratio), side_size
    elif (side == "short") ^ (aspect_ratio < 1.0):
        return side_size, int(side_size * aspect_ratio)
    else:
        return int(side_size / aspect_ratio), side_size


def resize(input, size, mode='bilinear', align_corners=False, side="short"):
    """
        Resize the input torch.Tensor to the given size.
    """

    input_size = h, w = input.shape[-2:]

    if isinstance(size, int):
        aspect_ratio = w / h
        size = _side_to_image_size(size, aspect_ratio, side)

    if size == input_size:
        return input

    return torch.nn.functional.interpolate(input, size=size, mode=mode, align_corners=align_corners)


def rescale(input, factor, mode="bilinear", align_corners=False):
    """
        Rescale the input torch.Tensor with the given factor.
    """
    if isinstance(factor, float):
        factor_vert = factor_horz = factor
    else:
        factor_vert, factor_horz = factor

    height, width = input.size()[-2:]
    size = (int(height * factor_vert), int(width * factor_horz))

    return resize(input, size, mode=mode, align_corners=align_corners)
