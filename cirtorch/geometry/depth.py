import torch
import torch.nn.functional as F

from .camera.perspective import project_points, unproject_points
from .conversions import normalize_pixel_coordinates
from .linalg import transform_points

from cirtorch.utils.grid import create_meshgrid

from kornia.filters import spatial_gradient


def depth_to_3d(depth, camera_matrix, normalize_points=False):
    """
        Compute a 3d point per pixel given its depth value and the camera intrinsics.
    """

    if not len(depth.shape) == 4 and depth.shape[-3] == 1:
        raise ValueError(f"Input depth musth have a shape (B, 1, H, W). Got: {depth.shape}")

    if not len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). "
                         f"Got: {camera_matrix.shape}.")

    # create base coordinates grid
    batch_size, _, height, width = depth.shape

    points_2d = create_meshgrid(height, width, normalized_coordinates=False)  # 1xHxWx2

    points_2d = points_2d.to(depth.device).to(depth.dtype)

    # depth should come in Bx1xHxW
    points_depth = depth.permute(0, 2, 3, 1)  # 1xHxWx1

    # project pixels to camera frame
    camera_matrix_tmp = camera_matrix[:, None, None]  # Bx1x1x3x3
    points_3d = unproject_points(points_2d, points_depth, camera_matrix_tmp, normalize=normalize_points)  # BxHxWx3

    return points_3d.permute(0, 3, 1, 2)  # Bx3xHxW


def depth_to_normals(depth, camera_matrix, normalize_points=False):
    """
        Compute the normal surface per pixel.
    """

    if not len(depth.shape) == 4 and depth.shape[-3] == 1:
        raise ValueError(f"Input depth musth have a shape (B, 1, H, W). Got: {depth.shape}")

    if not len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). "
                         f"Got: {camera_matrix.shape}.")

    # compute the 3d points from depth
    xyz = depth_to_3d(depth, camera_matrix, normalize_points)  # Bx3xHxW

    # compute the pointcloud spatial gradients
    gradients = spatial_gradient(xyz)  # Bx3x2xHxW

    # compute normals
    a, b = gradients[:, :, 0], gradients[:, :, 1]  # Bx3xHxW

    normals = torch.cross(a, b, dim=1)  # Bx3xHxW

    return F.normalize(normals, dim=1, p=2)


def warp_frame_depth(image_src, depth_dst, src_trans_dst, camera_matrix, normalize_points=False):
    """
        Warp a tensor from a source to destination frame by the depth in the destination.
        Compute 3d points from the depth, transform them using given transformation, then project the point cloud to an
        image plane.
    """

    if not len(image_src.shape) == 4:
        raise ValueError(f"Input image_src musth have a shape (B, D, H, W). Got: {image_src.shape}")

    if not len(depth_dst.shape) == 4 and depth_dst.shape[-3] == 1:
        raise ValueError(f"Input depth_dst musth have a shape (B, 1, H, W). Got: {depth_dst.shape}")

    if not len(src_trans_dst.shape) == 3 and src_trans_dst.shape[-2:] == (3, 3):
        raise ValueError(f"Input src_trans_dst must have a shape (B, 3, 3). "
                         f"Got: {src_trans_dst.shape}.")

    if not len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). "
                         f"Got: {camera_matrix.shape}.")

    # unproject source points to camera frame
    points_3d_dst = depth_to_3d(depth_dst, camera_matrix, normalize_points)  # Bx3xHxW

    # transform points from source to destination
    points_3d_dst = points_3d_dst.permute(0, 2, 3, 1)  # BxHxWx3

    # apply transformation to the 3d points
    points_3d_src = transform_points(src_trans_dst[:, None], points_3d_dst)  # BxHxWx3

    # project back to pixels
    camera_matrix_tmp = camera_matrix[:, None, None]  # Bx1x1xHxW
    points_2d_src = project_points(points_3d_src, camera_matrix_tmp)  # BxHxWx2

    # normalize points between [-1 / 1]
    height, width = depth_dst.shape[-2:]

    points_2d_src_norm = normalize_pixel_coordinates(points_2d_src, height, width)  # BxHxWx2

    return F.grid_sample(image_src, points_2d_src_norm, align_corners=True)
