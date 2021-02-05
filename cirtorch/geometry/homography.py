import warnings

import torch
from typing import Tuple, Optional

from .epipolar.fundamental import normalize_points
from .linalg import transform_points
from cirtorch.utils.helper import _extract_device_dtype


def find_homography_dlt(points1, points2, weights=None):
    """
        Computes the homography matrix using the DLT formulation.
        The linear system is solved by using the Weighted Least Squares Solution for the 4 Points algorithm.
    """

    assert points1.shape == points2.shape, points1.shape
    assert len(points1.shape) >= 1 and points1.shape[-1] == 2, points1.shape
    assert points1.shape[1] >= 4, points1.shape

    device, dtype = _extract_device_dtype([points1, points2])

    eps = 1e-8
    points1_norm, transform1 = normalize_points(points1)
    points2_norm, transform2 = normalize_points(points2)

    x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # BxNx1
    x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # BxNx1

    ones, zeros = torch.ones_like(x1), torch.zeros_like(x1)

    ax = torch.cat([zeros, zeros, zeros, -x1, -y1, -ones, y2 * x1, y2 * y1, y2], dim=-1)
    ay = torch.cat([x1, y1, ones, zeros, zeros, zeros, -x2 * x1, -x2 * y1, -x2], dim=-1)

    A = torch.cat((ax, ay), dim=-1).reshape(ax.shape[0], -1, ax.shape[-1])

    if weights is None:
        A = A.transpose(-2, -1) @ A

    else:
        assert len(weights.shape) == 2 and weights.shape == points1.shape[:2], weights.shape

        w_diag = torch.diag_embed(weights.unsqueeze(dim=-1).repeat(1, 1, 2).reshape(weights.shape[0], -1))
        A = A.transpose(-2, -1) @ w_diag @ A

    try:
        U, S, V = torch.svd(A)
    except:
        warnings.warn('SVD did not converge', RuntimeWarning)
        return torch.empty((points1_norm.size(0), 3, 3), device=device, dtype=dtype)

    H = V[..., -1].view(-1, 3, 3)

    H = transform2.inverse() @ (H @ transform1)

    H_norm = H / (H[..., -1:, -1:] + eps)

    return H_norm


def find_homography_dlt_iterated(points1, points2, weights, soft_inl_th=3.0, n_iter=5):
    """
        Computes the homography matrix using the iteratively-reweighted least squares (IRWLS).
        The linear system is solved by using the Reweighted Least Squares Solution for the 4 Points algorithm.
    """

    H = find_homography_dlt(points1, points2, weights)

    for i in range(n_iter - 1):
        pts1_in_2 = transform_points(H, points1)
        error_squared = (pts1_in_2 - points2).pow(2).sum(dim=-1)
        weights_new = torch.exp(-error_squared / (2.0 * (soft_inl_th ** 2)))
        H = find_homography_dlt(points1, points2, weights_new)
    return H