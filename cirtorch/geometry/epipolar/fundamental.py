import torch
from ..conversions import convert_points_to_homogeneous
from ..linalg import transform_points


def normalize_points(points, eps=1e-8):
    """
        Normalizes points (isotropic).
        Computes the transformation matrix such that the two principal moments of the set of points
        are equal to unity, forming an approximately symmetric circular cloud of points of radius 1
        about the origin. Reference: Hartley/Zisserman 4.4.4 pag.107
        This operation is an essential step before applying the DLT algorithm in order to consider
        the result as optimal.
    """
    assert len(points.shape) == 3, points.shape
    assert points.shape[-1] == 2, points.shape

    x_mean = torch.mean(points, dim=1, keepdim=True)  # Bx1x2

    scale = (points - x_mean).norm(dim=-1).mean(dim=-1)   # B
    scale = torch.sqrt(torch.tensor(2.)) / (scale + eps)  # B

    ones, zeros = torch.ones_like(scale), torch.zeros_like(scale)

    transform = torch.stack([
        scale, zeros, -scale * x_mean[..., 0, 0],
        zeros, scale, -scale * x_mean[..., 0, 1],
        zeros, zeros, ones], dim=-1)  # Bx9

    transform = transform.view(-1, 3, 3)  # Bx3x3
    points_norm = transform_points(transform, points)  # BxNx2

    return (points_norm, transform)


def normalize_transformation(M, eps=1e-8):
    """
        Normalizes a given transformation matrix.
        The function trakes the transformation matrix and normalize so that the value in
        the last row and column is one.
    """
    assert len(M.shape) >= 2, M.shape

    norm_val = M[..., -1:, -1:]

    return torch.where(norm_val.abs() > eps, M / (norm_val + eps), M)


def find_fundamental(points1, points2, weights):
    """
        Computes the fundamental matrix using the DLT formulation.
        The linear system is solved by using the Weighted Least Squares Solution for the 8 Points algorithm.
    """
    assert points1.shape == points2.shape, (points1.shape, points2.shape)
    assert len(weights.shape) == 2 and weights.shape[1] == points1.shape[1], weights.shape

    points1_norm, transform1 = normalize_points(points1)
    points2_norm, transform2 = normalize_points(points2)

    x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # Bx1xN
    x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # Bx1xN

    ones = torch.ones_like(x1)

    # build equations system and solve DLT
    # https://www.cc.gatech.edu/~afb/classes/CS4495-Fall2013/slides/CS4495-09-TwoViews-2.pdf
    # [x * x', x * y', x, y * x', y * y', y, x', y', 1]

    X = torch.cat([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, ones], dim=-1)  # BxNx9

    # apply the weights to the linear system
    w_diag = torch.diag_embed(weights)
    X = X.transpose(-2, -1) @ w_diag @ X

    # compute eigevectors and retrieve the one with the smallest eigenvalue
    _, _, V = torch.svd(X)
    F_mat = V[..., -1].view(-1, 3, 3)

    # reconstruct and force the matrix to have rank2
    U, S, V = torch.svd(F_mat)
    rank_mask = torch.tensor([1., 1., 0]).to(F_mat.device)

    F_projected = U @ (torch.diag_embed(S * rank_mask) @ V.transpose(-2, -1))
    F_est = transform2.transpose(-2, -1) @ (F_projected @ transform1)

    return normalize_transformation(F_est)


def compute_correspond_epilines(points, F_mat):
    """
        Computes the corresponding epipolar line for a given set of points.

    """
    assert len(points.shape) == 3 and points.shape[2] == 2, points.shape
    assert len(F_mat.shape) == 3 and F_mat.shape[-2:] == (3, 3), F_mat.shape

    points_h = convert_points_to_homogeneous(points)

    # project points and retrieve lines components
    a, b, c = torch.chunk(F_mat @ points_h.permute(0, 2, 1), dim=1, chunks=3)

    # compute normal and compose equation line
    nu = a * a + b * b
    nu = torch.where(nu > 0., 1. / torch.sqrt(nu), torch.ones_like(nu))

    line = torch.cat([a * nu, b * nu, c * nu], dim=1)  # Bx3xN

    return line.permute(0, 2, 1)  # BxNx3


def fundamental_from_essential(E_mat, K1, K2):
    """
        Get the Fundamental matrix from Essential and camera matrices.
        Uses the method from Hartley/Zisserman 9.6 pag 257 (formula 9.12).

    """
    assert len(E_mat.shape) >= 2 and E_mat.shape[-2:] == (3, 3), E_mat.shape
    assert len(K1.shape) >= 2 and K1.shape[-2:] == (3, 3), K1.shape
    assert len(K2.shape) >= 2 and K2.shape[-2:] == (3, 3), K2.shape
    assert len(E_mat.shape[:-2]) == len(K1.shape[:-2]) == len(K2.shape[:-2])

    return K2.inverse().transpose(-2, -1) @ E_mat @ K1.inverse()


# adapted from:
# https://github.com/openMVG/openMVG/blob/160643be515007580086650f2ae7f1a42d32e9fb/src/openMVG/multiview/projection.cpp#L134

def fundamental_from_projections(P1, P2):
    """
        Get the Fundamental matrix from Projection matrices.
    """
    assert len(P1.shape) >= 2 and P1.shape[-2:] == (3, 4), P1.shape
    assert len(P2.shape) >= 2 and P2.shape[-2:] == (3, 4), P2.shape
    assert P1.shape[:-2] == P2.shape[:-2]  # this function does not support broadcasting

    def vstack(x, y):
        return torch.cat([x, y], dim=-2)

    X1 = P1[..., 1:, :]
    X2 = vstack(P1[..., 2:3, :], P1[..., 0:1, :])
    X3 = P1[..., :2, :]

    Y1 = P2[..., 1:, :]
    Y2 = vstack(P2[..., 2:3, :], P2[..., 0:1, :])
    Y3 = P2[..., :2, :]

    X1Y1, X2Y1, X3Y1 = vstack(X1, Y1), vstack(X2, Y1), vstack(X3, Y1)
    X1Y2, X2Y2, X3Y2 = vstack(X1, Y2), vstack(X2, Y2), vstack(X3, Y2)
    X1Y3, X2Y3, X3Y3 = vstack(X1, Y3), vstack(X2, Y3), vstack(X3, Y3)

    F_vec = torch.cat([
        X1Y1.det(), X2Y1.det(), X3Y1.det(),
        X1Y2.det(), X2Y2.det(), X3Y2.det(),
        X1Y3.det(), X2Y3.det(), X3Y3.det(),
    ], dim=-1)

    return F_vec.view(*P1.shape[:-2], 3, 3)
