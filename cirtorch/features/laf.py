import math
import torch
import torch.nn.functional as F

from ..geometry.conversions import rad2deg, convert_points_from_homogeneous
from ..geometry.transform.imgwarp import angle_to_rotation_matrix
from ..geometry.transform.pyramid import pyrdown


def raise_error_if_laf_is_not_valid(laf):
    """
        Auxilary function, which verifies that input is a torch.tensor of [BxNx2x3] shape
    """

    laf_message = "Invalid laf shape, we expect BxNx2x3. Got: {}".format(laf.shape)

    if not torch.is_tensor(laf):
        raise TypeError("Laf type is not a torch.Tensor. Got {}"
                        .format(type(laf)))

    if len(laf.shape) != 4:
        raise ValueError(laf_message)

    if laf.size(2) != 2 or laf.size(3) != 3:
        raise ValueError(laf_message)

    return


def get_laf_scale(LAF, eps=1e-10):
    """
        Returns a scale of the LAFs
    """
    raise_error_if_laf_is_not_valid(LAF)

    out = LAF[..., 0:1, 0:1] * LAF[..., 1:2, 1:2] - LAF[..., 1:2, 0:1] * LAF[..., 0:1, 1:2] + eps

    return out.abs().sqrt()


def get_laf_center(LAF):
    """
        Returns a center (keypoint) of the LAFs
    """
    raise_error_if_laf_is_not_valid(LAF)

    out = LAF[..., 2]

    return out


def get_laf_orientation(LAF):
    """
        Returns orientation of the LAFs, in degrees.
    """
    raise_error_if_laf_is_not_valid(LAF)

    angle_rad = torch.atan2(LAF[..., 0, 1], LAF[..., 0, 0])

    return rad2deg(angle_rad).unsqueeze(-1)


def laf_from_center_scale_ori(xy, scale, ori):
    """
        Returns orientation of the LAFs, in radians. Useful to create kornia LAFs from OpenCV keypoints
    """
    names = ['xy', 'scale', 'ori']

    for var_name, var, req_shape in zip(names,
                                        [xy, scale, ori],
                                        [("B", "N", 2), ("B", "N", 1, 1), ("B", "N", 1)]):
        if not torch.is_tensor(var):
            raise TypeError("{} type is not a torch.Tensor. Got {}"
                            .format(var_name, type(var)))

        if len(var.shape) != len(req_shape):  # type: ignore  # because it does not like len(tensor.shape)
            raise TypeError("{} shape should be must be [{}]. "
                            "Got {}".format(var_name, str(req_shape), var.size()))

        for i, dim in enumerate(req_shape):

            if dim is not int:
                continue

            if var.size(i) != dim:
                raise TypeError("{} shape should be must be [{}]. "
                                "Got {}".format(var_name, str(req_shape), var.size()))

    unscaled_laf = torch.cat([angle_to_rotation_matrix(ori.squeeze(-1)), xy.unsqueeze(-1)], dim=-1)

    laf = scale_laf(unscaled_laf, scale)

    return laf


def scale_laf(laf, scale_coef):
    """
        Multiplies region part of LAF ([:, :, :2, :2]) by a scale_coefficient.
        So the center, shape and orientation of the local feature stays the same, but the region area changes.
    """
    if (type(scale_coef) is not float) and (type(scale_coef) is not torch.Tensor):
        raise TypeError("scale_coef should be float or torch.Tensor "
                        "Got {}".format(type(scale_coef)))

    raise_error_if_laf_is_not_valid(laf)

    centerless_laf = laf[:, :, :2, :2]

    return torch.cat([scale_coef * centerless_laf, laf[:, :, :, 2:]], dim=3)


def make_upright(laf, eps=1e-9):
    """
        Rectifies the affine matrix, so that it becomes upright
    """
    raise_error_if_laf_is_not_valid(laf)

    det = get_laf_scale(laf)
    scale = det

    # The function is equivalent to doing 2x2 SVD and reseting rotation
    # matrix to an identity: U, S, V = svd(LAF); LAF_upright = U * S.

    b2a2 = torch.sqrt(laf[..., 0:1, 1:2] ** 2 + laf[..., 0:1, 0:1] ** 2) + eps

    laf1_ell = torch.cat([(b2a2 / det).contiguous(), torch.zeros_like(det)], dim=3)

    laf2_ell = torch.cat([((laf[..., 1:2, 1:2] * laf[..., 0:1, 1:2] +
                            laf[..., 1:2, 0:1] * laf[..., 0:1, 0:1]) / (b2a2 * det)),
                          (det / b2a2).contiguous()], dim=3)

    laf_unit_scale = torch.cat([torch.cat([laf1_ell, laf2_ell], dim=2), laf[..., :, 2:3]], dim=3)

    return scale_laf(laf_unit_scale, scale)


def ellipse_to_laf(ells):
    """
        Converts ellipse regions to LAF format. Ellipse (a, b, c) and
        upright covariance matrix [a11 a12; 0 a22] are connected by inverse matrix square root:
    """

    n_dims = len(ells.size())

    if n_dims != 3:
        raise TypeError("ellipse shape should be must be [BxNx5]. "
                        "Got {}".format(ells.size()))

    B, N, dim = ells.size()

    if (dim != 5):
        raise TypeError("ellipse shape should be must be [BxNx5]. "
                        "Got {}".format(ells.size()))

    # Previous implementation was incorrectly using Cholesky decomp as matrix sqrt
    # ell_shape = torch.cat([torch.cat([ells[..., 2:3], ells[..., 3:4]], dim=2).unsqueeze(2),
    #                       torch.cat([ells[..., 3:4], ells[..., 4:5]], dim=2).unsqueeze(2)], dim=2).view(-1, 2, 2)
    # out = torch.matrix_power(torch.cholesky(ell_shape, False), -1).view(B, N, 2, 2)

    # We will calculate 2x2 matrix square root via special case formula
    # https://en.wikipedia.org/wiki/Square_root_of_a_matrix
    # "The Cholesky factorization provides another particular example of square root
    #  which should not be confused with the unique non-negative square root."
    # https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
    # M = (A 0; C D)
    # R = (sqrt(A) 0; C / (sqrt(A)+sqrt(D)) sqrt(D))

    a11 = ells[..., 2:3].abs().sqrt()
    a12 = torch.zeros_like(a11)
    a22 = ells[..., 4:5].abs().sqrt()
    a21 = ells[..., 3:4] / (a11 + a22).clamp(1e-9)

    A = torch.stack([a11, a12, a21, a22], dim=-1).view(B, N, 2, 2).inverse()

    out = torch.cat([A, ells[..., :2].view(B, N, 2, 1)], dim=3)

    return out


def laf_to_boundary_points(LAF, n_pts: int = 50):
    """
        Converts LAFs to boundary points of the regions + center.
        Used for local features visualization, see visualize_laf function
    """
    raise_error_if_laf_is_not_valid(LAF)

    B, N, _, _ = LAF.size()

    pts = torch.cat([torch.sin(torch.linspace(0, 2 * math.pi, n_pts - 1)).unsqueeze(-1),
                     torch.cos(torch.linspace(0, 2 * math.pi, n_pts - 1)).unsqueeze(-1),
                     torch.ones(n_pts - 1, 1)], dim=1)

    # Add origin to draw also the orientation
    pts = torch.cat([torch.tensor([0, 0, 1.]).view(1, 3), pts], dim=0).unsqueeze(0).expand(B * N, n_pts, 3)
    pts = pts.to(LAF.device).to(LAF.dtype)

    aux = torch.tensor([0, 0, 1.]).view(1, 1, 3).expand(B * N, 1, 3)

    HLAF = torch.cat([LAF.view(-1, 2, 3), aux.to(LAF.device).to(LAF.dtype)], dim=1)

    pts_h = torch.bmm(HLAF, pts.permute(0, 2, 1)).permute(0, 2, 1)

    return convert_points_from_homogeneous(pts_h.view(B, N, n_pts, 3))


def get_laf_pts_to_draw(LAF, img_idx=0):
    """
        Returns numpy array for drawing LAFs (local features).
    """

    raise_error_if_laf_is_not_valid(LAF)

    pts = laf_to_boundary_points(LAF[img_idx:img_idx + 1])[0]
    pts_np = pts.detach().permute(1, 0, 2).cpu().numpy()

    return (pts_np[..., 0], pts_np[..., 1])


def denormalize_laf(LAF, images):
    """
        De-normalizes LAFs from scale to image scale.
    """
    raise_error_if_laf_is_not_valid(LAF)

    n, ch, h, w = images.size()

    wf = float(w)
    hf = float(h)

    min_size = min(hf, wf)

    coef = torch.ones(1, 1, 2, 3).to(LAF.dtype).to(LAF.device) * min_size
    coef[0, 0, 0, 2] = wf
    coef[0, 0, 1, 2] = hf

    return coef.expand_as(LAF) * LAF


def normalize_laf(LAF, images):
    """
        Normalizes LAFs to [0,1] scale from pixel scale. See below:
    """

    raise_error_if_laf_is_not_valid(LAF)

    n, ch, h, w = images.size()

    wf = float(w)
    hf = float(h)

    min_size = min(hf, wf)

    coef = torch.ones(1, 1, 2, 3).to(LAF.dtype).to(LAF.device) / min_size
    coef[0, 0, 0, 2] = 1.0 / wf
    coef[0, 0, 1, 2] = 1.0 / hf

    return coef.expand_as(LAF) * LAF


def generate_patch_grid_from_normalized_LAF(img, LAF, PS=32):
    """
        Helper function for affine grid generation.
    """
    raise_error_if_laf_is_not_valid(LAF)

    B, N, _, _ = LAF.size()
    num, ch, h, w = img.size()

    # norm, then renorm is needed for allowing detection on one resolution
    # and extraction at arbitrary other
    LAF_renorm = denormalize_laf(LAF, img)

    grid = F.affine_grid(LAF_renorm.view(B * N, 2, 3), [B * N, ch, PS, PS], align_corners=False)

    grid[..., :, 0] = 2.0 * grid[..., :, 0].clone() / float(w) - 1.0
    grid[..., :, 1] = 2.0 * grid[..., :, 1].clone() / float(h) - 1.0

    return grid


def extract_patches_simple(img, laf, PS=32, normalize_lafs_before_extraction=True):
    """
        Extract patches defined by LAFs from image tensor.
        No smoothing applied, huge aliasing (better use extract_patches_from_pyramid)
    """
    raise_error_if_laf_is_not_valid(laf)

    if normalize_lafs_before_extraction:
        nlaf = normalize_laf(laf, img)

    else:
        nlaf = laf

    num, ch, h, w = img.size()

    B, N, _, _ = laf.size()

    out = []

    # for loop temporarily, to be refactored
    for i in range(B):
        grid = generate_patch_grid_from_normalized_LAF(img[i:i + 1], nlaf[i:i + 1], PS).to(img.device)

        out.append(F.grid_sample(img[i:i + 1].expand(grid.size(0), ch, h, w), grid,
                                 padding_mode="border",
                                 align_corners=False))

    return torch.cat(out, dim=0).view(B, N, ch, PS, PS)


def extract_patches_from_pyramid(img, laf, PS=32, normalize_lafs_before_extraction=True):
    """
        Extract patches defined by LAFs from image tensor.
        Patches are extracted from appropriate pyramid level
    """
    raise_error_if_laf_is_not_valid(laf)

    if normalize_lafs_before_extraction:
        nlaf = normalize_laf(laf, img)
    else:
        nlaf = laf

    B, N, _, _ = laf.size()

    num, ch, h, w = img.size()

    scale = 2.0 * get_laf_scale(denormalize_laf(nlaf, img)) / float(PS)

    pyr_idx = (scale.log2() + 0.5).relu().long()
    cur_img = img
    cur_pyr_level = int(0)

    out = torch.zeros(B, N, ch, PS, PS).to(nlaf.dtype).to(nlaf.device)

    while min(cur_img.size(2), cur_img.size(3)) >= PS:

        num, ch, h, w = cur_img.size()

        # for loop temporarily, to be refactored
        for i in range(B):

            scale_mask = (pyr_idx[i] == cur_pyr_level).bool().squeeze()

            if (scale_mask.float().sum()) == 0:
                continue

            scale_mask = scale_mask.bool().view(-1)

            grid = generate_patch_grid_from_normalized_LAF(cur_img[i:i + 1], nlaf[i:i + 1, scale_mask, :, :], PS)

            patches = F.grid_sample(cur_img[i:i + 1].expand(grid.size(0), ch, h, w), grid,
                                    padding_mode="border",
                                    align_corners=False)

            out[i].masked_scatter_(scale_mask.view(-1, 1, 1, 1), patches)

        cur_img = pyrdown(cur_img)

        cur_pyr_level += 1
    return out


def laf_is_inside_image(laf, images, border=0):
    """
        Checks if the LAF is touching or partly outside the image boundary. Returns the mask
        of LAFs, which are fully inside the image, i.e. valid.
    """

    raise_error_if_laf_is_not_valid(laf)

    n, ch, h, w = images.size()

    pts = laf_to_boundary_points(laf, 12)

    good_lafs_mask = (pts[..., 0] >= border) *\
        (pts[..., 0] <= w - border) *\
        (pts[..., 1] >= border) *\
        (pts[..., 1] <= h - border)

    good_lafs_mask = good_lafs_mask.min(dim=2)[0]

    return good_lafs_mask


def laf_to_three_points(laf):
    """
        Converts local affine frame(LAF) to alternative representation: coordinates of
        LAF center, LAF-x unit vector, LAF-y unit vector.
    """

    raise_error_if_laf_is_not_valid(laf)

    three_pts = torch.stack([laf[..., 2] + laf[..., 0],
                             laf[..., 2] + laf[..., 1],
                             laf[..., 2]],
                            dim=-1)

    return three_pts


def laf_from_three_points(threepts):
    """
        Converts three points to local affine frame. Order is (0,0), (0, 1), (1, 0).
    """
    laf = torch.stack([threepts[..., 0] - threepts[..., 2],
                       threepts[..., 1] - threepts[..., 2],
                       threepts[..., 2]],
                      dim=-1)

    return laf
