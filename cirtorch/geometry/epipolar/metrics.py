from ..conversions import convert_points_to_homogeneous


def sampson_epipolar_distance(pts1, pts2, Fm, squared=True, eps=1e-8):
    """
        Returns Sampson distance for correspondences given the fundamental matrix.
    """

    if (len(Fm.shape) != 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError(
            "Fm must be a (*, 3, 3) tensor. Got {}".format(
                Fm.shape))

    if pts1.size(-1) == 2:
        pts1 = convert_points_to_homogeneous(pts1)

    if pts2.size(-1) == 2:
        pts2 = convert_points_to_homogeneous(pts2)

    # From Hartley and Zisserman, Sampson error (11.9)
    # sam =  (x'^T F x) ** 2 / (  (((Fx)_1**2) + (Fx)_2**2)) +  (((F^Tx')_1**2) + (F^Tx')_2**2)) )

    # line1_in_2: torch.Tensor = (F @ pts1.permute(0,2,1)).permute(0,2,1)
    # line2_in_1: torch.Tensor = (F.permute(0,2,1) @ pts2.permute(0,2,1)).permute(0,2,1)
    # Instead we can just transpose F once and switch the order of multiplication

    F_t = Fm.permute(0, 2, 1)

    line1_in_2 = pts1 @ F_t
    line2_in_1 = pts2 @ Fm

    # numerator = (x'^T F x) ** 2
    numerator = (pts2 * line1_in_2).sum(2).pow(2)

    # denominator = (((Fx)_1**2) + (Fx)_2**2)) +  (((F^Tx')_1**2) + (F^Tx')_2**2))
    denominator = line1_in_2[..., :2].norm(2, dim=2).pow(2) + line2_in_1[..., :2].norm(2, dim=2).pow(2)

    out = numerator / denominator

    if squared:
        return out

    return (out + eps).sqrt()


def symmetrical_epipolar_distance(pts1, pts2, Fm, squared=True, eps=1e-8):
    """
        Returns symmetrical epipolar distance for correspondences given the fundamental matrix.
    """

    if (len(Fm.shape) != 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError(
            "Fm must be a (*, 3, 3) tensor. Got {}".format(
                Fm.shape))

    if pts1.size(-1) == 2:
        pts1 = convert_points_to_homogeneous(pts1)

    if pts2.size(-1) == 2:
        pts2 = convert_points_to_homogeneous(pts2)

    # From Hartley and Zisserman, symmetric epipolar distance (11.10)
    # sed = (x'^T F x) ** 2 /  (((Fx)_1**2) + (Fx)_2**2)) +  1/ (((F^Tx')_1**2) + (F^Tx')_2**2))

    # line1_in_2: torch.Tensor = (F @ pts1.permute(0,2,1)).permute(0,2,1)
    # line2_in_1: torch.Tensor = (F.permute(0,2,1) @ pts2.permute(0,2,1)).permute(0,2,1)

    # Instead we can just transpose F once and switch the order of multiplication
    F_t = Fm.permute(0, 2, 1)
    line1_in_2 = pts1 @ F_t
    line2_in_1 = pts2 @ Fm

    # numerator = (x'^T F x) ** 2
    numerator = (pts2 * line1_in_2).sum(2).pow(2)

    # denominator_inv =  1/ (((Fx)_1**2) + (Fx)_2**2)) +  1/ (((F^Tx')_1**2) + (F^Tx')_2**2))
    denominator_inv = (1. / (line1_in_2[..., :2].norm(2, dim=2).pow(2)) + 1. / (line2_in_1[..., :2].norm(2, dim=2).pow(2)))
    out = numerator * denominator_inv

    if squared:
        return out

    return (out + eps).sqrt()
