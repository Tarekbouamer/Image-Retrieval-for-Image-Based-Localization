import torch

from ..conversions import convert_affinematrix_to_homography
from ..transform.affwarp import warp_affine


def centralize(inp, valid_size):
    """
        Applies transformation to a tensor image to center image FOE.
        The transformation is computed so that the image center is kept invariant.
    """
    # TODO: double check this part
    # note input valid size is different than shape
    valid_size = [torch.tensor([item[0], item[1]]) for item in valid_size]

    valid_size = torch.stack(valid_size).to(device=inp.device, dtype=inp.dtype)

    max_size = inp.shape[-2:]
    max_size = torch.tensor([max_size[0], max_size[1]]).float().to(device=inp.device, dtype=inp.dtype)

    coef_size = max_size/2 - valid_size/2

    translation = [torch.tensor([1, 0, coef[1], 0, 1, coef[0]]).view(2, 3) for coef in coef_size]
    translation = torch.stack(translation, dim=0)

    # pad transform to get Bx3x3
    transform = convert_affinematrix_to_homography(translation).to(device=inp.device, dtype=inp.dtype)

    size = inp.shape[-2:]
    inp = warp_affine(inp, transform, size)

    bbx = [torch.tensor([coef[0], coef[1], size[0], size[1]]) for coef, size in zip(coef_size, valid_size)]
    bbx = torch.stack(bbx).to(device=inp.device, dtype=inp.dtype)

    return inp, bbx