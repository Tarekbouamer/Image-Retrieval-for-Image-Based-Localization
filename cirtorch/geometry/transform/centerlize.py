import torch

from ..conversions import convert_affinematrix_to_homography


def centralize(sizes, max_size):
    """
        Applies transformation to center image FOE on Tile FOE.
    """
    
    # get transformation matrix :: translation 
    coef_size = (max_size - sizes)/2

    transform = [torch.tensor([1, 0, coef[1], 0, 1, coef[0]]).view(2, 3) for coef in coef_size]
    transform = torch.stack(transform, dim=0).to(device=max_size.device, dtype=max_size.dtype)

    # pad transform to get Bx3x3
    transform = convert_affinematrix_to_homography(transform)
    
    return transform