import math

import torch
import torch.nn as nn


def rgb_to_hsv(image, eps=1e-6):
    """
        Convert an image from RGB to HSV.
        The image data is assumed to be in the range of (0, 1).
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    maxc, _ = image.max(-3)
    maxc_mask = image == maxc.unsqueeze(-3)
    _, max_indices = ((maxc_mask.cumsum(-3) == 1) & maxc_mask).max(-3)
    minc = image.min(-3)[0]

    v = maxc  # brightness

    deltac = maxc - minc
    s = deltac / (v + eps)

    # avoid division by zero
    deltac = torch.where(deltac == 0, torch.ones_like(deltac, device=deltac.device, dtype=deltac.dtype), deltac)

    maxc_tmp = maxc.unsqueeze(-3) - image
    rc = maxc_tmp[..., 0, :, :]
    gc = maxc_tmp[..., 1, :, :]
    bc = maxc_tmp[..., 2, :, :]

    h = torch.stack([
        bc - gc,
        2.0 * deltac + rc - bc,
        4.0 * deltac + gc - rc,
    ], dim=-3)

    h = torch.gather(h, dim=-3, index=max_indices[..., None, :, :])
    h = h.squeeze(-3)
    h = h / deltac

    h = (h / 6.0) % 1.0

    h = 2 * math.pi * h

    return torch.stack([h, s, v], dim=-3)


def hsv_to_rgb(image):
    """
        Convert an image from HSV to RGB.
        The image data is assumed to be in the range of (0, 1).
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    h = image[..., 0, :, :] / (2 * math.pi)
    s = image[..., 1, :, :]
    v = image[..., 2, :, :]

    hi = torch.floor(h * 6) % 6
    f = ((h * 6) % 6) - hi

    one = torch.tensor(1.).to(image.device)
    p = v * (one - s)
    q = v * (one - f * s)
    t = v * (one - (one - f) * s)

    hi = hi.long()

    indices = torch.stack([hi, hi + 6, hi + 12], dim=-3)

    out = torch.stack((
        v, q, p, p, t, v,
        t, v, v, q, p, p,
        p, p, t, v, v, q,
    ), dim=-3)

    out = torch.gather(out, -3, indices)

    return out