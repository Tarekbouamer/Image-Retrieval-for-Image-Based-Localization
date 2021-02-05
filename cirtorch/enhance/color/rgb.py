import torch
import torch.nn as nn


def rgb_to_bgr(image):
    """
        Convert a RGB image to BGR.
    """
    return bgr_to_rgb(image)


def bgr_to_rgb(image):
    """
        Convert a BGR image to RGB.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W).Got {}"
                         .format(image.shape))

    # flip image channels
    out = image.flip(-3)
    return out


def rgb_to_rgba(image, alpha_val):
    """
        Convert an image from RGB to RGBA.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")

    if not isinstance(alpha_val, (float, torch.Tensor)):
        raise TypeError(f"alpha_val type is not a float or torch.Tensor. Got {type(alpha_val)}")

    # add one channel
    r, g, b = torch.chunk(image, image.shape[-3], dim=-3)

    if isinstance(alpha_val, float):
        a = torch.full_like(r, fill_value=float(alpha_val))

    return torch.cat([r, g, b, a], dim=-3)


def bgr_to_rgba(image, alpha_val):
    """
        Convert an image from BGR to RGBA.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")

    if not isinstance(alpha_val, (float, torch.Tensor)):
        raise TypeError(f"alpha_val type is not a float or torch.Tensor. Got {type(alpha_val)}")

    # convert first to RGB, then add alpha channel
    x_rgb = bgr_to_rgb(image)
    return rgb_to_rgba(x_rgb, alpha_val)


def rgba_to_rgb(image):
    """
        Convert an image from RGBA to RGB.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 4:
        raise ValueError(f"Input size must have a shape of (*, 4, H, W).Got {image.shape}")

    # unpack channels
    r, g, b, a = torch.chunk(image, image.shape[-3], dim=-3)

    # compute new channels
    a_one = torch.tensor(1.) - a
    r_new = a_one * r + a * r
    g_new = a_one * g + a * g
    b_new = a_one * b + a * b

    return torch.cat([r, g, b], dim=-3)


def rgba_to_bgr(image):
    """
        Convert an image from RGBA to BGR.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 4:
        raise ValueError(f"Input size must have a shape of (*, 4, H, W).Got {image.shape}")

    # convert to RGB first, then to BGR
    x_rgb = rgba_to_rgb(image)
    return rgb_to_bgr(x_rgb)