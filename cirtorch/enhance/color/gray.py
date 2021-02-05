import torch
import torch.nn as nn

from .rgb import bgr_to_rgb


def rgb_to_grayscale(image: torch.Tensor) -> torch.Tensor:
    """
        Convert a RGB image to grayscale version of image.
        The image data is assumed to be in the range of (0, 1).
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r = image[..., 0:1, :, :]
    g = image[..., 1:2, :, :]
    b = image[..., 2:3, :, :]

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray


def bgr_to_grayscale(image: torch.Tensor) -> torch.Tensor:
    """
        Convert a BGR image to grayscale.
        The image data is assumed to be in the range of (0, 1). First flips to RGB, then converts.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    image_rgb = bgr_to_rgb(image)
    gray = rgb_to_grayscale(image_rgb)
    return gray