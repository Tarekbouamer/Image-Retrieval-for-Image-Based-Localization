import torch


def rot180(input):
    """
        Rotate a tensor image or a batch of tensor images 180 degrees
    """

    return torch.flip(input, [-2, -1])


def hflip(input):
    """
        Horizontally flip a tensor image or a batch of tensor images.
    """
    w = input.shape[-1]
    return input[..., torch.arange(w - 1, -1, -1, device=input.device)]


def vflip(input):
    """
        Vertically flip a tensor image or a batch of tensor images.
    """

    h = input.shape[-2]
    return input[..., torch.arange(h - 1, -1, -1, device=input.device), :]