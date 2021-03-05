import numpy as np
import torch

__all__ = [
    "image_to_tensor",
    "tensor_to_image",
    "normalize",
    "normalize_min_max",
    "denormalize",
]


def image_to_tensor(image, keepdim=True):
    """
        Converts a numpy image to a PyTorch 4d tensor image.
    """
    if not isinstance(image, (np.ndarray,)):
        raise TypeError("Input type must be a numpy.ndarray. Got {}".format(
            type(image)))

    if len(image.shape) > 4 or len(image.shape) < 2:
        raise ValueError(
            "Input size must be a two, three or four dimensional array")

    input_shape = image.shape
    tensor = torch.from_numpy(image)

    if len(input_shape) == 2:
        # (H, W) -> (1, H, W)
        tensor = tensor.unsqueeze(0)
    elif len(input_shape) == 3:
        # (H, W, C) -> (C, H, W)
        tensor = tensor.permute(2, 0, 1)
    elif len(input_shape) == 4:
        # (B, H, W, C) -> (B, C, H, W)
        tensor = tensor.permute(0, 3, 1, 2)
        keepdim = True  # no need to unsqueeze
    else:
        raise ValueError(
            "Cannot process image with shape {}".format(input_shape))

    return tensor.unsqueeze(0) if not keepdim else tensor


def tensor_to_image(tensor):
    """
        Converts a PyTorch tensor image to a numpy image.
        In case the tensor is in the GPU, it will be copied back to CPU.

    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(tensor)))

    if len(tensor.shape) > 4 or len(tensor.shape) < 2:
        raise ValueError(
            "Input size must be a two, three or four dimensional tensor")

    input_shape = tensor.shape
    image = tensor.cpu().detach().numpy()

    if len(input_shape) == 2:
        # (H, W) -> (H, W)
        image = image
    elif len(input_shape) == 3:
        # (C, H, W) -> (H, W, C)
        if input_shape[0] == 1:
            # Grayscale for proper plt.imshow needs to be (H,W)
            image = image.squeeze()
        else:
            image = image.transpose(1, 2, 0)
    elif len(input_shape) == 4:
        # (B, C, H, W) -> (B, H, W, C)
        image = image.transpose(0, 2, 3, 1)
        if input_shape[0] == 1:
            image = image.squeeze(0)
        if input_shape[1] == 1:
            image = image.squeeze(-1)
    else:
        raise ValueError(
            "Cannot process tensor with shape {}".format(input_shape))

    return image


def normalize(data, mean, std):
    """
        Normalize a tensor image with mean and standard deviation.
    """
    shape = data.shape

    if isinstance(mean, list):
        mean = torch.tensor(mean, device=data.device, dtype=data.dtype)

    if isinstance(std, list):
        std = torch.tensor(std , device=data.device, dtype=data.dtype)

    if not isinstance(data, torch.Tensor):
        raise TypeError("data should be a tensor. Got {}".format(type(data)))

    if not isinstance(mean, torch.Tensor):
        raise TypeError("mean should be a tensor or a float. Got {}".format(type(mean)))

    if not isinstance(std, torch.Tensor):
        raise TypeError("std should be a tensor or float. Got {}".format(type(std)))

    # Allow broadcast on channel dimension
    if mean.shape and mean.shape[0] != 1:
        if mean.shape[0] != data.shape[-3] and mean.shape[:2] != data.shape[:2]:
            raise ValueError(f"mean length and number of channels do not match. Got {mean.shape} and {data.shape}.")

    # Allow broadcast on channel dimension
    if std.shape and std.shape[0] != 1:
        if std.shape[0] != data.shape[-3] and std.shape[:2] != data.shape[:2]:
            raise ValueError(f"std length and number of channels do not match. Got {std.shape} and {data.shape}.")

    mean = torch.as_tensor(mean, device=data.device, dtype=data.dtype)
    std = torch.as_tensor(std, device=data.device, dtype=data.dtype)

    if mean.shape:
        mean = mean[..., :, None]
    if std.shape:
        std = std[..., :, None]

    out = (data.view(shape[0], shape[1], -1) - mean) / std

    return out.view(shape)


def denormalize(data, mean, std):
    """
        Denormalize a tensor image with mean and standard deviation.
    """
    shape = data.shape

    if isinstance(mean, float):
        mean = torch.tensor([mean] * shape[1], device=data.device, dtype=data.dtype)

    if isinstance(std, float):
        std = torch.tensor([std] * shape[1], device=data.device, dtype=data.dtype)

    if not isinstance(data, torch.Tensor):
        raise TypeError("data should be a tensor. Got {}".format(type(data)))

    if not isinstance(mean, torch.Tensor):
        raise TypeError("mean should be a tensor or a float. Got {}".format(type(mean)))

    if not isinstance(std, torch.Tensor):
        raise TypeError("std should be a tensor or float. Got {}".format(type(std)))

    # Allow broadcast on channel dimension
    if mean.shape and mean.shape[0] != 1:
        if mean.shape[0] != data.shape[-3] and mean.shape[:2] != data.shape[:2]:
            raise ValueError(f"mean length and number of channels do not match. Got {mean.shape} and {data.shape}.")

    # Allow broadcast on channel dimension
    if std.shape and std.shape[0] != 1:
        if std.shape[0] != data.shape[-3] and std.shape[:2] != data.shape[:2]:
            raise ValueError(f"std length and number of channels do not match. Got {std.shape} and {data.shape}.")

    mean = torch.as_tensor(mean, device=data.device, dtype=data.dtype)
    std = torch.as_tensor(std, device=data.device, dtype=data.dtype)

    if mean.shape:
        mean = mean[..., :, None]
    if std.shape:
        std = std[..., :, None]

    out = (data.view(shape[0], shape[1], -1) * std) + mean

    return out.view(shape)


def normalize_min_max(x, min_val=0., max_val=1., eps=1e-6):
    """
        Normalise an image tensor by Min Max and re-scales the value between a range.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"data should be a tensor. Got: {type(x)}.")

    if not isinstance(min_val, float):
        raise TypeError(f"'min_val' should be a float. Got: {type(min_val)}.")

    if not isinstance(max_val, float):
        raise TypeError(f"'b' should be a float. Got: {type(max_val)}.")

    if len(x.shape) != 4:
        raise ValueError(f"Input shape must be a 4d tensor. Got: {x.shape}.")

    B, C, H, W = x.shape

    x_min = x.view(B, C, -1).min(-1)[0].view(B, C, 1, 1)
    x_max = x.view(B, C, -1).max(-1)[0].view(B, C, 1, 1)

    x_out = ((max_val - min_val) * (x - x_min) / (x_max - x_min + eps) + min_val)

    return x_out.expand_as(x), x_min, x_max


def denormalize_min_max(x, x_min, x_max, eps=1e-6):
    """
        Normalise an image tensor by MinMax and re-scales the value between a range.
    """

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"data should be a tensor. Got: {type(x)}.")

    if not isinstance(x_min, torch.Tensor):
        raise TypeError(f"data should be a tensor. Got: {type(x)}.")

    if not isinstance(x_max, torch.Tensor):
        raise TypeError(f"data should be a tensor. Got: {type(x)}.")

    if len(x.shape) != 4:
        raise ValueError(f"Input shape must be a 4d tensor. Got: {x.shape}.")

    x_out = (x_max - x_min) * x + x_min

    return x_out