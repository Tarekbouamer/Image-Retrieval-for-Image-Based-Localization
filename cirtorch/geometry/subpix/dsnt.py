import torch
import torch.nn.functional as F

from cirtorch.utils.grid import create_meshgrid


def _validate_batched_image_tensor_input(tensor):
    if not len(tensor.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(tensor.shape))


def spatial_softmax2d(input, temperature=torch.tensor(1.0)):
    """
        Applies the Softmax function over features in each image channel.
        Note that this function behaves differently to `Softmax2d`, which
        instead applies Softmax over features at each spatial location.
        Returns a 2D probability distribution per image channel.
    """
    _validate_batched_image_tensor_input(input)

    batch_size, channels, height, width = input.shape
    temperature = temperature.to(device=input.device, dtype=input.dtype)

    x = input.view(batch_size, channels, -1)
    x_soft = F.softmax(x * temperature, dim=-1)

    return x_soft.view(batch_size, channels, height, width)


def spatial_expectation2d(input, normalized_coordinates=True):
    """
        Computes the expectation of coordinate values using spatial probabilities.
        The input heatmap is assumed to represent a valid spatial probability
        distribution, which can be achieved using `dsnt.spatial_softmax2d`.
        Returns the expected value of the 2D coordinates.
        The output order of the coordinates is (x, y).
    """
    _validate_batched_image_tensor_input(input)

    batch_size, channels, height, width = input.shape

    # Create coordinates grid.
    grid = create_meshgrid(height, width, normalized_coordinates, input.device)
    grid = grid.to(input.dtype)

    pos_x = grid[..., 0].reshape(-1)
    pos_y = grid[..., 1].reshape(-1)

    input_flat = input.view(batch_size, channels, -1)

    # Compute the expectation of the coordinates.
    expected_y = torch.sum(pos_y * input_flat, -1, keepdim=True)
    expected_x = torch.sum(pos_x * input_flat, -1, keepdim=True)

    output = torch.cat([expected_x, expected_y], -1)

    return output.view(batch_size, channels, 2)  # BxNx2


def _safe_zero_division(numerator, denominator, eps=1e-32):
    return numerator / torch.clamp(denominator, min=eps)


def render_gaussian2d( mean, std, size, normalized_coordinates=True):
    """
        Renders the PDF of a 2D Gaussian distribution.
    """

    if not (std.dtype == mean.dtype and std.device == mean.device):
        raise TypeError("Expected inputs to have the same dtype and device")

    height, width = size

    # Create coordinates grid.
    grid = create_meshgrid(height, width, normalized_coordinates, mean.device)
    grid = grid.to(mean.dtype)

    pos_x = grid[..., 0].view(height, width)
    pos_y = grid[..., 1].view(height, width)

    # dists <- (x - \mu)^2
    dist_x = (pos_x - mean[..., 0, None, None]) ** 2
    dist_y = (pos_y - mean[..., 1, None, None]) ** 2

    # ks <- -1 / (2 \sigma^2)
    k_x = -0.5 * torch.reciprocal(std[..., 0, None, None])
    k_y = -0.5 * torch.reciprocal(std[..., 1, None, None])

    # Assemble the 2D Gaussian.
    exps_x = torch.exp(dist_x * k_x)
    exps_y = torch.exp(dist_y * k_y)
    gauss = exps_x * exps_y

    # Rescale so that values sum to one.
    val_sum = gauss.sum(-2, keepdim=True).sum(-1, keepdim=True)
    gauss = _safe_zero_division(gauss, val_sum)

    return gauss