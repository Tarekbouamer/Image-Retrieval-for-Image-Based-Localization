import torch


def create_meshgrid(height, width, normalized_coordinates=True, device=torch.device('cpu')):
    """
        Generates a coordinate grid for an image.
        When the flag `normalized_coordinates` is set to True, the grid is
        normalized to be in the range [-1,1] to be consistent with the pytorch
        function grid_sample.
    """

    xs = torch.linspace(0, width - 1, width, device=device, dtype=torch.float)
    ys = torch.linspace(0, height - 1, height, device=device, dtype=torch.float)

    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2

    # generate grid by stacking coordinates
    base_grid = torch.stack(torch.meshgrid([xs, ys])).transpose(1, 2)  # 2xHxW

    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2


def create_meshgrid3d(depth, height, width, normalized_coordinates=True, device=torch.device('cpu')):
    """
        Generates a coordinate grid for an image.
        When the flag `normalized_coordinates` is set to True, the grid is
        normalized to be in the range [-1,1] to be consistent with the pytorch
        function grid_sample.
        http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample

    """
    xs = torch.linspace(0, width - 1, width, device=device, dtype=torch.float)
    ys = torch.linspace(0, height - 1, height, device=device, dtype=torch.float)
    zs = torch.linspace(0, depth - 1, depth, device=device, dtype=torch.float)

    # Fix TracerWarning
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
        zs = (ys / (height - 1) - 0.5) * 2

    # generate grid by stacking coordinates
    base_grid = torch.stack(torch.meshgrid([zs, xs, ys])).transpose(1, 2)  # 3xHxW

    return base_grid.unsqueeze(0).permute(0, 3, 4, 2, 1)  # 1xHxWx3