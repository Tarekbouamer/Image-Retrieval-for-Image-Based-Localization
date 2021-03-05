import torch.nn as nn


from .kernels import (
    get_motion_kernel2d,
    get_motion_kernel3d
)

from .filter import filter2D, filter3D


class MotionBlur(nn.Module):
    """
        Blur 2D images (4D tensor) using the motion filter.
    """

    def __init__(self, kernel_size, angle, direction, border_type='constant'):
        super(MotionBlur, self).__init__()

        self.kernel_size = kernel_size
        self.angle = angle
        self.direction = direction
        self.border_type = border_type

    def forward(self, x):
        return motion_blur(x, self.kernel_size, self.angle, self.direction, self.border_type)


class MotionBlur3D(nn.Module):
    """
        Blur 3D volumes (5D tensor) using the motion filter.
    """

    def __init__(self, kernel_size, angle, direction, border_type='constant') -> None:
        super(MotionBlur3D, self).__init__()
        self.kernel_size = kernel_size

        if isinstance(angle, float):
            self.angle = (angle, angle, angle)
        elif isinstance(angle, (tuple, list)) and len(angle) == 3:
            self.angle = angle
        else:
            raise ValueError(f"Expect angle to be either a float or a tuple of floats. Got {angle}.")

        self.direction = direction
        self.border_type = border_type

    def forward(self, x):
        return motion_blur3d(x, self.kernel_size, self.angle, self.direction, self.border_type)


def motion_blur(inp, kernel_size, angle, direction, border_type='constant', mode='nearest'):
    """
        Perform motion blur on 2D images (4D tensor).
    """
    assert border_type in ["constant", "reflect", "replicate", "circular"]
    
    kernel = get_motion_kernel2d(kernel_size, angle, direction, mode)

    return filter2D(inp, kernel, border_type)


def motion_blur3d(inp, kernel_size, angle, direction, border_type='constant', mode='nearest'):
    """
        Perform motion blur on 3D volumes (5D tensor).
    """
    assert border_type in ["constant", "reflect", "replicate", "circular"]

    kernel = get_motion_kernel3d(kernel_size, angle, direction, mode)

    return filter3D(inp, kernel, border_type)