import torch
from math import sqrt

from ..geometry.transform.affwarp import rotate, rotate3d
from cirtorch.utils.helper import _extract_device_dtype


def normalize_kernel2d(input):
    """
        Normalizes both derivative and smoothing kernel.
    """
    if len(input.size()) < 2:
        raise TypeError("input should be at least 2D tensor. Got {}"
                        .format(input.size()))

    norm = input.abs().sum(dim=-1).sum(dim=-1)

    return input / (norm.unsqueeze(-1).unsqueeze(-1))


def gaussian(window_size, sigma):

    x = torch.arange(window_size) - window_size // 2

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp((-x.pow(2.0) / (2 * sigma ** 2)))

    return gauss / gauss.sum()


def gaussian_discrete_erf(window_size, sigma):
    """
        Discrete Gaussian by interpolating the error function. Adapted from:
        https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """

    device = sigma.device if isinstance(sigma, torch.Tensor) else None

    sigma = torch.as_tensor(sigma, dtype=torch.float, device=device)

    x = torch.arange(window_size).float() - window_size // 2

    t = 0.70710678 / torch.abs(sigma)

    gauss = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())

    gauss = gauss.clamp(min=0)

    return gauss / gauss.sum()


def _modified_bessel_0(x):
    """
        from: https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """
    if torch.abs(x) < 3.75:
        y = (x / 3.75) * (x / 3.75)

        return 1.0 + y * (
            3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2))))
        )

    ax = torch.abs(x)
    y = 3.75 / ax

    ans = 0.916281e-2 + y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1 + y * 0.392377e-2)))

    return (torch.exp(ax) / torch.sqrt(ax)) * (
        0.39894228 + y * (0.1328592e-1 + y * (0.225319e-2 + y * (-0.157565e-2 + y * ans)))
    )


def _modified_bessel_1(x):
    """
        from: https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """

    if torch.abs(x) < 3.75:

        y = (x / 3.75) * (x / 3.75)

        ans = 0.51498869 + y * (0.15084934 + y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3)))

        return torch.abs(x) * (0.5 + y * (0.87890594 + y * ans))

    ax = torch.abs(x)

    y = 3.75 / ax

    ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1 - y * 0.420059e-2))

    ans = 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2 + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))))

    ans = ans * torch.exp(ax) / torch.sqrt(ax)

    return -ans if x < 0.0 else ans


def _modified_bessel_i(n, x):
    """
        from: https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """

    if n < 2:
        raise ValueError("n must be greater than 1.")
    if x == 0.0:
        return x

    device = x.device
    tox = 2.0 / torch.abs(x)
    ans = torch.tensor(0.0, device=device)
    bip = torch.tensor(0.0, device=device)
    bi = torch.tensor(1.0, device=device)

    m = int(2 * (n + int(sqrt(40.0 * n))))

    for j in range(m, 0, -1):
        bim = bip + float(j) * tox * bi
        bip = bi
        bi = bim

        if abs(bi) > 1.0e10:
            ans = ans * 1.0e-10
            bi = bi * 1.0e-10
            bip = bip * 1.0e-10

        if j == n:
            ans = bip

    ans = ans * _modified_bessel_0(x) / bi

    return -ans if x < 0.0 and (n % 2) == 1 else ans


def gaussian_discrete(window_size, sigma):
    """
        Discrete Gaussian kernel based on the modified Bessel functions.
        from: https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """
    device = sigma.device if isinstance(sigma, torch.Tensor) else None

    sigma = torch.as_tensor(sigma, dtype=torch.float, device=device)
    sigma2 = sigma * sigma

    tail = int(window_size // 2)

    out_pos = [None] * (tail + 1)
    out_pos[0] = _modified_bessel_0(sigma2)
    out_pos[1] = _modified_bessel_1(sigma2)

    for k in range(2, len(out_pos)):
        out_pos[k] = _modified_bessel_i(k, sigma2)

    out = out_pos[:0:-1]
    out.extend(out_pos)
    out = torch.stack(out) * torch.exp(sigma2)

    return out / out.sum()


def laplacian_1d(window_size):
    """
        One could also use the Laplacian of Gaussian formula to design the filter.
    """

    filter_1d = torch.ones(window_size)
    filter_1d[window_size // 2] = 1 - window_size
    laplacian_1d = filter_1d
    return laplacian_1d


def get_box_kernel2d(kernel_size):
    """
        Utility function that returns a box filter.
    """
    kx = float(kernel_size[0])
    ky = float(kernel_size[1])

    scale = torch.tensor(1.) / torch.tensor([kx * ky])

    tmp_kernel = torch.ones(1, kernel_size[0], kernel_size[1])

    return scale.to(tmp_kernel.dtype) * tmp_kernel


def get_binary_kernel2d(window_size):
    """
        Creates a binary kernel to extract the patches. If the window size
        is HxW will create a (H*W)xHxW kernel.
    """
    window_range = window_size[0] * window_size[1]

    kernel = torch.zeros(window_range, window_range)

    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])


def get_sobel_kernel_3x3():
    """
        Utility function that returns a sobel kernel of 3x3
    """
    return torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.],
    ])


def get_sobel_kernel_5x5_2nd_order():
    """
        Utility function that returns a 2nd order sobel kernel of 5x5
    """
    return torch.tensor([
        [-1., 0., 2., 0., -1.],
        [-4., 0., 8., 0., -4.],
        [-6., 0., 12., 0., -6.],
        [-4., 0., 8., 0., -4.],
        [-1., 0., 2., 0., -1.]
    ])


def _get_sobel_kernel_5x5_2nd_order_xy():
    """
        Utility function that returns a 2nd order sobel kernel of 5x5
    """
    return torch.tensor([
        [-1., -2., 0., 2., 1.],
        [-2., -4., 0., 4., 2.],
        [0., 0., 0., 0., 0.],
        [2., 4., 0., -4., -2.],
        [1., 2., 0., -2., -1.]
    ])


def get_diff_kernel_3x3():
    """
        Utility function that returns a sobel kernel of 3x3
    """
    return torch.tensor([
        [-0., 0., 0.],
        [-1., 0., 1.],
        [-0., 0., 0.],
    ])


def get_diff_kernel3d(device=torch.device('cpu'), dtype=torch.float):
    """
        Utility function that returns a first order derivative kernel of 3x3x3
    """
    kernel = torch.tensor([[[[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]],

                            [[0.0, 0.0, 0.0],
                             [-0.5, 0.0, 0.5],
                             [0.0, 0.0, 0.0]],

                            [[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]], ],

                           [[[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]],

                            [[0.0, -0.5, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.5, 0.0]],

                            [[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]], ],

                           [[[0.0, 0.0, 0.0],
                             [0.0, -0.5, 0.0],
                             [0.0, 0.0, 0.0]],

                            [[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]],

                            [[0.0, 0.0, 0.0],
                             [0.0, 0.5, 0.0],
                             [0.0, 0.0, 0.0]], ], ], device=device, dtype=dtype)

    return kernel.unsqueeze(1)


def get_diff_kernel3d_2nd_order(device=torch.device('cpu'), dtype=torch.float):
    """
        Utility function that returns a first order derivative kernel of 3x3x3
    """
    kernel = torch.tensor([[[[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]],

                            [[0.0, 0.0, 0.0],
                             [1.0, -2.0, 1.0],
                             [0.0, 0.0, 0.0]],

                            [[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]], ],

                           [[[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]],

                            [[0.0, 1.0, 0.0],
                             [0.0, -2.0, 0.0],
                             [0.0, 1.0, 0.0]],

                            [[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]], ],

                           [[[0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0]],

                            [[0.0, 0.0, 0.0],
                             [0.0, -2.0, 0.0],
                             [0.0, 0.0, 0.0]],

                            [[0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0]], ],

                           [[[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]],

                            [[1.0, 0.0, -1.0],
                             [0.0, 0.0, 0.0],
                             [-1.0, 0.0, 1.0]],

                            [[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]], ],

                           [[[0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, -1.0, 0.0]],

                            [[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]],

                            [[0.0, -1.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0]], ],

                           [[[0.0, 0.0, 0.0],
                             [1.0, 0.0, -1.0],
                             [0.0, 0.0, 0.0]],

                            [[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]],

                            [[0.0, 0.0, 0.0],
                             [-1.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0]], ], ], device=device, dtype=dtype)

    return kernel.unsqueeze(1)


def get_sobel_kernel2d():

    kernel_x = get_sobel_kernel_3x3()
    kernel_y = kernel_x.transpose(0, 1)

    return torch.stack([kernel_x, kernel_y])


def get_diff_kernel2d():

    kernel_x = get_diff_kernel_3x3()
    kernel_y = kernel_x.transpose(0, 1)

    return torch.stack([kernel_x, kernel_y])


def get_sobel_kernel2d_2nd_order():

    gxx = get_sobel_kernel_5x5_2nd_order()
    gyy = gxx.transpose(0, 1)
    gxy = _get_sobel_kernel_5x5_2nd_order_xy()

    return torch.stack([gxx, gxy, gyy])


def get_diff_kernel2d_2nd_order():

    gxx = torch.tensor([
        [0., 0., 0.],
        [1., -2., 1.],
        [0., 0., 0.],
    ])

    gyy = gxx.transpose(0, 1)

    gxy = torch.tensor([
        [-1., 0., 1.],
        [0., 0., 0.],
        [1., 0., -1.],
    ])

    return torch.stack([gxx, gxy, gyy])


def get_spatial_gradient_kernel2d(mode, order):
    """
        Function that returns kernel for 1st or 2nd order image gradients,
        using one of the following operators: sobel, diff"""

    if mode not in ['sobel', 'diff']:
        raise TypeError("mode should be either sobel\
                         or diff. Got {}".format(mode))

    if order not in [1, 2]:
        raise TypeError("order should be either 1 or 2\
                         Got {}".format(order))

    if mode == 'sobel' and order == 1:
        kernel = get_sobel_kernel2d()

    elif mode == 'sobel' and order == 2:
        kernel = get_sobel_kernel2d_2nd_order()

    elif mode == 'diff' and order == 1:
        kernel = get_diff_kernel2d()

    elif mode == 'diff' and order == 2:
        kernel = get_diff_kernel2d_2nd_order()

    else:
        raise NotImplementedError("")

    return kernel


def get_spatial_gradient_kernel3d(mode, order, device=torch.device('cpu'), dtype=torch.float):
    """
        Function that returns kernel for 1st or 2nd order scale pyramid gradients,
        using one of the following operators: sobel, diff
    """

    if mode not in ['sobel', 'diff']:
        raise TypeError("mode should be either sobel\
                         or diff. Got {}".format(mode))

    if order not in [1, 2]:
        raise TypeError("order should be either 1 or 2\
                         Got {}".format(order))

    if mode == 'sobel':
        raise NotImplementedError("Sobel kernel for 3d gradient is not implemented yet")

    elif mode == 'diff' and order == 1:
        kernel = get_diff_kernel3d(device, dtype)

    elif mode == 'diff' and order == 2:
        kernel = get_diff_kernel3d_2nd_order(device, dtype)

    else:
        raise NotImplementedError("")

    return kernel


def get_gaussian_kernel1d(kernel_size, sigma, force_even=False):
    """
        Function that returns Gaussian filter coefficients.
    """
    if (not isinstance(kernel_size, int) or ((kernel_size % 2 == 0) and not force_even) or (kernel_size <= 0)):
        raise TypeError(
            "kernel_size must be an odd positive integer. "
            "Got {}".format(kernel_size))

    window_1d = gaussian(kernel_size, sigma)

    return window_1d


def get_gaussian_discrete_kernel1d(kernel_size, sigma, force_even=False):
    """
        Function that returns Gaussian filter coefficients based on the modified Bessel functions.
        from:https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """
    if (not isinstance(kernel_size, int) or ((kernel_size % 2 == 0) and not force_even) or (kernel_size <= 0)):
        raise TypeError(
            "kernel_size must be an odd positive integer. "
            "Got {}".format(kernel_size)
        )

    window_1d = gaussian_discrete(kernel_size, sigma)

    return window_1d


def get_gaussian_erf_kernel1d(kernel_size, sigma, force_even=False):
    """
        Function that returns Gaussian filter coefficients by interpolating the error fucntion,
        from: https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/layers/convutils.py
    """
    if (not isinstance(kernel_size, int) or (
            (kernel_size % 2 == 0) and not force_even) or (
            kernel_size <= 0)):
        raise TypeError(
            "kernel_size must be an odd positive integer. "
            "Got {}".format(kernel_size)
        )

    window_1d = gaussian_discrete_erf(kernel_size, sigma)

    return window_1d


def get_gaussian_kernel2d(kernel_size, sigma, force_even=False):
    """
        Function that returns Gaussian filter matrix coefficients.
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(
            "kernel_size must be a tuple of length two. Got {}".format(
                kernel_size
            )
        )
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(
            "sigma must be a tuple of length two. Got {}".format(sigma)
        )
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma

    kernel_x = get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
    kernel_y = get_gaussian_kernel1d(ksize_y, sigma_y, force_even)

    kernel_2d = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())

    return kernel_2d


def get_laplacian_kernel1d(kernel_size):
    """
        Function that returns the coefficients of a 1D Laplacian filter.
    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or \
            kernel_size <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(kernel_size))

    window_1d = laplacian_1d(kernel_size)

    return window_1d


def get_laplacian_kernel2d(kernel_size):
    """
        Function that returns Gaussian filter matrix coefficients.
    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or \
            kernel_size <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(kernel_size))

    kernel = torch.ones((kernel_size, kernel_size))
    mid = kernel_size // 2
    kernel[mid, mid] = 1 - kernel_size ** 2

    kernel_2d = kernel

    return kernel_2d


def get_motion_kernel2d(kernel_size, angle, direction=0., mode='nearest'):
    """
        Return 2D motion blur filter.
    """
    device, dtype = _extract_device_dtype([
        angle if isinstance(angle, torch.Tensor) else None,
        direction if isinstance(direction, torch.Tensor) else None,
    ])

    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or kernel_size < 3:
        raise TypeError("ksize must be an odd integer >= than 3")

    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor([angle], device=device, dtype=dtype)

    if angle.dim() == 0:
        angle = angle.unsqueeze(0)

    assert angle.dim() == 1, f"angle must be a 1-dim tensor. Got {angle}."

    if not isinstance(direction, torch.Tensor):
        direction = torch.tensor([direction], device=device, dtype=dtype)

    if direction.dim() == 0:
        direction = direction.unsqueeze(0)

    assert direction.dim() == 1, f"direction must be a 1-dim tensor. Got {direction}."

    assert direction.size(0) == angle.size(0), \
        f"direction and angle must have the same length. Got {direction} and {angle}."

    kernel_tuple = (kernel_size, kernel_size)

    # direction from [-1, 1] to [0, 1] range

    direction = (torch.clamp(direction, -1., 1.) + 1.) / 2.
    k = torch.stack(
        [(direction + ((1 - 2 * direction) / (kernel_size - 1)) * i) for i in range(kernel_size)], dim=-1)

    kernel = torch.nn.functional.pad(k[:, None], [0, 0, kernel_size // 2, kernel_size // 2, 0, 0])

    assert kernel.shape == torch.Size([direction.size(0), *kernel_tuple])

    kernel = kernel.unsqueeze(1)

    # rotate (counterclockwise) kernel by given angle
    kernel = rotate(kernel, angle, mode=mode, align_corners=True)
    kernel = kernel[:, 0]
    kernel = kernel / kernel.sum(dim=(1, 2), keepdim=True)

    return kernel


def get_motion_kernel3d(kernel_size, angle, direction=0., mode='nearest'):
    """
        Return 3D motion blur filter.
    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or kernel_size < 3:
        raise TypeError(f"ksize must be an odd integer >= than 3. Got {kernel_size}.")

    device, dtype = _extract_device_dtype([
        angle if isinstance(angle, torch.Tensor) else None,
        direction if isinstance(direction, torch.Tensor) else None,
    ])

    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor([angle], device=device, dtype=dtype)

    if angle.dim() == 1:
        angle = angle.unsqueeze(0)

    assert len(angle.shape) == 2 and angle.size(1) == 3, f"angle must be (B, 3). Got {angle}."

    if not isinstance(direction, torch.Tensor):
        direction = torch.tensor([direction], device=device, dtype=dtype)

    if direction.dim() == 0:
        direction = direction.unsqueeze(0)

    assert direction.dim() == 1, f"direction must be a 1-dim tensor. Got {direction}."

    assert direction.size(0) == angle.size(0), \
        f"direction and angle must have the same length. Got {direction} and {angle}."

    kernel_tuple = (kernel_size, kernel_size, kernel_size)

    # direction from [-1, 1] to [0, 1] range
    direction = (torch.clamp(direction, -1., 1.) + 1.) / 2.
    kernel = torch.zeros((direction.size(0), *kernel_tuple), device=device, dtype=dtype)

    k = torch.stack(
        [(direction + ((1 - 2 * direction) / (kernel_size - 1)) * i) for i in range(kernel_size)], dim=-1)

    kernel = torch.nn.functional.pad(
        k[:, None, None], [0, 0, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, 0, 0])

    assert kernel.shape == torch.Size([direction.size(0), *kernel_tuple])

    kernel = kernel.unsqueeze(1)

    # rotate (counterclockwise) kernel by given angle
    kernel = rotate3d(kernel, angle[:, 0], angle[:, 1], angle[:, 2], mode=mode, align_corners=True)
    kernel = kernel[:, 0]
    kernel = kernel / kernel.sum(dim=(1, 2, 3), keepdim=True)

    return kernel