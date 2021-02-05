import torch.nn.functional as F

from .kernels import normalize_kernel2d


def _compute_padding(kernel_size):
    """
        Computes padding tuple.
    """

    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad

    assert len(kernel_size) >= 2, kernel_size
    computed = [k // 2 for k in kernel_size]

    # for even kernels we need to do asymetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        if kernel_size[i] % 2 == 0:
            padding = computed_tmp - 1

        else:
            padding = computed_tmp

        out_padding[2 * i + 0] = padding
        out_padding[2 * i + 1] = computed_tmp

    return out_padding


def filter2D(input, kernel, border_type='reflect', normalized=False):
    """
        Convolve a tensor with a 2d kernel.
        The function applies a given kernel to a tensor. The kernel is applied
        independently at each depth channel of the tensor. Before applying the
        kernel, the function applies padding according to the specified mode so
        that the output remains in the same shape.

    """
    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    if not len(kernel.shape) == 3 and kernel.shape[0] != 1:
        raise ValueError("Invalid kernel shape, we expect 1xHxW. Got: {}"
                         .format(kernel.shape))

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel.unsqueeze(1).to(input)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]

    padding_shape = _compute_padding([height, width])
    input_pad = F.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input_pad = input_pad.view(-1, tmp_kernel.size(0), input_pad.size(-2), input_pad.size(-1))

    # convolve the tensor with the kernel.
    output = F.conv2d(input_pad, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    return output.view(b, c, h, w)


def filter3D(input, kernel, border_type='replicate', normalized=False):
    """
        Convolve a tensor with a 3d kernel.
        The function applies a given kernel to a tensor. The kernel is applied
        independently at each depth channel of the tensor. Before applying the
        kernel, the function applies padding according to the specified mode so
        that the output remains in the same shape.

    """
    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    if not len(input.shape) == 5:
        raise ValueError("Invalid input shape, we expect BxCxDxHxW. Got: {}"
                         .format(input.shape))

    if not len(kernel.shape) == 4 and kernel.shape[0] != 1:
        raise ValueError("Invalid kernel shape, we expect 1xDxHxW. Got: {}"
                         .format(kernel.shape))

    # prepare kernel
    b, c, d, h, w = input.shape
    tmp_kernel = kernel.unsqueeze(1).to(input)

    if normalized:
        bk, dk, hk, wk = kernel.shape
        tmp_kernel = normalize_kernel2d(tmp_kernel.view(bk, dk, hk * wk)).view_as(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1, -1)

    # pad the input tensor
    depth, height, width = tmp_kernel.shape[-3:]
    padding_shape = _compute_padding([depth, height, width])
    input_pad = F.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, depth, height, width)
    input_pad = input_pad.view(-1, tmp_kernel.size(0), input_pad.size(-3), input_pad.size(-2), input_pad.size(-1))

    # convolve the tensor with the kernel.
    output = F.conv3d(input_pad, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    return output.view(b, c, d, h, w)
