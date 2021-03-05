import torch

from ..geometry.conversions import pi

from .color.hsv import rgb_to_hsv, hsv_to_rgb

__all__ = [
    "adjust_brightness",
    "adjust_contrast",
    "adjust_gamma",
    "adjust_hue",
    "adjust_saturation",
    "adjust_hue_raw",
    "adjust_saturation_raw",
    "solarize",
    "equalize",
    "equalize3d",
    "posterize",
    "sharpness"
]


def adjust_saturation_raw(inp, saturation_factor):
    """
        Adjust color saturation of an image. Expecting inp to be in hsv format already.
    """

    if not isinstance(inp, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(inp)}")

    if not isinstance(saturation_factor, (float, torch.Tensor,)):
        raise TypeError(f"The saturation_factor should be a float number or torch.Tensor."
                        f"Got {type(saturation_factor)}")

    if isinstance(saturation_factor, float):
        saturation_factor = torch.as_tensor(saturation_factor)

    saturation_factor = saturation_factor.to(inp.device).to(inp.dtype)

    if (saturation_factor < 0).any():
        raise ValueError(f"Saturation factor must be non-negative. Got {saturation_factor}")

    for _ in inp.shape[1:]:
        saturation_factor = torch.unsqueeze(saturation_factor, dim=-1)

    # unpack the hsv values
    h, s, v = torch.chunk(inp, chunks=3, dim=-3)

    # transform the hue value and appl module
    s_out = torch.clamp(s * saturation_factor, min=0, max=1)

    # pack back back the corrected hue
    out = torch.cat([h, s_out, v], dim=-3)

    return out


def adjust_saturation(inp, saturation_factor):
    """
        Adjust color saturation of an image.
        The inp image is expected to be an RGB image in the range of [0, 1].
    """

    # convert the rgb image to hsv
    x_hsv = rgb_to_hsv(inp)

    # perform the conversion
    x_adjusted = adjust_saturation_raw(x_hsv, saturation_factor)

    # convert back to rgb
    out = hsv_to_rgb(x_adjusted)

    return out


def adjust_hue_raw(tensor, hue_factor):
    """
        Adjust hue of an image. Expecting inp to be in hsv format already.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if not isinstance(hue_factor, (float, torch.Tensor)):
        raise TypeError(f"The hue_factor should be a float number or torch.Tensor in the range between"
                        f" [-PI, PI]. Got {type(hue_factor)}")

    if isinstance(hue_factor, float):
        hue_factor = torch.as_tensor(hue_factor)

    hue_factor = hue_factor.to(tensor.device, tensor.dtype)

    if ((hue_factor < -pi) | (hue_factor > pi)).any():
         raise ValueError(f"Hue-factor must be in the range [-PI, PI]. Got {hue_factor}")

    for _ in tensor.shape[1:]:
        hue_factor = torch.unsqueeze(hue_factor, dim=-1)

    # unpack the hsv values
    h, s, v = torch.chunk(tensor, chunks=3, dim=-3)

    # transform the hue value and appl module
    divisor = (2 * pi).cuda(device=tensor.device)


    h_out = torch.fmod((h + hue_factor), divisor)

    # pack back back the corrected hue
    out = torch.cat([h_out, s, v], dim=-3)

    return out


def adjust_hue(tensor, hue_factor):
    """
        Adjust hue of an image.
        The inp image is expected to be an RGB image in the range of [0, 1].
    """

    # convert the rgb image to hsv
    x_hsv = rgb_to_hsv(tensor)

    # perform the conversion
    x_adjusted = adjust_hue_raw(x_hsv, hue_factor)

    # convert back to rgb
    out = hsv_to_rgb(x_adjusted)

    return out


def adjust_gamma(inp, gamma, gain=1.):
    """
        Perform gamma correction on an image.
        The inp image is expected to be in the range of [0, 1].
    """

    if not isinstance(inp, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(inp)}")

    if not isinstance(gamma, (float, torch.Tensor)):
        raise TypeError(f"The gamma should be a positive float or torch.Tensor. Got {type(gamma)}")

    if not isinstance(gain, (float, torch.Tensor)):
        raise TypeError(f"The gain should be a positive float or torch.Tensor. Got {type(gain)}")

    if isinstance(gamma, float):
        gamma = torch.tensor([gamma])

    if isinstance(gain, float):
        gain = torch.tensor([gain])

    gamma = gamma.to(inp.device).to(inp.dtype)
    gain = gain.to(inp.device).to(inp.dtype)

    if (gamma < 0.0).any():
        raise ValueError(f"Gamma must be non-negative. Got {gamma}")

    if (gain < 0.0).any():
        raise ValueError(f"Gain must be non-negative. Got {gain}")

    for _ in inp.shape[1:]:
        gamma = torch.unsqueeze(gamma, dim=-1)
        gain = torch.unsqueeze(gain, dim=-1)

    # Apply the gamma correction
    x_adjust = gain * torch.pow(inp, gamma)

    # Truncate between pixel values
    out = torch.clamp(x_adjust, 0.0, 1.0)

    return out


def adjust_contrast(inp, contrast_factor):
    """
        Adjust Contrast of an image.
        This implementation aligns OpenCV, not PIL. Hence, the output differs from TorchVision.
        The inp image is expected to be in the range of [0, 1].
    """

    if not isinstance(inp, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(inp)}")

    if not isinstance(contrast_factor, (float, torch.Tensor,)):
        raise TypeError(f"The factor should be either a float or torch.Tensor. "
                        f"Got {type(contrast_factor)}")

    if isinstance(contrast_factor, float):
        contrast_factor = torch.tensor([contrast_factor])

    contrast_factor = contrast_factor.to(inp.device).to(inp.dtype)

    if (contrast_factor < 0).any():
        raise ValueError(f"Contrast factor must be non-negative. Got {contrast_factor}")

    for _ in inp.shape[1:]:
        contrast_factor = torch.unsqueeze(contrast_factor, dim=-1)

    # Apply contrast factor to each channel
    x_adjust = inp * contrast_factor

    # Truncate between pixel values
    out = torch.clamp(x_adjust, 0.0, 1.0)

    return out


def adjust_brightness(inp, brightness_factor):
    """
        Adjust Brightness of an image.
        This implementation aligns OpenCV, not PIL. Hence, the output differs from TorchVision.
        The inp image is expected to be in the range of [0, 1].
    """

    if not isinstance(inp, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(inp)}")

    if not isinstance(brightness_factor, (float, torch.Tensor,)):
        raise TypeError(f"The factor should be either a float or torch.Tensor. "
                        f"Got {type(brightness_factor)}")

    if isinstance(brightness_factor, float):
        brightness_factor = torch.tensor([brightness_factor])

    brightness_factor = brightness_factor.to(inp.device).to(inp.dtype)

    for _ in inp.shape[1:]:
        brightness_factor = torch.unsqueeze(brightness_factor, dim=-1)

    # Apply brightness factor to each channel
    x_adjust = inp + brightness_factor

    # Truncate between pixel values
    out = torch.clamp(x_adjust, 0.0, 1.0)

    return out


def _solarize(inp, thresholds=0.5):
    """
        For each pixel in the image, select the pixel if the value is less than the threshold.
        Otherwise, subtract 1.0 from the pixel.
    """

    if not isinstance(inp, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(inp)}")

    if not isinstance(thresholds, (float, torch.Tensor,)):
        raise TypeError(f"The factor should be either a float or torch.Tensor. "
                        f"Got {type(thresholds)}")

    if isinstance(thresholds, torch.Tensor) and len(thresholds.shape) != 0:
        assert inp.size(0) == len(thresholds) and len(thresholds.shape) == 1, \
            f"threshholds must be a 1-d vector of shape ({inp.size(0)},). Got {thresholds}"

        thresholds = thresholds.to(inp.device).to(inp.dtype)
        thresholds = torch.stack([x.expand(*inp.shape[1:]) for x in thresholds])

    return torch.where(inp < thresholds, inp, 1.0 - inp)


def solarize(inp, thresholds=0.5, additions=None):
    """
        For each pixel in the image less than threshold.
        We add 'addition' amount to it and then clip the pixel value to be between 0 and 1.0.
        The value of 'addition' is between -0.5 and 0.5.
    """
    if not isinstance(inp, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(inp)}")

    if not isinstance(thresholds, (float, torch.Tensor,)):
        raise TypeError(f"The factor should be either a float or torch.Tensor. "
                        f"Got {type(thresholds)}")

    if isinstance(thresholds, float):
        thresholds = torch.tensor(thresholds)

    if additions is not None:
        if not isinstance(additions, (float, torch.Tensor,)):
            raise TypeError(f"The factor should be either a float or torch.Tensor. "
                            f"Got {type(additions)}")

        if isinstance(additions, float):
            additions = torch.tensor(additions)

        assert torch.all((additions < 0.5) * (additions > -0.5)), \
            f"The value of 'addition' is between -0.5 and 0.5. Got {additions}."

        if isinstance(additions, torch.Tensor) and len(additions.shape) != 0:
            assert inp.size(0) == len(additions) and len(additions.shape) == 1, \
                f"additions must be a 1-d vector of shape ({inp.size(0)},). Got {additions}"

            additions = additions.to(inp.device).to(inp.dtype)
            additions = torch.stack([x.expand(*inp.shape[1:]) for x in additions])

        inp = inp + additions
        inp = inp.clamp(0., 1.)

    return _solarize(inp, thresholds)


def posterize(inp, bits):
    """
        Reduce the number of bits for each color channel.
        Non-differentiable function, torch.uint8 involved.
    """

    if not isinstance(inp, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(inp)}")

    if not isinstance(bits, (int, torch.Tensor,)):
        raise TypeError(f"bits type is not an int or torch.Tensor. Got {type(bits)}")

    if isinstance(bits, int):
        bits = torch.tensor(bits)

    if not torch.all((bits >= 0) * (bits <= 8)) and bits.dtype == torch.int:
        raise ValueError(f"bits must be integers within range [0, 8]. Got {bits}.")

    # TODO: Make a differentiable version
    # Current version:
    # Ref: https://github.com/open-mmlab/mmcv/pull/132/files#diff-309c9320c7f71bedffe89a70ccff7f3bR19
    # Ref: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L222
    # Potential approach: implementing kornia.LUT with floating points
    # https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/functional.py#L472

    def _left_shift(inp, shift):
        return ((inp * 255).to(torch.uint8) * (2 ** shift)).to(inp.dtype) / 255.

    def _right_shift(inp, shift):
        return (inp * 255).to(torch.uint8) / (2 ** shift).to(inp.dtype) / 255.

    def _posterize_one(inp, bits):
        # Single bits value condition
        if bits == 0:
            return torch.zeros_like(inp)
        if bits == 8:
            return inp.clone()
        bits = 8 - bits
        return _left_shift(_right_shift(inp, bits), bits)

    if len(bits.shape) == 0 or (len(bits.shape) == 1 and len(bits) == 1):
        return _posterize_one(inp, bits)

    res = []
    if len(bits.shape) == 1:
        assert bits.shape[0] == inp.shape[0], \
            f"Batch size must be equal between bits and inp. Got {bits.shape[0]}, {inp.shape[0]}."

        for i in range(inp.shape[0]):
            res.append(_posterize_one(inp[i], bits[i]))
        return torch.stack(res, dim=0)

    assert bits.shape == inp.shape[:len(bits.shape)], \
        f"Batch and channel must be equal between bits and inp. Got {bits.shape}, {inp.shape[:len(bits.shape)]}."

    _inp = inp.view(-1, *inp.shape[len(bits.shape):])

    _bits = bits.flatten()

    for i in range(inp.shape[0]):
        res.append(_posterize_one(_inp[i], _bits[i]))

    return torch.stack(res, dim=0).reshape(*inp.shape)


def sharpness(inp, sharpness_factor):
    """
        Apply sharpness to the inp tensor.
        Implemented Sharpness function from PIL using torch ops. This implementation refers to:
        https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L326
    """

    if not isinstance(sharpness_factor, torch.Tensor):
        sharpness_factor = torch.tensor(sharpness_factor, device=inp.device, dtype=inp.dtype)

    if len(sharpness_factor.size()) != 0:
        assert sharpness_factor.shape == torch.Size([inp.size(0)]), (
            "Input batch size shall match with factor size if factor is not a 0-dim tensor. "
            f"Got {inp.size(0)} and {fasharpness_factorctor.shape}")

    kernel = torch.tensor([
        [1, 1, 1],
        [1, 5, 1],
        [1, 1, 1]
    ], dtype=inp.dtype, device=inp.device).view(1, 1, 3, 3).repeat(inp.size(1), 1, 1, 1) / 13

    # This shall be equivalent to depthwise conv2d:
    # Ref: https://discuss.pytorch.org/t/depthwise-and-separable-convolutions-in-pytorch/7315/2

    degenerate = torch.nn.functional.conv2d(inp, kernel, bias=None, stride=1, groups=inp.size(1))
    degenerate = torch.clamp(degenerate, 0., 1.)

    # For the borders of the resulting image, fill in the values of the original image.
    mask = torch.ones_like(degenerate)
    padded_mask = torch.nn.functional.pad(mask, [1, 1, 1, 1])
    padded_degenerate = torch.nn.functional.pad(degenerate, [1, 1, 1, 1])
    result = torch.where(padded_mask == 1, padded_degenerate, inp)

    if len(sharpness_factor.size()) == 0:
        return _blend_one(result, inp, sharpness_factor)

    return torch.stack([_blend_one(result[i], inp[i], sharpness_factor[i]) for i in range(len(sharpness_factor))])


def _blend_one(inp1, inp2, factor):
    """
        Blend two images into one.
    """
    assert isinstance(inp1, torch.Tensor), f"`inp1` must be a tensor. Got {inp1}."
    assert isinstance(inp2, torch.Tensor), f"`inp1` must be a tensor. Got {inp2}."

    if isinstance(factor, torch.Tensor):
        assert len(factor.size()) == 0, f"Factor shall be a float or single element tensor. Got {factor}."
    if factor == 0.:
        return inp1
    if factor == 1.:
        return inp2

    diff = (inp2 - inp1) * factor

    res = inp1 + diff

    if factor > 0. and factor < 1.:
        return res

    return torch.clamp(res, 0, 1)


def _build_lut(histo, step):
    # Compute the cumulative sum, shifting by step // 2
    # and then normalization by step.

    lut = (torch.cumsum(histo, 0) + (step // 2)) // step

    # Shift lut, prepending with 0.
    lut = torch.cat([torch.zeros(1, device=lut.device, dtype=lut.dtype), lut[:-1]])

    # Clip the counts to be in range.  This is done
    # in the C code for image.point.

    return torch.clamp(lut, 0, 255)


# Code taken from: https://github.com/pytorch/vision/pull/796
def _scale_channel(im):
    """
        Scale the data in the channel to implement equalize.
    """
    min_ = im.min()
    max_ = im.max()

    if min_.item() < 0. and not torch.isclose(min_, torch.tensor(0., dtype=min_.dtype)):
        raise ValueError(
            f"Values in the inp tensor must greater or equal to 0.0. Found {min_.item()}."
        )
    if max_.item() > 1. and not torch.isclose(max_, torch.tensor(1., dtype=max_.dtype)):
        raise ValueError(
            f"Values in the inp tensor must lower or equal to 1.0. Found {max_.item()}."
        )

    ndims = len(im.shape)
    if ndims not in (2, 3):
        raise TypeError(f"Input tensor must have 2 or 3 dimensions. Found {ndims}.")

    im = im * 255

    # Compute the histogram of the image channel.
    histo = torch.histc(im, bins=256, min=0, max=255)

    # For the purposes of computing the step, filter out the nonzeros.
    nonzero_histo = torch.reshape(histo[histo != 0], [-1])
    step = (torch.sum(nonzero_histo) - nonzero_histo[-1]) // 255

    # If step is zero, return the original image.  Otherwise, build
    # lut from the full histogram and step and then index from it.
    if step == 0:
        result = im
    else:
        # can't index using 2d index. Have to flatten and then reshape
        result = torch.gather(_build_lut(histo, step), 0, im.flatten().long())
        result = result.reshape_as(im)

    return result / 255.


def equalize(inp):
    """
        Apply equalize on the inp tensor.
        Implements Equalize function from PIL using PyTorch ops based on uint8 format:
        https://github.com/tensorflow/tpu/blob/5f71c12a020403f863434e96982a840578fdd127/models/official/efficientnet/autoaugment.py#L355
    """

    res = []
    for image in inp:
        # Assumes RGB for now.  Scales each channel independently
        # and then stacks the result.
        scaled_image = torch.stack([_scale_channel(image[i, :, :]) for i in range(len(image))])
        res.append(scaled_image)

    return torch.stack(res)


def equalize3d(inp):
    """
        Equalizes the values for a 3D volumetric tensor.
        Implements Equalize function for a sequence of images using PyTorch ops based on uint8 format:
        https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L352
    """

    res = []
    for volume in inp:
        # Assumes RGB for now.  Scales each channel independently
        # and then stacks the result.
        scaled_inp = torch.stack([_scale_channel(volume[i, :, :, :]) for i in range(len(volume))])
        res.append(scaled_inp)

    return torch.stack(res)
