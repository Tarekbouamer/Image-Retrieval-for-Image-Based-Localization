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


def adjust_saturation_raw(input, saturation_factor):
    """
        Adjust color saturation of an image. Expecting input to be in hsv format already.
    """

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(saturation_factor, (float, torch.Tensor,)):
        raise TypeError(f"The saturation_factor should be a float number or torch.Tensor."
                        f"Got {type(saturation_factor)}")

    if isinstance(saturation_factor, float):
        saturation_factor = torch.as_tensor(saturation_factor)

    saturation_factor = saturation_factor.to(input.device).to(input.dtype)

    if (saturation_factor < 0).any():
        raise ValueError(f"Saturation factor must be non-negative. Got {saturation_factor}")

    for _ in input.shape[1:]:
        saturation_factor = torch.unsqueeze(saturation_factor, dim=-1)

    # unpack the hsv values
    h, s, v = torch.chunk(input, chunks=3, dim=-3)

    # transform the hue value and appl module
    s_out = torch.clamp(s * saturation_factor, min=0, max=1)

    # pack back back the corrected hue
    out = torch.cat([h, s_out, v], dim=-3)

    return out


def adjust_saturation(input, saturation_factor):
    """
        Adjust color saturation of an image.
        The input image is expected to be an RGB image in the range of [0, 1].
    """

    # convert the rgb image to hsv
    x_hsv = rgb_to_hsv(input)

    # perform the conversion
    x_adjusted = adjust_saturation_raw(x_hsv, saturation_factor)

    # convert back to rgb
    out = hsv_to_rgb(x_adjusted)

    return out


def adjust_hue_raw(input, hue_factor):
    """
        Adjust hue of an image. Expecting input to be in hsv format already.
    """

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(hue_factor, (float, torch.Tensor)):
        raise TypeError(f"The hue_factor should be a float number or torch.Tensor in the range between"
                        f" [-PI, PI]. Got {type(hue_factor)}")

    if isinstance(hue_factor, float):
        hue_factor = torch.as_tensor(hue_factor)

    hue_factor = hue_factor.to(input.device, input.dtype)

    if ((hue_factor < -pi) | (hue_factor > pi)).any():
         raise ValueError(f"Hue-factor must be in the range [-PI, PI]. Got {hue_factor}")

    for _ in input.shape[1:]:
        hue_factor = torch.unsqueeze(hue_factor, dim=-1)

    # unpack the hsv values
    h, s, v = torch.chunk(input, chunks=3, dim=-3)

    # transform the hue value and appl module
    divisor = 2 * pi
    h_out = torch.fmod(h + hue_factor, divisor)

    # pack back back the corrected hue
    out = torch.cat([h_out, s, v], dim=-3)

    return out


def adjust_hue(input, hue_factor):
    """
        Adjust hue of an image.
        The input image is expected to be an RGB image in the range of [0, 1].
    """

    # convert the rgb image to hsv
    x_hsv = rgb_to_hsv(input)

    # perform the conversion
    x_adjusted = adjust_hue_raw(x_hsv, hue_factor)

    # convert back to rgb
    out = hsv_to_rgb(x_adjusted)

    return out


def adjust_gamma(input, gamma, gain=1.):
    """
        Perform gamma correction on an image.
        The input image is expected to be in the range of [0, 1].
    """

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(gamma, (float, torch.Tensor)):
        raise TypeError(f"The gamma should be a positive float or torch.Tensor. Got {type(gamma)}")

    if not isinstance(gain, (float, torch.Tensor)):
        raise TypeError(f"The gain should be a positive float or torch.Tensor. Got {type(gain)}")

    if isinstance(gamma, float):
        gamma = torch.tensor([gamma])

    if isinstance(gain, float):
        gain = torch.tensor([gain])

    gamma = gamma.to(input.device).to(input.dtype)
    gain = gain.to(input.device).to(input.dtype)

    if (gamma < 0.0).any():
        raise ValueError(f"Gamma must be non-negative. Got {gamma}")

    if (gain < 0.0).any():
        raise ValueError(f"Gain must be non-negative. Got {gain}")

    for _ in input.shape[1:]:
        gamma = torch.unsqueeze(gamma, dim=-1)
        gain = torch.unsqueeze(gain, dim=-1)

    # Apply the gamma correction
    x_adjust = gain * torch.pow(input, gamma)

    # Truncate between pixel values
    out = torch.clamp(x_adjust, 0.0, 1.0)

    return out


def adjust_contrast(input, contrast_factor):
    """
        Adjust Contrast of an image.
        This implementation aligns OpenCV, not PIL. Hence, the output differs from TorchVision.
        The input image is expected to be in the range of [0, 1].
    """

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(contrast_factor, (float, torch.Tensor,)):
        raise TypeError(f"The factor should be either a float or torch.Tensor. "
                        f"Got {type(contrast_factor)}")

    if isinstance(contrast_factor, float):
        contrast_factor = torch.tensor([contrast_factor])

    contrast_factor = contrast_factor.to(input.device).to(input.dtype)

    if (contrast_factor < 0).any():
        raise ValueError(f"Contrast factor must be non-negative. Got {contrast_factor}")

    for _ in input.shape[1:]:
        contrast_factor = torch.unsqueeze(contrast_factor, dim=-1)

    # Apply contrast factor to each channel
    x_adjust = input * contrast_factor

    # Truncate between pixel values
    out = torch.clamp(x_adjust, 0.0, 1.0)

    return out


def adjust_brightness(input, brightness_factor):
    """
        Adjust Brightness of an image.
        This implementation aligns OpenCV, not PIL. Hence, the output differs from TorchVision.
        The input image is expected to be in the range of [0, 1].
    """

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(brightness_factor, (float, torch.Tensor,)):
        raise TypeError(f"The factor should be either a float or torch.Tensor. "
                        f"Got {type(brightness_factor)}")

    if isinstance(brightness_factor, float):
        brightness_factor = torch.tensor([brightness_factor])

    brightness_factor = brightness_factor.to(input.device).to(input.dtype)

    for _ in input.shape[1:]:
        brightness_factor = torch.unsqueeze(brightness_factor, dim=-1)

    # Apply brightness factor to each channel
    x_adjust = input + brightness_factor

    # Truncate between pixel values
    out = torch.clamp(x_adjust, 0.0, 1.0)

    return out


def _solarize(input, thresholds=0.5):
    """
        For each pixel in the image, select the pixel if the value is less than the threshold.
        Otherwise, subtract 1.0 from the pixel.
    """

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(thresholds, (float, torch.Tensor,)):
        raise TypeError(f"The factor should be either a float or torch.Tensor. "
                        f"Got {type(thresholds)}")

    if isinstance(thresholds, torch.Tensor) and len(thresholds.shape) != 0:
        assert input.size(0) == len(thresholds) and len(thresholds.shape) == 1, \
            f"threshholds must be a 1-d vector of shape ({input.size(0)},). Got {thresholds}"

        thresholds = thresholds.to(input.device).to(input.dtype)
        thresholds = torch.stack([x.expand(*input.shape[1:]) for x in thresholds])

    return torch.where(input < thresholds, input, 1.0 - input)


def solarize(input, thresholds=0.5, additions=None):
    """
        For each pixel in the image less than threshold.
        We add 'addition' amount to it and then clip the pixel value to be between 0 and 1.0.
        The value of 'addition' is between -0.5 and 0.5.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

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
            assert input.size(0) == len(additions) and len(additions.shape) == 1, \
                f"additions must be a 1-d vector of shape ({input.size(0)},). Got {additions}"

            additions = additions.to(input.device).to(input.dtype)
            additions = torch.stack([x.expand(*input.shape[1:]) for x in additions])

        input = input + additions
        input = input.clamp(0., 1.)

    return _solarize(input, thresholds)


def posterize(input, bits):
    """
        Reduce the number of bits for each color channel.
        Non-differentiable function, torch.uint8 involved.
    """

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

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

    def _left_shift(input, shift):
        return ((input * 255).to(torch.uint8) * (2 ** shift)).to(input.dtype) / 255.

    def _right_shift(input, shift):
        return (input * 255).to(torch.uint8) / (2 ** shift).to(input.dtype) / 255.

    def _posterize_one(input, bits):
        # Single bits value condition
        if bits == 0:
            return torch.zeros_like(input)
        if bits == 8:
            return input.clone()
        bits = 8 - bits
        return _left_shift(_right_shift(input, bits), bits)

    if len(bits.shape) == 0 or (len(bits.shape) == 1 and len(bits) == 1):
        return _posterize_one(input, bits)

    res = []
    if len(bits.shape) == 1:
        assert bits.shape[0] == input.shape[0], \
            f"Batch size must be equal between bits and input. Got {bits.shape[0]}, {input.shape[0]}."

        for i in range(input.shape[0]):
            res.append(_posterize_one(input[i], bits[i]))
        return torch.stack(res, dim=0)

    assert bits.shape == input.shape[:len(bits.shape)], \
        f"Batch and channel must be equal between bits and input. Got {bits.shape}, {input.shape[:len(bits.shape)]}."

    _input = input.view(-1, *input.shape[len(bits.shape):])

    _bits = bits.flatten()

    for i in range(input.shape[0]):
        res.append(_posterize_one(_input[i], _bits[i]))

    return torch.stack(res, dim=0).reshape(*input.shape)


def sharpness(input, factor):
    """
        Apply sharpness to the input tensor.
        Implemented Sharpness function from PIL using torch ops. This implementation refers to:
        https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L326
    """

    if not isinstance(factor, torch.Tensor):
        factor = torch.tensor(factor, device=input.device, dtype=input.dtype)

    if len(factor.size()) != 0:
        assert factor.shape == torch.Size([input.size(0)]), (
            "Input batch size shall match with factor size if factor is not a 0-dim tensor. "
            f"Got {input.size(0)} and {factor.shape}")

    kernel = torch.tensor([
        [1, 1, 1],
        [1, 5, 1],
        [1, 1, 1]
    ], dtype=input.dtype, device=input.device).view(1, 1, 3, 3).repeat(input.size(1), 1, 1, 1) / 13

    # This shall be equivalent to depthwise conv2d:
    # Ref: https://discuss.pytorch.org/t/depthwise-and-separable-convolutions-in-pytorch/7315/2

    degenerate = torch.nn.functional.conv2d(input, kernel, bias=None, stride=1, groups=input.size(1))
    degenerate = torch.clamp(degenerate, 0., 1.)

    # For the borders of the resulting image, fill in the values of the original image.
    mask = torch.ones_like(degenerate)
    padded_mask = torch.nn.functional.pad(mask, [1, 1, 1, 1])
    padded_degenerate = torch.nn.functional.pad(degenerate, [1, 1, 1, 1])
    result = torch.where(padded_mask == 1, padded_degenerate, input)

    if len(factor.size()) == 0:
        return _blend_one(result, input, factor)

    return torch.stack([_blend_one(result[i], input[i], factor[i]) for i in range(len(factor))])


def _blend_one(input1, input2, factor):
    """
        Blend two images into one.
    """
    assert isinstance(input1, torch.Tensor), f"`input1` must be a tensor. Got {input1}."
    assert isinstance(input2, torch.Tensor), f"`input1` must be a tensor. Got {input2}."

    if isinstance(factor, torch.Tensor):
        assert len(factor.size()) == 0, f"Factor shall be a float or single element tensor. Got {factor}."
    if factor == 0.:
        return input1
    if factor == 1.:
        return input2

    diff = (input2 - input1) * factor

    res = input1 + diff

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
            f"Values in the input tensor must greater or equal to 0.0. Found {min_.item()}."
        )
    if max_.item() > 1. and not torch.isclose(max_, torch.tensor(1., dtype=max_.dtype)):
        raise ValueError(
            f"Values in the input tensor must lower or equal to 1.0. Found {max_.item()}."
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


def equalize(input):
    """
        Apply equalize on the input tensor.
        Implements Equalize function from PIL using PyTorch ops based on uint8 format:
        https://github.com/tensorflow/tpu/blob/5f71c12a020403f863434e96982a840578fdd127/models/official/efficientnet/autoaugment.py#L355
    """

    res = []
    for image in input:
        # Assumes RGB for now.  Scales each channel independently
        # and then stacks the result.
        scaled_image = torch.stack([_scale_channel(image[i, :, :]) for i in range(len(image))])
        res.append(scaled_image)

    return torch.stack(res)


def equalize3d(input):
    """
        Equalizes the values for a 3D volumetric tensor.
        Implements Equalize function for a sequence of images using PyTorch ops based on uint8 format:
        https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L352
    """

    res = []
    for volume in input:
        # Assumes RGB for now.  Scales each channel independently
        # and then stacks the result.
        scaled_input = torch.stack([_scale_channel(volume[i, :, :, :]) for i in range(len(volume))])
        res.append(scaled_input)

    return torch.stack(res)
