
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Uniform

from cirtorch.geometry.linalg import(
    transform_points,
    transform_boxes,
    validate_points
    )

from cirtorch.geometry.conversions import convert_affinematrix_to_homography
from cirtorch.geometry.transform.centerlize import centralize
from cirtorch.geometry.transform.flips import (
    hflip,
    vflip
)

from cirtorch.geometry.transform.affwarp import (
    _compute_tensor_center,
    _compute_rotation_matrix,
    rotate
)

from cirtorch.geometry.transform.imgwarp import (
    warp_perspective,
    get_perspective_transform,
    warp_affine,
    get_affine_matrix2d
)

from cirtorch.enhance.adjust import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    adjust_hue,
    solarize,
    equalize,
    posterize,
    sharpness,
)

from cirtorch.enhance.color.gray import (
    rgb_to_grayscale
)

from cirtorch.filters.motion import motion_blur


class AugmentationBase(nn.Module):

    def __init__(self, p=0.2):
        super(AugmentationBase, self).__init__()
        self.p = p

    def _adapted_uniform(self, shape, low, high):
        """
            The Uniform Dist function, used to choose parameters within a range 
        """
        low = torch.as_tensor(low, device=low.device, dtype=low.dtype)
        high = torch.as_tensor(high, device=high.device, dtype=high.dtype)

        dist = Uniform(low, high)
        return dist.rsample(shape)

    def _adapted_sampling(self, shape, device, dtype):
        """
            The Bernoulli sampling function, used to sample from batch
        """
        _bernoulli = Bernoulli(torch.tensor(float(self.p), device=device, dtype=dtype))
        target = _bernoulli.sample((shape,)).bool()
        return target

    def generator(self):
        raise NotImplemented

    def apply_img(self, img=None, transform=None, target=None):
        raise NotImplemented

    def apply_kpts(self, kpts=None, transform=None, target=None):
        raise NotImplemented

    def apply_bbx(self, bbx=None, transform=None, target=None):
        raise NotImplemented
    
    def apply_transform(self, transform0=None, transform=None, target=None):
        out = transform0.clone()

        out[target] = torch.matmul(transform0, transform)[target]
        
        return out 

    def get_transform(self, params=None):
        raise NotImplemented

    def forward(self, inp):

        img_size = inp["img"].shape[-2:]
        batch_size = inp["img"].shape[0]
        valid_size = inp["valid_size"]

        # get target tensors
        params = self.generator(batch_size, img_size, valid_size, device=inp["img"].device, dtype=inp["img"].dtype)

        # Get transform
        transform = self.get_transform(params=params)

        # Apply transform to img
        inp["img"] = self.apply_img(img=inp["img"], transform=transform, target=params["target"])

        # Apply transform to kpts
        if inp.__contains__("kpts"):
            inp["kpts"] = self.apply_kpts(kpts=inp["kpts"], transform=transform, img_size=img_size, target=params["target"])

        # Apply transform to bbx
        if inp.__contains__("bbx"):
            inp["bbx"] = self.apply_bbx(bbx=inp["bbx"], transform=transform,target=params["target"])      
        
        # Total transform
        if inp.__contains__("transform"):
            if isinstance(transform, torch.Tensor):
                inp["transform"] = self.apply_transform(transform0=inp["transform"], transform=transform, target=params["target"])

        return inp

# --------------------------------------
#             Centralize
# --------------------------------------

class Centralize(AugmentationBase):
    """ 
        Applies a transformatioon to centralize images in tile.
    """
    def __init__(self, interpolation='bilinear', padding_mode='zeros', align_corners=False, bbx_mode="xywh"):
        super(Centralize, self).__init__(p=1)

        self.bbx_mode = bbx_mode
        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self.align_corners = align_corners
    
    
    def generator(self, batch_size, img_size, valid_size, device, dtype):

        target = self._adapted_sampling(batch_size, device, dtype)

        sizes = [torch.tensor([item[0], item[1]]) for item in valid_size]
        sizes = torch.stack(sizes).to(device=device, dtype=dtype)

        max_size = img_size
        max_size = torch.tensor([max_size[0], max_size[1]]).float().to(device=device, dtype=dtype)

        # params
        params = dict()
        
        params["target"] = target
        
        params["sizes"] = sizes
        params["max_size"] = max_size
        
        return params

    def get_transform(self, params):
        
        transform = centralize(sizes=params["sizes"], max_size=params["max_size"])

        return transform

    def apply_img(self, img, transform, target):

        out = img.clone()
        size = img.shape[-2:]

        out[target]= warp_affine(img, transform, size,
                                mode=self.interpolation,
                                padding_mode=self.padding_mode,
                                align_corners=self.align_corners)[target]

        return out
    
    def apply_kpts(self, kpts, transform, target, img_size):

        out = kpts.clone()

        out[target] =  transform_points(trans_01=transform, points_1=kpts)[target]

        return out
    
    def apply_bbx(self, bbx, transform, target):

        out = bbx.clone()

        out[target] =  transform_boxes(trans_mat=transform, boxes=bbx, mode=self.bbx_mode)[target]

        return out

# --------------------------------------
#             Geometric
# --------------------------------------

class RandomPerspective(AugmentationBase):
    r"""Applies a random perspective transformation to an image tensor with a given probability.

    """

    def __init__(self, p, distortion_scale, interpolation='bilinear', border_mode='zeros', align_corners=False, bbx_mode="xywh"):
        super(RandomPerspective, self).__init__(p=p)
        self.distortion_scale = distortion_scale
        
        self.bbx_mode = bbx_mode
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.align_corners = align_corners

    def generator(self, batch_size, img_size, valid_size, device, dtype):

        target = self._adapted_sampling(batch_size, device, dtype)

        height, width = img_size

        distortion_scale = torch.as_tensor(self.distortion_scale, device=device, dtype=dtype)

        assert distortion_scale.dim() == 0 and 0 <= distortion_scale <= 1, \
            f"'distortion_scale' must be a scalar within [0, 1]. Got {distortion_scale}."

        assert type(height) == int and height > 0 and type(width) == int and width > 0, \
            f"'height' and 'width' must be integers. Got {height}, {width}."

        start_points = torch.tensor([[
            [0., 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ]], device=device, dtype=dtype).expand(batch_size, -1, -1)

        # generate random offset not larger than half of the image
        fx = distortion_scale * width / 2
        fy = distortion_scale * height / 2

        factor = torch.stack([fx, fy], dim=0).view(-1, 1, 2)

        pts_norm = torch.tensor([[
            [1, 1],
            [-1, 1],
            [-1, -1],
            [1, -1]
        ]], device=device, dtype=dtype)

        rand_val = self._adapted_uniform(start_points.shape,
                                         torch.tensor(0, device=device, dtype=dtype),
                                         torch.tensor(1, device=device, dtype=dtype)
                                         ).to(device=device, dtype=dtype)

        end_points = start_points + factor * rand_val * pts_norm

        params = dict()
        params["target"] = target
        params["start_points"] = start_points
        params["end_points"] = end_points

        return params

    def get_transform(self, params):
        
        transform = get_perspective_transform(params['start_points'], params['end_points'])

        return transform

    def apply_img(self, img, transform, target):

        out = img.clone()
        size = img.shape[-2:]

        out[target] = warp_perspective(img, transform, size,
                                       mode=self.interpolation,
                                       border_mode=self.border_mode,
                                       align_corners=self.align_corners)[target]

        return out
    
    def apply_kpts(self, kpts, transform, target, img_size):

        out = kpts.clone()

        out[target] =  transform_points(trans_01=transform, points_1=kpts)[target]

        out[target] = validate_points(points=out, img_size=img_size, offset=7)[target]

        return out
    
    def apply_bbx(self, bbx, transform, target):

        out = bbx.clone()

        out[target] =  transform_boxes(trans_mat=transform, boxes=bbx, mode=self.bbx_mode)[target]

        return out


class RandomAffine(AugmentationBase):
    """
        Applies a random 2D affine transformation to a tensor image.
    """

    def __init__(self, p, theta, h_trans, v_trans, scale, shear,
                 interpolation='bilinear', padding_mode='zeros', align_corners=False, bbx_mode="xywh"):
        super(RandomAffine, self).__init__(p=p)
        self.theta = [-theta, theta]
        self.translate = [h_trans, v_trans]
        self.scale = scale
        self.shear = shear

        self.bbx_mode = bbx_mode
        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def generator(self, batch_size, img_size, valid_size, device, dtype):

        height, width = img_size

        target = self._adapted_sampling(batch_size, device, dtype)

        assert isinstance(width, (int,)) and isinstance(height, (int,)) and width > 0 and height > 0, \
            f"`width` and `height` must be positive integers. Got {width}, {height}."

        degrees = torch.as_tensor(self.theta).to(device=device, dtype=dtype)
        angle = self._adapted_uniform((batch_size,), degrees[0], degrees[1]).to(device=device, dtype=dtype)

        # compute tensor ranges
        if self.scale is not None:
            scale = torch.as_tensor(self.scale).to(device=device, dtype=dtype)

            assert len(scale.shape) == 1 and (len(scale) == 2), \
                f"`scale` shall have 2 or 4 elements. Got {scale}."

            _scale = self._adapted_uniform((batch_size,), scale[0], scale[1]).unsqueeze(1).repeat(1, 2)

        else:
            _scale = torch.ones((batch_size, 2), device=device, dtype=dtype)

        if self.translate is not None:
            translate = torch.as_tensor(self.translate).to(device=device, dtype=dtype)

            max_dx = translate[0] * width
            max_dy = translate[1] * height

            translations = torch.stack([
                self._adapted_uniform((batch_size,), max_dx * 0, max_dx),
                self._adapted_uniform((batch_size,), max_dy * 0, max_dy)
            ], dim=-1).to(device=device, dtype=dtype)

        else:
            translations = torch.zeros((batch_size, 2), device=device, dtype=dtype)

        center = torch.tensor([width, height], device=device, dtype=dtype).view(1, 2) / 2. - 0.5
        center = center.expand(batch_size, -1)

        if self.shear is not None:
            shear = torch.as_tensor(self.shear).to(device=device, dtype=dtype)

            sx = self._adapted_uniform((batch_size,), shear[0], shear[1]).to(device=device, dtype=dtype)
            sy = self._adapted_uniform((batch_size,), shear[0], shear[1]).to(device=device, dtype=dtype)

            sx = sx.to(device=device, dtype=dtype)
            sy = sy.to(device=device, dtype=dtype)
        else:
            sx = sy = torch.tensor([0] * batch_size, device=device, dtype=dtype)

        # params
        params = dict()
        params["target"] = target
        params["translations"] = translations
        params["center"] = center
        params["scale"] = _scale
        params["angle"] = angle
        params["sx"] = sx
        params["sy"] = sy
        return params

    def get_transform(self, params):
        
        # concatenate transforms
        transform = get_affine_matrix2d(translations=params["translations"],
                                        center=params["center"],
                                        scale=params["scale"],
                                        angle=params["angle"],
                                        sx=params["sx"],
                                        sy=params["sy"])
        return transform

    def apply_img(self, img, transform, target):

        out = img.clone()
        size = img.shape[-2:]

        out[target] = warp_affine(img, transform, size,
                                  mode=self.interpolation,
                                  padding_mode=self.padding_mode,
                                  align_corners=self.align_corners)[target]
        return out
    
    def apply_kpts(self, kpts, transform, target, img_size):

        out = kpts.clone()

        out[target] =  transform_points(trans_01=transform, points_1=kpts)[target]

        out[target] = validate_points(points=out, img_size=img_size, offset=7)[target]

        return out
    
    def apply_bbx(self, bbx, transform, target):

        out = bbx.clone()

        out[target] =  transform_boxes(trans_mat=transform, boxes=bbx, mode=self.bbx_mode)[target]

        return out



class RandomRotation(AugmentationBase):
    """
        Applies a random rotation to a tensor image or a batch of tensor images given an amount of degrees.
    """

    def __init__(self, p, theta, interpolation='bilinear', padding_mode='zeros', align_corners=False, bbx_mode="xywh"):
        super(RandomRotation, self).__init__(p=p)

        self.theta = [-theta, theta]

        self.bbx_mode = bbx_mode
        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def generator(self, batch_size, img_size, valid_size, device, dtype):
        """
            Get parameters for ``rotate`` for a random rotate transform.
        """
        target = self._adapted_sampling(batch_size, device, dtype)

        angle = torch.as_tensor(self.theta).to(device=device, dtype=dtype)
        angle = self._adapted_uniform((batch_size,), angle[0], angle[1]).to(device=device, dtype=dtype)

        #
        center = _compute_tensor_center(img_size, device=device, dtype=dtype)
        center = center.expand(batch_size, -1)
        
        angle = angle.expand(batch_size)

        # params
        params = dict()
        params["target"] = target
        params["angle"] = angle
        params["center"] = center


        return params
    
    def get_transform(self, params):

        transform = _compute_rotation_matrix(params["angle"], params["center"])
        return transform

    def apply_img(self, img, transform, target):

        out = img.clone()
        size = img.shape[-2:]


        out[target] = warp_affine(img, transform, size,
                                  mode=self.interpolation,
                                  padding_mode=self.padding_mode,
                                  align_corners=self.align_corners)[target]

        #out[target] = rotate(img, transform,
        #                    align_corners=self.align_corners,
        #                    mode=self.interpolation)[target]
        return out
    
    def apply_kpts(self, kpts, transform, target, img_size):

        out = kpts.clone()

        out[target] =  transform_points(trans_01=transform, points_1=kpts)[target]

        out[target] = validate_points(points=out, img_size=img_size, offset=7)[target]

        return out
    
    def apply_bbx(self, bbx, transform, target):

        out = bbx.clone()

        out[target] =  transform_boxes(trans_mat=transform, boxes=bbx, mode=self.bbx_mode)[target]

        return out


# --------------------------------------
#             Photometric
# --------------------------------------

class ColorJitter(AugmentationBase):

    def __init__(self, p, brightness=None, contrast=None, saturation=None, hue=None):
        super(ColorJitter, self).__init__(p=p)

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def generator(self, batch_size, img_size, valid_size, device, dtype):
        r"""Generate random color jiter parameters for a batch of images.
        """

        target = self._adapted_sampling(batch_size, device, dtype)

        if self.brightness is not None:
            brightness = torch.as_tensor(self.brightness).to(device=device, dtype=dtype)
        else:
            brightness = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

        if self.contrast is not None:
            contrast = torch.as_tensor(self.contrast).to(device=device, dtype=dtype)
        else:
            contrast = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

        if self.saturation is not None:
            saturation = torch.as_tensor(self.saturation).to(device=device, dtype=dtype)
        else:
            saturation = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

        if self.hue is not None:
            hue = torch.as_tensor(self.hue).to(device=device, dtype=dtype)
        else:
            hue = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

        brightness_factor = self._adapted_uniform((batch_size,), brightness[0], brightness[1]).to(device=device, dtype=dtype)
        contrast_factor = self._adapted_uniform((batch_size,), contrast[0], contrast[1]).to(device=device, dtype=dtype)
        saturation_factor = self._adapted_uniform((batch_size,), saturation[0], saturation[1]).to(device=device, dtype=dtype)
        hue_factor = self._adapted_uniform((batch_size,), hue[0], hue[1]).to(device=device, dtype=dtype)
        # Params
        params = dict()

        params["brightness_factor"] = brightness_factor
        params["contrast_factor"] = contrast_factor
        params["hue_factor"] = hue_factor
        params["saturation_factor"] = saturation_factor

        params["target"] = target

        return params

    def get_transform(self, params):

        transform = [
            lambda img: adjust_brightness(img, brightness_factor=params["brightness_factor"]),
            lambda img: adjust_contrast(img, contrast_factor=params["contrast_factor"]),
            lambda img: adjust_saturation(img, saturation_factor=params["saturation_factor"]),
            lambda img: adjust_hue(img, hue_factor=params["hue_factor"])
        ]

        return transform

    def apply_img(self, img, transform, target):

        out = img.clone()
        
        order = torch.randperm(4, device=img.device, dtype=img.dtype).long()

        for idx in order.tolist():

            JIT = transform[idx]
            
            out[target] = JIT(img)[target]

        return out

    def apply_kpts(self, kpts, transform, target, img_size):

        return kpts
    
    def apply_bbx(self, bbx, transform, target):
        return bbx


class RandomSolarize(AugmentationBase):
    r"""Solarize given tensor image or a batch of tensor images randomly.
    """
    def __init__(self, p, thresholds=0.1, additions=0.1):
        super(RandomSolarize, self).__init__(p=p)

        self.thresholds = thresholds
        self.additions = additions

    def generator(self, batch_size, img_size, valid_size, device, dtype):

        """
            Generate random solarize parameters for a batch of images.
            For each pixel in the image less than threshold, 
            we add 'addition' amount to it and then clip the pixel value
            to be between 0 and 1.0
        """
        target = self._adapted_sampling(batch_size, device, dtype)

        if self.thresholds is not None:
            thresholds = torch.as_tensor(self.thresholds).to(device=device, dtype=dtype)
        else:
            thresholds= torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

        if self.additions is not None:
            additions = torch.as_tensor(self.additions).to(device=device, dtype=dtype)
        else:
            additions = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

        thresholds = self._adapted_uniform((batch_size,), thresholds[0], thresholds[1]).to(device=device, dtype=dtype)
        additions = self._adapted_uniform((batch_size,), additions[0], additions[1]).to(device=device, dtype=dtype)

        # Params
        params = dict()

        params["thresholds"] = thresholds
        params["additions"] = additions

        params["target"] = target

        return params
    
    def get_transform(self, params):

        transform = lambda img: solarize(img, thresholds=params["thresholds"], additions=params["additions"])

        return transform
    
    def apply_img(self, img, transform, target):

        out = img.clone()

        out[target] = transform(img)[target]

        return out
    
    def apply_kpts(self, kpts, transform, target, img_size):

        return kpts
    
    def apply_bbx(self, bbx, transform, target):

        return bbx


class RandomPosterize(AugmentationBase):
    r"""Posterize given tensor image or a batch of tensor images randomly.
    """
    def __init__(self, p, bits=3):
        super(RandomPosterize, self).__init__(p=p)
        self.bits = bits

    def generator(self, batch_size, img_size, valid_size, device, dtype):
        """
            Generate random posterize parameters for a batch of images.
        """

        target = self._adapted_sampling(batch_size, device, dtype)

        if self.bits is not None:
            bits = torch.as_tensor(self.bits).to(device=device, dtype=dtype)
        else:
            bits = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

        bits = self._adapted_uniform((batch_size,), bits[0], bits[1]).to(device=device, dtype=dtype).int()

        # Params
        params = dict()

        params["bits"] = bits
        params["target"] = target
        return params

    
    def get_transform(self, params):

        transform = lambda img: posterize(img, bits=params["bits"])

        return transform
    
    def apply_img(self, img, transform, target):

        out = img.clone()

        out[target] = transform(img)[target]

        return out
    
    def apply_kpts(self, kpts, transform, target, img_size):

        return kpts
    
    def apply_bbx(self, bbx, transform, target):

        return bbx


class RandomSharpness(AugmentationBase):
    """
        Sharpen given tensor image or a batch of tensor images randomly.
    """
    def __init__(self, p, sharpness=0.5):
        super(RandomSharpness, self).__init__(p=p)
        self.sharpness = sharpness

    def generator(self, batch_size, img_size, valid_size, device, dtype):
        """
            Generate random sharpness parameters for a batch of images.
        """

        target = self._adapted_sampling(batch_size, device, dtype)

        if self.sharpness is not None:
            sharpness = torch.as_tensor(self.sharpness).to(device=device, dtype=dtype)
        else:
            sharpness = torch.as_tensor([0., 0.]).to(device=device, dtype=dtype)

        sharpness = self._adapted_uniform((batch_size,), sharpness[0], sharpness[1]).to(device=device, dtype=dtype)

        # Params
        params = dict()

        params["sharpness"] = sharpness
        params["target"] = target

        return params


    def get_transform(self, params):

        transform = lambda img: sharpness(img, sharpness_factor=params["sharpness"])

        return transform
    
    def apply_img(self, img, transform, target):

        out = img.clone()

        out[target] = transform(img)[target]

        return out
    
    def apply_kpts(self, kpts, transform, target, img_size):

        return kpts
    
    def apply_bbx(self, bbx, transform, target):

        return bbx


class RandomEqualize(AugmentationBase):
    def __init__(self, p):
        super(RandomEqualize, self).__init__(p=p)

    def generator(self, batch_size, img_size, valid_size, device, dtype):
        r"""Generate random Equalize parameters for a batch of images.

        """
        target = self._adapted_sampling(batch_size, device, dtype)

        # Params
        params = dict()

        params["target"] = target
        return params


    def get_transform(self, params):

        transform = lambda img: equalize(img)

        return transform
    
    def apply_img(self, img, transform, target):

        out = img.clone()

        out[target] = transform(img)[target]

        return out
    
    def apply_kpts(self, kpts, transform, target, img_size):

        return kpts
    
    def apply_bbx(self, bbx, transform, target):

        return bbx


class RandomGrayscale(AugmentationBase):
    r"""Applies random transformation to Grayscale according to a probability p value.
    """
    def __init__(self, p=0.1):
        super(RandomGrayscale, self).__init__(p=p)

    def generator(self, batch_size, img_size, valid_size, device, dtype):
        r"""Generate random Equalize parameters for a batch of images.

        """
        target = self._adapted_sampling(batch_size, device, dtype)

        # Params
        params = dict()

        params["target"] = target
        return params

    def get_transform(self, params):

        transform = lambda img: rgb_to_grayscale(img)

        return transform
    
    def apply_img(self, img, transform, target):

        out = img.clone()

        out[target] = transform(img)[target]

        return out
    
    def apply_kpts(self, kpts, transform, target, img_size):

        return kpts
    
    def apply_bbx(self, bbx, transform, target):

        return bbx

# --------------------------------------
#             Filters
# --------------------------------------


class RandomMotionBlur(AugmentationBase):
    """
        Perform motion blur on 2D images (4D tensor).
    """

    def __init__(self, p, kernel_size, angle, direction,
            interpolation='bilinear', border_mode='zeros', align_corners=False) -> None:
        super(RandomMotionBlur, self).__init__(p=p)

        self.kernel_size = kernel_size
        self.angle = angle
        self.direction = direction

        self.interpolation = interpolation
        self.border_mode = border_mode
        self.align_corners = align_corners

    def generator(self, batch_size, img_size, valid_size, device, dtype):
        """
            Get parameters for motion blur.
        """

        target = self._adapted_sampling(batch_size, device, dtype)

        if self.kernel_size is not None:
            kernel_size = torch.as_tensor(self.kernel_size).to(device=device, dtype=dtype)
        else:
            kernel_size = torch.as_tensor([3, 3]).to(device=device, dtype=dtype)

        if self.angle is not None:
            angle = torch.as_tensor(self.angle).to(device=device, dtype=dtype)
        else:
            angle = torch.as_tensor([0, 0]).to(device=device, dtype=dtype)

        if self.direction is not None:
            direction = torch.as_tensor(self.direction).to(device=device, dtype=dtype)
        else:
            direction = torch.as_tensor([0, 0]).to(device=device, dtype=dtype)

        kernel_size = self._adapted_uniform((1,),
                                            kernel_size[0] // 2,
                                            kernel_size[1] // 2).to(device=device, dtype=dtype).int() * 2 + 1

        angle = self._adapted_uniform((batch_size,), angle[0], angle[1]).to(device=device, dtype=dtype)
        direction = self._adapted_uniform((batch_size,), direction[0], direction[1]).to(device=device, dtype=dtype)

        # Params
        params = dict()

        params["kernel_size"] = kernel_size
        params["angle"] = angle
        params["direction"] = direction

        params["target"] = target

        return params

    def get_transform(self, params):

        transform = lambda img: motion_blur(img, 
                                            kernel_size=params["kernel_size"],
                                            angle=params["angle"],
                                            direction=params["direction"])

        return transform
    
    def apply_img(self, img, transform, target):

        out = img.clone()

        out[target] = transform(img)[target]

        return out
    
    def apply_kpts(self, kpts, transform, target, img_size):

        return kpts
    
    def apply_bbx(self, bbx, transform, target):

        return bbx

# --------------------------------------
#             Not used
# --------------------------------------


class RandomHorizontalFlip(AugmentationBase):
    """
        Applies a random horizontal flip to a batch of tensor images with a given probability.
    """

    def __init__(self, p=0.2):
        super(RandomHorizontalFlip, self).__init__(p=p)

    def generator(self, batch_size, img_size, device, dtype):

        target = self._adapted_sampling(batch_size, device, dtype)

        # params
        params = dict()
        params["target"] = target
        return params


    def get_transform(self, params):
        return None

    def apply_img(self, img, transform, target):

        out = img.clone()

        out[target] = hflip(img)[target]
        
        return out


class RandomVerticalFlip(AugmentationBase):

    r"""Applies a random vertical flip to a tensor image or a batch of tensor images with a given probability.
    """
    def __init__(self, p=0.2):
        super(RandomVerticalFlip, self).__init__(p=p)

    def generator(self, batch_size, img_size, device, dtype):
        target = self._adapted_sampling(batch_size, device, dtype)

        # params
        params = dict()
        params["target"] = target
        return params

    def apply(self, inp, params):
        out = inp.clone()
        target = params["target"]
        out[target] = vflip(inp)[target]
        return out











