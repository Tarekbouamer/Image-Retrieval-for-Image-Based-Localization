from collections import OrderedDict
import torch
import torch.nn as nn

from torchvision import transforms

from PIL import ImageDraw

from cirtorch.utils.parallel import PackedSequence
from cirtorch.utils.sequence import pad_packed_images


from .augmentation import (
    Centralize,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomPosterize,
    RandomSharpness,
    RandomEqualize,
    RandomRotation,
    RandomSolarize,
    RandomAffine,
    RandomPerspective,
    RandomMotionBlur,
    RandomVerticalFlip,
    ColorJitter
)

from cirtorch.geometry.linalg import eye_like

from cirtorch.utils.image import normalize_min_max, denormalize_min_max, normalize


class RandomAugmentation(nn.Module):
    """ 
        Random Augmentation that takes packed sequence and perform sequence of 
        transformations of selected images in batch based Bernoulli distribution. 
        The module  translates all padded images to the center of the max-image
        in the input batch and normalize every image based on dataset mean and std. 
        we guarantee a better control in photometric and geometric operations.
    """
    def __init__(self, rgb_mean, rgb_std):
        super(RandomAugmentation, self).__init__()

        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std

        self.centralize = nn.Sequential(OrderedDict([ 
            ('Centralize', Centralize())
        ]))

        self.photometric = nn.Sequential(OrderedDict([
            ('ColorJitter', ColorJitter(p=0.3, brightness=[0.0, 0.2], contrast=[0.8, 1.0], saturation=[0.3, 0.1], hue=[0.0, 0.5])),
            ('Grayscale', RandomGrayscale(p=0.3)), 
            ('Posterize', RandomPosterize(p=0.3, bits=[4, 8])),

            #('Solarize', RandomSolarize(p=1.0, thresholds=[0.95, 1.0], additions=None)),
            #('Sharpness', RandomSharpness(p=1.0, sharpness=[0.0, 1.0])),
            #('Equalize', RandomEqualize(p=1.0)),

        ]))

        self.geometric = nn.Sequential(OrderedDict([
            ('Perspective', RandomPerspective(p=1.0, distortion_scale=0.3)),
            ('Affine', RandomAffine(p=1.0, theta=30, h_trans=0.0, v_trans=0.0, scale=[0.8, 1.6], shear=[0.0, 0.3])),
            ('Rotation', RandomRotation(p=1.0, theta=60))
        ]
        ))

        self.filter = nn.Sequential(OrderedDict([
            ('MotionBlur', RandomMotionBlur(p=1.0, kernel_size=[5, 11], angle=[0, 90], direction=[-1, 1]))
        ]))

    def show(self, inp, out):

        for (inp_img, inp_kpt, out_img, out_kpt) in zip(inp["img"], inp["kpts"], out["img"], out["kpts"]):
            
            inp_img = transforms.ToPILImage()(inp_img.cpu()).convert("RGB")
            out_img = transforms.ToPILImage()(out_img.cpu()).convert("RGB")

            inp_draw = ImageDraw.Draw(inp_img)
            out_draw = ImageDraw.Draw(out_img)
            
            inp_kpt = inp_kpt.view(-1).int().tolist()
            out_kpt = out_kpt.view(-1).int().tolist()
            
            # Draw points
            inp_draw.point(inp_kpt, fill=255)
            out_draw.point(out_kpt, fill=255)
            
            # Plot
            inp_img.show()
            out_img.show()

    def _unpack_input(self, inp):
        """
            PackedSequence List(C W H )--> Tensors B C W H
        """
        if isinstance(inp, dict):

            # Pad the input images
            inp["img"], inp["valid_size"] = pad_packed_images(inp["img"])
            
            # add keypoints if available  
            if inp.__contains__("kpts"):
                inp["kpts"], _ = inp["kpts"].contiguous

            # add keypoints if available  
            if inp.__contains__("bbx"):
                inp["bbx"], _ = inp["bbx"].contiguous
            else:
                size = [torch.tensor([item[0], item[1]]) for item in inp["valid_size"]]
                size = torch.stack(size).to(device=inp["img"].device, dtype=inp["img"].dtype)
                
                zeros = torch.zeros_like(size)
                
                # from B*4 to B*N*4  where N=1 for this case
                inp["bbx"] = torch.cat((zeros, size), dim=1).unsqueeze(1)

            if not inp.__contains__("transform"):
                inp["transform"] = eye_like(n=3, input=inp["img"])            
            return inp
        

        else:
            raise TypeError("Input type is not a PackedSequence Got {}".format(type(input)))
    
    def pack_output(self, out):
        """
            Tensors B C W H  --- > PackedSequence List(* W H )
        """
        for key, items in out.items():
            if isinstance(items[0], torch.Tensor):
                out[key] = PackedSequence([item for item in items])
            else:
                out[key] = items

    def _msk(self, img, bbx):
        """
            zero out again the padded region, and discard the effect of photometric and filtering operations out.
        """ 
        #TODO: move this operation insidde augmentaion and perform it after each operation, for better masking 
        # Note the function does not support batched boxes
        msk = torch.zeros_like(img)
        for msk_i, bbx_i in zip(msk, bbx):

            kernel = torch.ones(int(bbx_i[2]), int(bbx_i[3]))
            msk_i[:, int(bbx_i[0]): int(bbx_i[0]+bbx_i[2]), int(bbx_i[1]): int(bbx_i[1]+bbx_i[3])] = kernel

        return img * msk

    def forward(self, inp, do_augmentation=False):

        # Input dictionary that contains img, kpt, bbx 
        out = self._unpack_input(inp)

        # translate pure image to center
        #out = self.centralize(out)

        if do_augmentation:

            # normalize min_max !better photometric augmentation control
            out["img"], x_min, x_max = normalize_min_max(out["img"])
            
            # run augmentation
            #out = self.geometric(out)
            #out = self.photometric(out)
            #out = self.filter(out)

            # denormalize min_max
            out["img"] = denormalize_min_max(out["img"], x_max=x_max, x_min=x_min)

        # normalize based on mean and std
        out["img"] = normalize(out["img"], mean=self.rgb_mean, std=self.rgb_std)

        # show
        #self.show(inp=tmp, out=out)

        # Pack output
        self.pack_output(out)
        
        return out
