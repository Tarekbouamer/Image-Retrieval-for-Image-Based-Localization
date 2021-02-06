import random

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as tfn

from cirtorch.geometry.epipolar.projection import scale_intrinsics
class ISSTransform:

    def __init__(self, shortest_size=None, longest_max_size=None, random_flip=False, random_scale=None):

        self.shortest_size = shortest_size
        self.longest_max_size = longest_max_size
        self.random_scale = random_scale

    def _adjusted_scale(self, in_width, in_height, target_size):
        min_size = min(in_width, in_height)
        max_size = max(in_width, in_height)

        scale = target_size / min_size

        if int(max_size * scale) > self.longest_max_size:

            scale = self.longest_max_size / max_size

        return scale

    def _random_target_size(self):
        if len(self.random_scale) == 2:
            target_size = random.uniform(self.shortest_size * self.random_scale[0],
                                         self.shortest_size * self.random_scale[1])
        else:
            target_sizes = [self.shortest_size * scale for scale in self.random_scale]
            target_size = random.choice(target_sizes)

        return int(target_size)

    def __call__(self, inp):

        # Adjust scale, possibly at random
        if self.random_scale is not None:
            target_size = self._random_target_size()
        else:
            target_size = self.longest_max_size

        scale_1 = self._adjusted_scale(inp["img1"].size[0], inp["img1"].size[1], target_size)
        scale_2 = self._adjusted_scale(inp["img2"].size[0], inp["img2"].size[1], target_size)

        out_size_1 = tuple(int(dim * scale_1) for dim in inp["img1"].size)
        out_size_2 = tuple(int(dim * scale_2) for dim in inp["img2"].size)

        img_1 = inp["img1"].resize(out_size_1, resample=Image.BILINEAR)
        img_2 = inp["img2"].resize(out_size_2, resample=Image.BILINEAR)
        
        # Image transformations
        inp["img1"] = tfn.to_tensor(img_1)
        inp["img2"] = tfn.to_tensor(img_2)

        # Transform intrinsics to tensor
        inp["intrinsics1"] = torch.from_numpy(inp["intrinsics1"].astype(np.float))
        inp["intrinsics2"] = torch.from_numpy(inp["intrinsics2"].astype(np.float))
        
        # Scale intrinsics
        inp["intrinsics1"] = scale_intrinsics(inp["intrinsics1"], scale_1)
        inp["intrinsics2"] = scale_intrinsics(inp["intrinsics2"], scale_2)

        # Transform extrinsics to tensor
        inp["extrinsics1"] = torch.from_numpy(inp["extrinsics1"].astype(np.float))
        inp["extrinsics2"] = torch.from_numpy(inp["extrinsics2"].astype(np.float))

        # Transform keypoints to tensor
        inp["kpts"] = torch.from_numpy(inp["kpts"].astype(np.float))

        return inp


class ISSTestTransform:
    def __init__(self,
                 shortest_size=None,
                 longest_max_size=None,
                 random_scale=None):

        self.shortest_size = shortest_size
        self.longest_max_size = longest_max_size
        self.random_scale = random_scale

    def _adjusted_scale(self, in_width, in_height):
        min_size = min(in_width, in_height)
        max_size = max(in_width, in_height)

        window = self.shortest_size * self.random_scale
        scale = 1.0

        # resize to mean size if out of window
        if int(min_size) > window[1] or int(min_size) < window[0]:
            scale = self.shortest_size / min_size

        # resize to max if longer
        if int(max_size * scale) > self.longest_max_size:
            scale = self.longest_max_size / max_size

        return scale

    def __call__(self, img, bbx=None):

        # Crop  bbx
        if bbx is not None:
            img = img.crop(box=bbx)

        if self.shortest_size:

            scale = self._adjusted_scale(img.size[0], img.size[1])

            out_size = tuple(int(dim * scale) for dim in img.size)
            img = img.resize(out_size, resample=Image.BILINEAR)

        # Image transformations
        img = tfn.to_tensor(img)

        return dict(img=img)
