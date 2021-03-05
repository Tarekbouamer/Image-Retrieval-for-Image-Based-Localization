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

    def _adjust_points(self, kpts, scale):
        return (kpts * scale).astype(int)

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
        inp["intrinsics1"] = torch.from_numpy(inp["intrinsics1"]).float()
        inp["intrinsics2"] = torch.from_numpy(inp["intrinsics2"]).float()
        
        # Scale intrinsics
        inp["intrinsics1"] = scale_intrinsics(inp["intrinsics1"], scale_1).unsqueeze(0)
        inp["intrinsics2"] = scale_intrinsics(inp["intrinsics2"], scale_2).unsqueeze(0)

        # Transform extrinsics to tensor
        inp["extrinsics1"] = torch.from_numpy(inp["extrinsics1"]).float().unsqueeze(0)
        inp["extrinsics2"] = torch.from_numpy(inp["extrinsics2"]).float().unsqueeze(0)

        # Scale and Transform keypoints to tensor
        scaled_kpts = self._adjust_points(inp["kpts"], scale_1)
        inp["kpts"] = torch.from_numpy(scaled_kpts).float().unsqueeze(0)

        return inp


class ISSTestTransform:
    def __init__(self,
                 shortest_size=None,
                 longest_max_size=None):

        self.shortest_size = shortest_size
        self.longest_max_size = longest_max_size

    def _adjusted_scale(self, in_width, in_height):
        min_size = min(in_width, in_height)
        scale = self.shortest_size / min_size
        return scale

    def _adjust_points(self, kpts, scale):
        return (kpts * scale).astype(int)

    def _adjust_homography(self, H, scale1, scale2):

        # Get the scaling matrices
        scale1_matrix = np.diag([1. / scale1, 1. / scale1, 1.]).astype(float)
        scale2_matrix = np.diag([1. / scale2, 1. / scale2, 1.]).astype(float)
        
        H = scale2_matrix @ H @ np.linalg.inv(scale1_matrix)
        H_inv = np.linalg.inv(H)
        
        return H, H_inv

    def __call__(self, out):

        # Adjust scale
        scale1 = self._adjusted_scale(out["img1"].size[0], out["img1"].size[1])
        scale2 = self._adjusted_scale(out["img2"].size[0], out["img2"].size[1])

        out_size = tuple(int(dim * scale1) for dim in out["img1"].size)
        out["img1"] = out["img1"].resize(out_size, resample=Image.BILINEAR)

        out_size = tuple(int(dim * scale2) for dim in out["img2"].size)
        out["img2"] = out["img2"].resize(out_size, resample=Image.BILINEAR)

        # Scale keypoints
        out["kpts1"] = self._adjust_points(out["kpts1"], scale1)
        out["kpts2"] = self._adjust_points(out["kpts2"], scale2)

        # Scale homographies
        out["H"], out["H_inv"] = self._adjust_homography(out["H"], scale1, scale2)

        # Image transformations
        out["img1"] = tfn.to_tensor(out["img1"])
        out["img2"] = tfn.to_tensor(out["img2"])

        out["kpts1"]  = torch.from_numpy(out["kpts1"] ).float().unsqueeze(0)
        out["kpts2"]  = torch.from_numpy(out["kpts2"] ).float().unsqueeze(0)

        out["H"]  = torch.from_numpy(out["H"] ).float().unsqueeze(0)
        out["H_inv"]  = torch.from_numpy(out["H_inv"] ).float().unsqueeze(0)

        return out
