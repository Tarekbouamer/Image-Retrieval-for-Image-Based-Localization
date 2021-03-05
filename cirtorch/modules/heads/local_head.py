from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as functional

from functools import partial
from inplace_abn import ABN

from cirtorch.utils.misc import try_index

from cirtorch.modules.pools import POOLING_LAYERS
from cirtorch.modules.normalizations import NORMALIZATION_LAYERS

from cirtorch.geometry.conversions import (
    normalize_pixel_coordinates, 
    denormalize_pixel_coordinates)

class localHead(nn.Module):
    """
        Local Feature head 
    """

    def __init__(self, dim, embedding_size=None, norm_act=ABN):
        super(localHead, self).__init__()

        self.whiten = nn.Linear(dim, embedding_size, bias=True)
        self.norm = functional.normalize

        self.reset_parameters()

    def reset_parameters(self):
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear):
                    nn.init.xavier_normal_(mod.weight, 0.1)

            elif isinstance(mod, ABN):
                nn.init.constant_(mod.weight, 1.)

            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.constant_(mod.bias, 0.)

    def keypoints_sampling(self, x, kpts_norm):    
        """
            sample from normalized coordinates

            x: [B C H W]
            kpts_norm : [B N 2]
        """
        
        x_desc = functional.grid_sample(x, kpts_norm.unsqueeze(2), mode="bilinear", padding_mode="zeros")
        x_desc = x_desc.squeeze(-1)
        
        return x_desc

        
    def forward(self, x, kpts=None, img_size=None):
        """
            Local features head for FPN
        """

        # Sample
        desc = self.keypoints_sampling(x=x, kpts_norm=kpts)

        desc = desc.permute(0, 2, 1)
        
        desc = self.whiten(desc)

        desc = self.norm(desc, dim=2)

        return desc
