from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as functional
from inplace_abn import ABN

from ..backbones.misc import GlobalAvgPool2d


class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=16, norm_act=ABN):
        super(SELayer, self).__init__()

        layers = [
            ("avg_pool", GlobalAvgPool2d()),

            ("fc_down", nn.Linear(in_channels, in_channels // reduction)),
            ("relu", functional.relu(inplace=True)),

            ("fc_up", nn.Linear(in_channels // reduction, in_channels)),
            ("sigmoid", functional.sigmoid())
        ]

        self.se = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        y = self.se(x)
        return x * y.expand_as(x)

