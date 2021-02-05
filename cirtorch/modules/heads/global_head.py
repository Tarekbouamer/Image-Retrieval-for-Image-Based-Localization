from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as functional

from functools import partial
from inplace_abn import ABN

from cirtorch.utils.misc import try_index

from cirtorch.modules.pools import POOLING_LAYERS
from cirtorch.modules.normalizations import NORMALIZATION_LAYERS


class globalHead(nn.Module):
    """ImageRetrievalHead for FPN
    Parameters
    """

    def __init__(self, pooling=None, normal=None, dim=None, norm_act=ABN):

        super(globalHead, self).__init__()

        self.dim = dim
        self.whiten = nn.Linear(dim, dim, bias=True)

        # TODO This might fail, since we are going to train in batch distribution system

        # pooling
        if pooling["name"] == "GeMmp":
            self.pool = POOLING_LAYERS[pooling["name"]](**pooling["params"], mp=self.dim)
        else:
            self.pool = POOLING_LAYERS[pooling["name"]](**pooling["params"])

        # normalization
        self.norm = NORMALIZATION_LAYERS[normal["name"]](eps=1e-6)

        self.reset_parameters()

    def reset_parameters(self):
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear):
                    nn.init.xavier_normal_(mod.weight, 0.1)

            elif isinstance(mod, ABN):
                nn.init.constant_(mod.weight, 1.)

            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.constant_(mod.bias, 0.)

    def forward(self, x, do_whitening=True):
        """ImageRetrievalHead for FPN
        Parameters
        """
        # pool and normalize
        x = self.pool(x)

        x = self.norm(x).squeeze(-1).squeeze(-1)

        # pool and normalize
        if do_whitening:
            x = self.whiten(x)
            x = self.norm(x)

        # permute
        return x.permute(1, 0)
