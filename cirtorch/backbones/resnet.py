import sys
from collections import OrderedDict
from functools import partial

import torch.nn as nn
from inplace_abn import ABN

from .misc import GlobalAvgPool2d, ResidualBlock
from .util import try_index, convert

CONV_PARAMS = ["weight"]
BN_PARAMS = ["weight", "bias", "running_mean", "running_var"]


class ResNet(nn.Module):
    """Standard residual network

    Parameters
    ----------
    structure : list of int
        Number of residual blocks in each of the four modules of the network
    bottleneck : bool
        If `True` use "bottleneck" residual blocks with 3 convolutions, otherwise use standard blocks
    norm_act : callable or list of callable
        Function to create normalization / activation Module. If a list is passed it should have four elements, one for
        each module of the network
    classes : int
        If not `0` also include globalFeatures average pooling and a fully-connected layer with `classes` outputs at the end
        of the network
    dilation : int or list of int
        List of dilation factors for the four modules of the network, or `1` to ignore dilation
    dropout : list of float or None
        If present, specifies the amount of dropout to apply in the blocks of each of the four modules of the network
    caffe_mode : bool
        If `True`, use bias in the first convolution for compatibility with the Caffe pretrained models
    """

    def __init__(self,
                 structure,
                 bottleneck,
                 norm_act=ABN,
                 config=None,
                 classes=0,
                 dilation=1,
                 dropout=None,
                 caffe_mode=False):
        super(ResNet, self).__init__()
        self.structure = structure
        self.bottleneck = bottleneck
        self.dilation = dilation
        self.dropout = dropout
        self.caffe_mode = caffe_mode

        if len(structure) != 4:
            raise ValueError("Expected a structure with four values")
        if dilation != 1 and len(dilation) != 4:
            raise ValueError("If dilation is not 1 it must contain four values")

        # Initial layers
        layers = [
            ("conv1", nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=caffe_mode)),
            ("bn1", try_index(norm_act, 0)(64))
        ]
        if try_index(dilation, 0) == 1:
            layers.append(("pool1", nn.MaxPool2d(3, stride=2, padding=1)))
        self.mod1 = nn.Sequential(OrderedDict(layers))

        # Groups of residual blocks
        in_channels = 64
        if self.bottleneck:
            channels = (64, 64, 256)
        else:
            channels = (64, 64)
        for mod_id, num in enumerate(structure):
            mod_dropout = None
            if self.dropout is not None:
                if self.dropout[mod_id] is not None:
                    mod_dropout = partial(nn.Dropout, p=self.dropout[mod_id])

            # Create blocks for module
            blocks = []
            for block_id in range(num):
                stride, dil = self._stride_dilation(dilation, mod_id, block_id)
                blocks.append((
                    "block%d" % (block_id + 1),
                    ResidualBlock(in_channels, channels, norm_act=try_index(norm_act, mod_id),
                                  stride=stride, dilation=dil, dropout=mod_dropout)
                ))

                # Update channels and p_keep
                in_channels = channels[-1]

            # Create module
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))

            # Double the number of channels for the next module
            channels = [c * 2 for c in channels]

        # Pooling and predictor
        if classes != 0:
            self.classifier = nn.Sequential(OrderedDict([
                ("avg_pool", GlobalAvgPool2d()),
                ("fc", nn.Linear(in_channels, classes))
            ]))

    @staticmethod
    def _stride_dilation(dilation, mod_id, block_id):
        d = try_index(dilation, mod_id)
        s = 2 if d == 1 and block_id == 0 and mod_id > 0 else 1
        return s, d

    def copy_layer(self, inm, outm, name_in, name_out, params):
        for param_name in params:
            outm[name_out + "." + param_name] = inm[name_in + "." + param_name]

    def convert(self, model):
        out = dict()
        num_convs = 3 if self.bottleneck else 2

        # Initial module
        self.copy_layer(model, out, "conv1", "mod1.conv1", CONV_PARAMS)
        self.copy_layer(model, out, "bn1", "mod1.bn1", BN_PARAMS)

        # Other modules
        for mod_id, num in enumerate(self.structure):
            for block_id in range(num):
                for conv_id in range(num_convs):
                    self.copy_layer(model, out,
                               "layer{}.{}.conv{}".format(mod_id + 1, block_id, conv_id + 1),
                               "mod{}.block{}.convs.conv{}".format(mod_id + 2, block_id + 1, conv_id + 1),
                               CONV_PARAMS)
                    self.copy_layer(model, out,
                               "layer{}.{}.bn{}".format(mod_id + 1, block_id, conv_id + 1),
                               "mod{}.block{}.convs.bn{}".format(mod_id + 2, block_id + 1, conv_id + 1),
                               BN_PARAMS)

                # Try copying projection module
                try:
                    self.copy_layer(model, out,
                               "layer{}.{}.downsample.0".format(mod_id + 1, block_id),
                               "mod{}.block{}.proj_conv".format(mod_id + 2, block_id + 1),
                               CONV_PARAMS)
                    self.copy_layer(model, out,
                               "layer{}.{}.downsample.1".format(mod_id + 1, block_id),
                               "mod{}.block{}.proj_bn".format(mod_id + 2, block_id + 1),
                               BN_PARAMS)
                except KeyError:
                    pass
        return out

    def forward(self, x):
        outs = OrderedDict()

        outs["mod1"] = self.mod1(x)
        outs["mod2"] = self.mod2(outs["mod1"])
        outs["mod3"] = self.mod3(outs["mod2"])

        #outs["mod4"] = self.mod4(outs["mod3"])
        #outs["mod5"] = self.mod5(outs["mod4"])

        if hasattr(self, "classifier"):
            outs["classifier"] = self.classifier(outs["mod5"])

        return outs


_NETS = {
    "18": {"structure": [2, 2, 2, 2], "bottleneck": False},
    "34": {"structure": [3, 4, 6, 3], "bottleneck": False},
    "50": {"structure": [3, 4, 6, 3], "bottleneck": True},
    "101": {"structure": [3, 4, 23, 3], "bottleneck": True},
    "152": {"structure": [3, 8, 36, 3], "bottleneck": True},
}

__all__ = []

for name, params in _NETS.items():
    net_name = "resnet" + name
    setattr(sys.modules[__name__], net_name, partial(ResNet, **params))
    __all__.append(net_name)
