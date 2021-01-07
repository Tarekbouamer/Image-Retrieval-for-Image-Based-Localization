import sys
from collections import OrderedDict
from functools import partial

from .util import try_index, init_weights
from .misc import GlobalAvgPool2d

import torch.nn as nn
from inplace_abn import ABN



from .util import load_state_dict_from_url
from .url import model_urls


CONV_PARAMS = ["weight", "bias"]
BN_PARAMS = ["weight", "bias"]
EXT_PARAMS = ["running_mean", "running_var"]


class VGG(nn.Module):

    def __init__(self, arch, structure, norm_act=ABN, config=None, classes=1000):
        super(VGG, self).__init__()

        self.arch = arch

        self.structure = structure

        self.bn = True if config["normalization_mode"] != "off" else False

        self.dim = 7 * 7

        self.classes = classes

        if len(self.structure) != 5:
            raise ValueError("Expected a structure with five values")

        input_channels = 3
        base_channels = 64

        # Create block layer for module
        for mod_id, num in enumerate(self.structure):

            layers = []

            for i in range(num):

                output_channels = base_channels * pow(2, mod_id)

                if output_channels > 512:
                    output_channels = 512

                layers.append(
                    ("conv%d" % (i + 1), nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)))

                if self.bn:
                    layers.append(("bn%d" % (i + 1), try_index(norm_act, 0)(output_channels)))

                else:
                    layers.append(("ac%d" % (i + 1), nn.ReLU(inplace=True)))

                input_channels = output_channels

            layers.append(("pool%d" % (i + 1), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)))

            self.add_module("mod%d" % (mod_id + 1), nn.Sequential(OrderedDict(layers)))

        self.out_channels = output_channels

        # Classifier
        if classes != 0:
            self.classifier = nn.Sequential(OrderedDict([
                ("avg_pool", GlobalAvgPool2d()),

                ("fc1", nn.Linear(output_channels * self.dim, 4096)),
                ("bn1", nn.ReLU(inplace=True)),
                ("do1", nn.Dropout(p=0.5)),

                ("fc2", nn.Linear(4096, 4096)),
                ("bn2", nn.ReLU(inplace=True)),
                ("do2", nn.Dropout(p=0.5)),

                ("fc3", nn.Linear(4096, classes)),
            ]))
            self.out_channels = 4096

    def copy_layer(self, original, out, name_in, name_out, params):
        for param_name in params:
            out[name_out + "." + param_name] = original[name_in + "." + param_name]

    def convert(self, original, bottleneck=False):
        out = dict()

        # Modules
        idx = 0
        for mod_id, num in enumerate(self.structure):

            for i in range(num):

                self.copy_layer(original, out, "features.{}".format(idx), "mod{}.conv{}".format(mod_id + 1, i + 1), CONV_PARAMS)

                if self.bn:
                    idx += 1

                    self.copy_layer(original, out, "features.{}".format(idx), "mod{}.bn{}".format(mod_id + 1, i + 1), BN_PARAMS)

                    try:
                        self.copy_layer(original, out, "features.{}".format(idx), "mod{}.bn{}".format(mod_id + 1, i + 1),
                                        EXT_PARAMS)
                    except KeyError:
                        pass

                idx += 2

            idx += 1

        # Classifier
        if self.classes != 0:

            for i in range(3):
                # Note original implementation does not have bn in classifier dict, therefore we use xsavier
                self.copy_layer(original, out, "classifier.{}".format(i * 3), "classifier.fc{}".format(i + 1), CONV_PARAMS)

                idx += 1

        return out

    def forward(self, x):
        outs = OrderedDict()

        outs["mod1"] = self.mod1(x)
        outs["mod2"] = self.mod2(outs["mod1"])
        outs["mod3"] = self.mod3(outs["mod2"])
        outs["mod4"] = self.mod4(outs["mod3"])
        outs["mod5"] = self.mod5(outs["mod4"])

        if hasattr(self, "classifier"):
            outs["classifier"] = self.classifier(outs["mod5"])

        return outs


_Structures = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

Structures = {
    'A': [1, 1, 2, 2, 2],
    'B': [2, 2, 2, 2, 2],
    'D': [2, 2, 3, 3, 3],
    'E': [2, 2, 4, 4, 4],
}

_NETS = {
    "11": {"arch": 'vgg11', "structure": Structures['A']},
    "13": {"arch": 'vgg13', "structure": Structures['B']},
    "16": {"arch": 'vgg16', "structure": Structures['D']},
    "19": {"arch": 'vgg19', "structure": Structures['E']},
}

__all__ = []

for name, params in _NETS.items():
    net_name = "vgg" + name
    setattr(sys.modules[__name__], net_name, partial(VGG, **params))
    __all__.append(net_name)
