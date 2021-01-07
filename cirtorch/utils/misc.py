import io
from collections import OrderedDict
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as functional

from inplace_abn import InPlaceABN, InPlaceABNSync, ABN

from . import scheduler as lr_scheduler

NORM_LAYERS = [ABN, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm]
OTHER_LAYERS = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]


class Empty(Exception):
    """Exception to facilitate handling of empty predictions, annotations etc."""
    pass


class GlobalAvgPool2d(nn.Module):
    """Global average pooling over the input's spatial dimensions"""

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


class Interpolate(nn.Module):
    """nn.Module wrapper to nn.functional.interpolate"""

    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return functional.interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)


class ActivatedAffine(ABN):
    """Drop-in replacement for ABN which performs inference-mode BN + activation"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, activation="leaky_relu",
                 activation_param=0.01):
        super(ActivatedAffine, self).__init__(num_features, eps, momentum, affine, activation, activation_param)

    @staticmethod
    def _broadcast_shape(x):
        out_size = []
        for i, s in enumerate(x.size()):
            if i != 1:
                out_size.append(1)
            else:
                out_size.append(s)
        return out_size

    def forward(self, x):
        inv_var = torch.rsqrt(self.running_var + self.eps)
        if self.affine:
            alpha = self.weight * inv_var
            beta = self.bias - self.running_mean * alpha
        else:
            alpha = inv_var
            beta = - self.running_mean * alpha

        x.mul_(alpha.view(self._broadcast_shape(x)))
        x.add_(beta.view(self._broadcast_shape(x)))

        if self.activation == "relu":
            return functional.relu(x, inplace=True)
        elif self.activation == "leaky_relu":
            return functional.leaky_relu(x, negative_slope=self.activation_param, inplace=True)
        elif self.activation == "elu":
            return functional.elu(x, alpha=self.activation_param, inplace=True)
        elif self.activation == "identity":
            return x
        else:
            raise RuntimeError("Unknown activation function {}".format(self.activation))


class ActivatedGroupNorm(ABN):
    """GroupNorm + activation function compatible with the ABN interface"""

    def __init__(self, num_channels, num_groups, eps=1e-5, affine=True, activation="leaky_relu", activation_param=0.01):
        super(ActivatedGroupNorm, self).__init__(num_channels, eps, affine=affine, activation=activation,
                                                 activation_param=activation_param)
        self.num_groups = num_groups

        # Delete running mean and var since they are not used here
        delattr(self, "running_mean")
        delattr(self, "running_var")

    def reset_parameters(self):
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = functional.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)

        if self.activation == "relu":
            return functional.relu(x, inplace=True)
        elif self.activation == "leaky_relu":
            return functional.leaky_relu(x, negative_slope=self.activation_param, inplace=True)
        elif self.activation == "elu":
            return functional.elu(x, alpha=self.activation_param, inplace=True)
        elif self.activation == "identity":
            return x
        else:
            raise RuntimeError("Unknown activation function {}".format(self.activation))


def config_to_string(config):
    with io.StringIO() as sio:
        config.write(sio)
        config_str = sio.getvalue()
    return config_str


def scheduler_from_config(scheduler_config, optimizer, epoch_length):
    assert scheduler_config["type"] in ("linear", "step", "poly", "multistep")

    params = scheduler_config.getstruct("params")

    if scheduler_config["type"] == "linear":
        if scheduler_config["update_mode"] == "batch":
            count = epoch_length * scheduler_config.getint("epochs")
        else:
            count = scheduler_config.getint("epochs")

        beta = float(params["from"])
        alpha = float(params["to"] - beta) / count

        scheduler = lr_scheduler.LambdaLR(optimizer,
                                          lambda it: it * alpha + beta)

    elif scheduler_config["type"] == "step":
        scheduler = lr_scheduler.StepLR(optimizer,
                                        params["step_size"],
                                        params["gamma"])

    elif scheduler_config["type"] == "poly":
        if scheduler_config["update_mode"] == "batch":
            count = epoch_length * scheduler_config.getint("epochs")
        else:
            count = scheduler_config.getint("epochs")
        scheduler = lr_scheduler.LambdaLR(optimizer,
                                          lambda it: (1 - float(it) / count) ** params["gamma"])

    elif scheduler_config["type"] == "multistep":
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             params["milestones"],
                                             params["gamma"])

    else:
        raise ValueError("Unrecognized scheduler type {}, valid options: 'linear', 'step', 'poly', 'multistep'"
                         .format(scheduler_config["type"]))

    if scheduler_config.getint("burn_in_steps") != 0:
        scheduler = lr_scheduler.BurnInLR(scheduler,
                                          scheduler_config.getint("burn_in_steps"),
                                          scheduler_config.getfloat("burn_in_start"))

    return scheduler


def norm_act_from_config(body_config):
    """Make normalization + activation function from configuration

    Available normalization modes are:
      - `bn`: Standard In-Place Batch Normalization
      - `syncbn`: Synchronized In-Place Batch Normalization
      - `syncbn+bn`: Synchronized In-Place Batch Normalization in the "static" part of the network, Standard In-Place
        Batch Normalization in the "dynamic" parts
      - `gn`: Group Normalization
      - `syncbn+gn`: Synchronized In-Place Batch Normalization in the "static" part of the network, Group Normalization
        in the "dynamic" parts
      - `off`: No normalization (preserve scale and bias parameters)

    The "static" part of the network includes the backbone, FPN and semantic segmentation components, while the
    "dynamic" part of the network includes the RPN, detection and instance segmentation components. Note that this
    distinction is due to historical reasons and for back-compatibility with the CVPR2019 pre-trained models.

    Parameters
    ----------
    body_config
        Configuration object containing the following fields: `normalization_mode`, `activation`, `activation_slope`
        and `gn_groups`

    Returns
    -------
    norm_act_static : callable
        Function that returns norm_act modules for the static parts of the network
    norm_act_dynamic : callable
        Function that returns norm_act modules for the dynamic parts of the network
    """
    mode = body_config["normalization_mode"]
    activation = body_config["activation"]
    slope = body_config.getfloat("activation_slope")
    groups = body_config.getint("gn_groups")

    if mode == "bn":
        norm_act_static = norm_act_dynamic = partial(InPlaceABN, activation=activation, activation_param=slope)

    elif mode == "syncbn":
        norm_act_static = norm_act_dynamic = partial(InPlaceABNSync, activation=activation, activation_param=slope)

    elif mode == "syncbn+bn":
        norm_act_static = partial(InPlaceABNSync, activation=activation, activation_param=slope)
        norm_act_dynamic = partial(InPlaceABN, activation=activation, activation_param=slope)

    elif mode == "gn":
        norm_act_static = norm_act_dynamic = partial(
            ActivatedGroupNorm, num_groups=groups, activation=activation, activation_param=slope)

    elif mode == "syncbn+gn":
        norm_act_static = partial(InPlaceABNSync, activation=activation, activation_param=slope)
        norm_act_dynamic = partial(ActivatedGroupNorm, num_groups=groups, activation=activation, activation_param=slope)

    elif mode == "off":
        norm_act_static = norm_act_dynamic = partial(ActivatedAffine, activation=activation, activation_param=slope)

    else:
        raise ValueError("Unrecognized normalization_mode {}, valid options: 'bn', 'syncbn', 'syncbn+bn', 'gn', "
                         "'syncbn+gn', 'off'".format(mode))

    return norm_act_static, norm_act_dynamic


def freeze_params(module):
    """Freeze all parameters of the given module"""
    for p in module.parameters():
        p.requires_grad_(False)


def all_reduce_losses(losses):
    """Coalesced mean all reduce over a dictionary of 0-dimensional tensors"""
    names, values = [], []
    for k, v in losses.items():
        names.append(k)
        values.append(v)

    # Peform the actual coalesced all_reduce
    values = torch.cat([v.view(1) for v in values], dim=0)
    dist.all_reduce(values, dist.ReduceOp.SUM)
    values.div_(dist.get_world_size())
    values = torch.chunk(values, values.size(0), dim=0)

    # Reconstruct the dictionary
    return OrderedDict((k, v.view(())) for k, v in zip(names, values))


def try_index(scalar_or_list, i):
    try:
        return scalar_or_list[i]
    except TypeError:
        return scalar_or_list
