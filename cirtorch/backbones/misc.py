from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as functional

from inplace_abn import ABN


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


class ResidualBlock(nn.Module):
    """Configurable residual block

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : list of int
        Number of channels in the internal feature maps. Can either have two or three elements: if three construct
        a residual block with two `3 x 3` convolutions, otherwise construct a bottleneck block with `1 x 1`, then
        `3 x 3` then `1 x 1` convolutions.
    stride : int
        Stride of the first `3 x 3` convolution
    dilation : int
        Dilation to apply to the `3 x 3` convolutions.
    groups : int
        Number of convolution groups. This is used to create ResNeXt-style blocks and is only compatible with
        bottleneck blocks.
    norm_act : callable
        Function to create normalization / activation Module.
    dropout: callable
        Function to create Dropout Module.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 stride=1,
                 dilation=1,
                 groups=1,
                 norm_act=ABN,
                 dropout=None):
        super(ResidualBlock, self).__init__()

        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        if not is_bottleneck:
            bn2 = norm_act(channels[1])
            bn2.activation = "identity"
            layers = [
                ("conv1", nn.Conv2d(in_channels, channels[0], 3, stride=stride, padding=dilation, bias=False,
                                    dilation=dilation)),
                ("bn1", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False,
                                    dilation=dilation)),
                ("bn2", bn2)
            ]
            if dropout is not None:
                layers = layers[0:2] + [("dropout", dropout())] + layers[2:]
        else:
            bn3 = norm_act(channels[2])
            bn3.activation = "identity"
            layers = [
                ("conv1", nn.Conv2d(in_channels, channels[0], 1, stride=1, padding=0, bias=False)),
                ("bn1", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0], channels[1], 3, stride=stride, padding=dilation, bias=False,
                                    groups=groups, dilation=dilation)),
                ("bn2", norm_act(channels[1])),
                ("conv3", nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False)),
                ("bn3", bn3)
            ]
            if dropout is not None:
                layers = layers[0:4] + [("dropout", dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))

        if need_proj_conv:
            self.proj_conv = nn.Conv2d(in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)
            self.proj_bn = norm_act(channels[-1])
            self.proj_bn.activation = "identity"

    def forward(self, x):
        if hasattr(self, "proj_conv"):
            residual = self.proj_conv(x)

            residual = self.proj_bn(residual)
        else:
            residual = x

        x = self.convs(x) + residual

        if self.convs.bn1.activation == "relu":
            return functional.relu(x, inplace=True)
        elif self.convs.bn1.activation == "leaky_relu":
            return functional.leaky_relu(x, negative_slope=self.convs.bn1.activation_param, inplace=True)
        elif self.convs.bn1.activation == "elu":
            return functional.elu(x, alpha=self.convs.bn1.activation_param, inplace=True)
        elif self.convs.bn1.activation == "identity":
            return x
        else:
            raise RuntimeError("Unknown activation function {}".format(self.activation))


class IdentityResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 stride=1,
                 dilation=1,
                 groups=1,
                 norm_act=ABN,
                 dropout=None):
        """Configurable identity-mapping residual block
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        channels : list of int
            Number of channels in the internal feature maps. Can either have two or three elements: if three construct
            a residual block with two `3 x 3` convolutions, otherwise construct a bottleneck block with `1 x 1`, then
            `3 x 3` then `1 x 1` convolutions.
        stride : int
            Stride of the first `3 x 3` convolution
        dilation : int
            Dilation to apply to the `3 x 3` convolutions.
        groups : int
            Number of convolution groups. This is used to create ResNeXt-style blocks and is only compatible with
            bottleneck blocks.
        norm_act : callable
            Function to create normalization / activation Module.
        dropout: callable
            Function to create Dropout Module.
        """
        super(IdentityResidualBlock, self).__init__()

        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [
                ("conv1", nn.Conv2d(in_channels, channels[0], 3, stride=stride, padding=dilation, bias=False,
                                    dilation=dilation)),
                ("bn2", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False,
                                    dilation=dilation))
            ]
            if dropout is not None:
                layers = layers[0:2] + [("dropout", dropout())] + layers[2:]
        else:
            layers = [
                ("conv1", nn.Conv2d(in_channels, channels[0], 1, stride=stride, padding=0, bias=False)),
                ("bn2", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False,
                                    groups=groups, dilation=dilation)),
                ("bn3", norm_act(channels[1])),
                ("conv3", nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False))
            ]
            if dropout is not None:
                layers = layers[0:4] + [("dropout", dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))

        if need_proj_conv:
            self.proj_conv = nn.Conv2d(in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        if hasattr(self, "proj_conv"):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x.clone()
            bn1 = self.bn1(x)

        out = self.convs(bn1)
        out.add_(shortcut)

        return out


class DenseModule(nn.Module):
    def __init__(self, in_channels, growth, layers, bottleneck_factor=4, norm_act=ABN, dilation=1):
        super(DenseModule, self).__init__()
        self.in_channels = in_channels
        self.growth = growth
        self.layers = layers

        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for i in range(self.layers):
            self.convs1.append(nn.Sequential(OrderedDict([
                ("bn", norm_act(in_channels)),
                ("conv", nn.Conv2d(in_channels, self.growth * bottleneck_factor, 1, bias=False))
            ])))
            self.convs2.append(nn.Sequential(OrderedDict([
                ("bn", norm_act(self.growth * bottleneck_factor)),
                ("conv", nn.Conv2d(self.growth * bottleneck_factor, self.growth, 3, padding=dilation, bias=False,
                                   dilation=dilation))
            ])))
            in_channels += self.growth

    @property
    def out_channels(self):
        return self.in_channels + self.growth * self.layers

    def forward(self, x):
        inputs = [x]
        for i in range(self.layers):
            x = torch.cat(inputs, dim=1)
            x = self.convs1[i](x)
            x = self.convs2[i](x)
            inputs += [x]

        return torch.cat(inputs, dim=1)