# general utils used for all models
import torch.nn as nn
from inplace_abn import ABN



def try_index(scalar_or_list, i):
    try:
        return scalar_or_list[i]
    except TypeError:
        return scalar_or_list


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

CONV_PARAMS = ["weight"]
BN_PARAMS = ["weight", "bias", "running_mean", "running_var"]
FC_PARAMS = ["weight", "bias"]

def copy_layer(inm, outm, name_in, name_out, params):
    for param_name in params:
        #print( outm[name_out + "." + param_name])
        print( inm[name_in + "." + param_name])
        outm[name_out + "." + param_name] = inm[name_in + "." + param_name]


def convert(model, structure, bottleneck):
    out = dict()
    num_convs = 3 if bottleneck else 2

    # Initial module
    copy_layer(model, out, "conv1", "mod1.conv1", CONV_PARAMS)
    copy_layer(model, out, "bn1", "mod1.bn1", BN_PARAMS)

    # Other modules
    for mod_id, num in enumerate(structure):
        for block_id in range(num):
            for conv_id in range(num_convs):
                copy_layer(model, out,
                           "layer{}.{}.conv{}".format(mod_id + 1, block_id, conv_id + 1),
                           "mod{}.block{}.convs.conv{}".format(mod_id + 2, block_id + 1, conv_id + 1),
                           CONV_PARAMS)
                copy_layer(model, out,
                           "layer{}.{}.bn{}".format(mod_id + 1, block_id, conv_id + 1),
                           "mod{}.block{}.convs.bn{}".format(mod_id + 2, block_id + 1, conv_id + 1),
                           BN_PARAMS)

            # Try copying projection module
            try:
                copy_layer(model, out,
                           "layer{}.{}.downsample.0".format(mod_id + 1, block_id),
                           "mod{}.block{}.proj_conv".format(mod_id + 2, block_id + 1),
                           CONV_PARAMS)
                copy_layer(model, out,
                           "layer{}.{}.downsample.1".format(mod_id + 1, block_id),
                           "mod{}.block{}.proj_bn".format(mod_id + 2, block_id + 1),
                           BN_PARAMS)
            except KeyError:
                pass
    return out


def init_weights(model, config):

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            init_fn = getattr(nn.init, config.get("initializer") + '_')

            # Xavier or Orthogonal
            if config.get("initializer").startswith("xavier") or config.get("initializer") == "orthogonal":

                gain = config.getfloat("weight_gain_multiplier")

                if config.get("activation") == "relu" or config.get("activation") == "elu":

                    gain *= nn.init.calculate_gain("relu")

                elif config.get("activation") == "leaky_relu":

                    gain *= nn.init.calculate_gain("leaky_relu", config.getfloat("activation_slope"))

                init_fn(m.weight, gain)

            # Kaiming He
            elif config.get("initializer").startswith("kaiming"):

                if config.get("activation") == "relu" or config.get("activation") == "elu":
                    init_fn(m.weight, 0)
                else:
                    init_fn(m.weight, config.getfloat("activation_slope"))
            # Bias
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.)

        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, ABN):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, .1)
            nn.init.constant_(m.bias, 0.)