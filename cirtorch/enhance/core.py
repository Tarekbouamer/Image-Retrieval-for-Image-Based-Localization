import torch
import torch.nn as nn


__all__ = [
    "add_weighted"]


def add_weighted(src1, alpha, src2, beta, gamma):
    """
        Calculates the weighted sum of two Tensors.
    """
    if not isinstance(src1, torch.Tensor):
        raise TypeError("src1 should be a tensor. Got {}".format(type(src1)))

    if not isinstance(src2, torch.Tensor):
        raise TypeError("src2 should be a tensor. Got {}".format(type(src2)))

    if not isinstance(alpha, float):
        raise TypeError("alpha should be a float. Got {}".format(type(alpha)))

    if not isinstance(beta, float):
        raise TypeError("beta should be a float. Got {}".format(type(beta)))

    if not isinstance(gamma, float):
        raise TypeError("gamma should be a float. Got {}".format(type(gamma)))

    return src1 * alpha + src2 * beta + gamma

