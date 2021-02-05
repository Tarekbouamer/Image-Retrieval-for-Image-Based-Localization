import torch
from os import path
from cirtorch.utils.parallel import PackedSequence


def iss_collate_fn(items):
    """Collate function for ISS batches"""
    out = {}
    if len(items) > 0:
        for key in items[0]:
            out[key] = [item[key] for item in items]
            if isinstance(items[0][key], torch.Tensor):
                out[key] = PackedSequence(out[key])
    return out


def cid2filename(cid, prefix):
    """
    Creates a training image path out of its CID name

    Arguments
    ---------
    cid      : name of the image
    prefix   : root directory where images are saved

    Returns
    -------
    filename : full image filename
    """
    return path.join(prefix, cid[-2:], cid[-4:-2], cid[-6:-4], cid)
