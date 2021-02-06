import torch
import torch.nn as nn

from collections import OrderedDict

from cirtorch.utils.sequence import pad_packed_images
from cirtorch.utils.parallel import PackedSequence

NETWORK_INPUTS = ['img1', 'img2', 'intrinsics1', 'intrinsics2', 'extrinsics1', 'extrinsics2', 'kpts']


class localNet(nn.Module):

    def __init__(self, body, local_algo, local_head, augment=None):
        super(localNet, self).__init__()
        self.augment = augment
        self.body = body

        self.local_algo = local_algo
        self.local_head = local_head

    def forward(self, img1=None, img2=None, intrinsics1=None, intrinsics2=None, extrinsics1=None, extrinsics2=None, kpts=None,
                do_augmentaton=False, do_loss=False, do_prediction=True, **varargs):

        # Perform augmentation if true and exists
        #if self.augment:
            #img = self.augment(img, masking=True, do_augmentation=do_augmentaton)

        # Pad the input images
        img1, valid_size_1 = pad_packed_images(img1)
        img_size_1 = img1.shape[-2:]

        img2, valid_size_2 = pad_packed_images(img2)
        img_size_2 = img2.shape[-2:]

        # Run network body
        x1 = self.body(img1)
        x2 = self.body(img2)

        # Run local Net
        if do_loss:
            ret_loss, ret_pred = self.local_algo.training(self.local_head, x1, x2, kpts, img_size_1, img_size_2)

        elif do_prediction:
            ret_pred = self.ret_algo.inference(self.ret_head, x, valid_size)
            ret_loss = None

        else:
            ret_pred, ret_loss = None, None

        # Prepare outputs
        loss = OrderedDict([
            ("ret_loss", ret_loss)
        ])

        pred = OrderedDict([
            ("ret_pred", ret_pred)
        ])

        return loss, pred
