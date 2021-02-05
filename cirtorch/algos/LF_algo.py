import numpy as np
import torch
import torch.nn.functional as functional
from inplace_abn import active_group, set_active_group
from random import randint
from cirtorch.modules.losses import triplet_loss, contrastive_loss
from cirtorch.utils.misc import Empty
from cirtorch.utils.parallel import PackedSequence


class localFeatureLoss:
    """Image Retrieval loss

    """
    def __init__(self, name=None, sigma=0.1, epsilon=1e-6):

        self.name = name
        self.sigma = sigma
        self.epsilon = epsilon

    def _triplet_loss(self, x, label, label_msk):
        """ Triplet Loss for tuples (Q, P, N1, N2, ...,Nk)
        """
        return triplet_loss(x, label=label, label_msk=label_msk, margin=self.sigma)

    def _contrastive_loss(self, x, label, label_msk):
        """ Constractive Loss
        """
        return contrastive_loss(x, label=label, label_msk=label_msk, margin=self.sigma, eps=self.epsilon)

    def __call__(self, x, label, label_msk):

        loss = getattr(self, '_' + self.name + '_loss')(x, label, label_msk)

        return loss


class localFeatureAlgo:
    """Base class for Image Retrieval algorithms

    """
    def __init__(self,
                 loss,
                 min_level,
                 fpn_levels):

        self.loss = loss

        self.min_level = min_level
        self.fpn_levels = fpn_levels

    def _get_level(self, x):
        if isinstance(x, list):
            x = x[self.min_level:self.min_level + self.fpn_levels][0]
        elif isinstance(x, dict):
            x = x["mod5"]
        else:
            raise NameError("unknown input type")

        return x

    def _head(self, head, x):
        return head(x)

    @staticmethod
    def normalize(kpts, size):
        '''
            turn the coordinates from pixel indices to the range of [-1, 1]
        '''

        h, w = size
        center = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).to(kpts.device).float()
        kpts_normlized = (kpts - center) / center
        return kpts_normlized

    @staticmethod
    def denormalize(kpts_norm, size):
        '''
            turn the coordinates from normalized value ([-1, 1]) to actual pixel indices
        '''

        h, w = size

        center = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).to(kpts_norm.device)
        kpts = kpts_norm * center + center
        return kpts

    def training(self, head, x1, x2, kpts, img_size_1, img_size_2):

        # select fpn levels
        x1 = self._get_level(x1)
        x2 = self._get_level(x2)

        try:

            if kpts.all_none:
                raise Empty

            # normalize kpts

            kpts_norm = self.normalize(kpts, img_size_1)

            # Run head
            set_active_group(head, active_group(True))

            labels, labels_idx = labels.contiguous

            ret_pred = self._head(head, x)

            # Calculate losses
            ret_loss = self.loss(ret_pred, labels, labels_idx)

        except Empty:
            active_group(False)
            ret_loss = sum(x_i.sum() for x_i in x) * 0

        return ret_loss, ret_pred

    def inference(self, head, x, img_size):

        x = self._get_level(x)
        try:
            # Run head on the given proposals
            ret_pred = self._head(head, x)
        except Empty:
            ret_pred = PackedSequence([None for _ in range(x[0].size(0))])

        return ret_pred