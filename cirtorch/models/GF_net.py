import torch
import torch.nn as nn

from collections import OrderedDict

from cirtorch.utils.sequence import pad_packed_images
from cirtorch.utils.parallel import PackedSequence


class ImageRetrievalNet(nn.Module):
    
    def __init__(self, body, ret_algo, ret_head, augment=None):
        super(ImageRetrievalNet, self).__init__()
        self.augment = augment
        self.body = body

        self.ret_algo = ret_algo
        self.ret_head = ret_head

    def _prepare_pyramid_inputs(self, img, scales):

        scaled_imgs = []

        for scale in scales:

            # original scale
            if scale == 1:
                scaled_imgs.append(img)

            else:
                scaled_imgs.append(
                    PackedSequence([nn.functional.interpolate(img_id.unsqueeze(dim=0),
                                                              scale_factor=scale,
                                                              mode='bilinear',
                                                              align_corners=False).squeeze(dim=0)
                                    for img_id in img])
                )

        # return list of packed images in different scales
        return scaled_imgs

    def _prepare_inputs(self, img, positive_img, negative_img, labels=None):

        img_batch = []
        lbl_batch = []

        for img_id, pos_id, negs_ids, label_i in zip(img, positive_img, negative_img, labels):

            img_batch.append(img_id)
            img_batch.append(pos_id)

            for neg_id in negs_ids:
                img_batch.append(neg_id)

            lbl_batch.append(label_i)

        # Pack all images
        img = PackedSequence(img_batch)
        lbl = PackedSequence(lbl_batch)

        return img, lbl

    def forward(self, img=None, positive_img=None, negative_img=None, scales=[1], do_augmentaton=False, do_loss=False, do_prediction=True, **varargs):

        # Convert ground truth to the internal format
        if do_loss:

            if positive_img is None or negative_img is None:
                raise IOError(" Tuples is not correctly created ")

            img, labels = self._prepare_inputs(img, positive_img, negative_img, labels=varargs["tuple_labels"])

        # For evaluation only we create image pyramid of desired scales
        if len(scales) > 1:
            imgs = self._prepare_pyramid_inputs(img, scales)

            all_pred = []

            for img in imgs:
                _, pred = self.forward(img=img, scales=[1], do_prediction=True, do_loss=False)
                all_pred.append(pred["ret_pred"].unsqueeze(dim=0))

            # concatenate
            pred = torch.cat(all_pred, dim=0).permute(1, 2, 0)
            pred = nn.functional.avg_pool1d(pred, kernel_size=len(scales)).squeeze(-1)

            #

            pred = OrderedDict([
                ("ret_pred", pred)
            ])
            return _, pred

        # Perform augmentation if true and exists
        if self.augment:
            img = self.augment(img, masking=True, do_augmentation=do_augmentaton)

        # Pad the input images
        img, valid_size = pad_packed_images(img)
        img_size = img.shape[-2:]
        #print(img_size)

        # Run network body
        x = self.body(img)

        # Run image retrieval
        if do_loss:
            ret_loss, ret_pred = self.ret_algo.training(self.ret_head, x, labels, valid_size)

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
