import torch
import torch.nn as nn

from collections import OrderedDict

from torchvision import transforms
from PIL import ImageDraw


from cirtorch.utils.sequence import pad_packed_images
from cirtorch.utils.parallel import PackedSequence

NETWORK_TRAIN_INPUTS = ['img1', 'img2', 'intrinsics1', 'intrinsics2', 'extrinsics1', 'extrinsics2', 'kpts']
NETWORK_TEST_INPUTS = ['img', 'kpts']


class localNet(nn.Module):

    def __init__(self, body, local_algo, local_head, augment=None):
        super(localNet, self).__init__()
        self.augment = augment
        self.body = body

        self.local_algo = local_algo
        self.local_head = local_head
    
    def show(self, inp, out):
        print(type(inp["kpts"]))
        print(type(out["kpts"]))

        for (inp_img, inp_kpt, out_img, out_kpt) in zip(inp["img"], inp["kpts"], out["img"], out["kpts"]):
            
            inp_img = transforms.ToPILImage()(inp_img.cpu()).convert("RGB")
            out_img = transforms.ToPILImage()(out_img.cpu()).convert("RGB")

            inp_draw = ImageDraw.Draw(inp_img)
            out_draw = ImageDraw.Draw(out_img)
            
            print(inp_kpt.shape)
            print(out_kpt.shape)

            inp_kpt = inp_kpt.view(-1).int().tolist()
            out_kpt = out_kpt.view(-1).int().tolist()
            
            # Draw points
            inp_draw.point(inp_kpt, fill=255)
            out_draw.point(out_kpt, fill=255)
            
            # Plot
            inp_img.show()
            out_img.show()

            input()

    def forward(self, img=None, img1=None, img2=None, intrinsics1=None, intrinsics2=None, extrinsics1=None, extrinsics2=None, kpts=None,
                do_augmentaton=False, do_loss=False, do_prediction=True, **varargs):

        # Perform augmentation if true and exists
        if self.augment:
            with torch.no_grad():
                
                if img:
                    inp = {"img": img, "kpts": kpts}
                    out = self.augment(inp, do_augmentation=False)

                if img1:
                    inp_1 = {"img": img1, "kpts": kpts}
                    out_1 = self.augment(inp_1, do_augmentation=False)

                if img2:
                    inp_2 = {"img": img2}
                    out_2 = self.augment(inp_2, do_augmentation=False)

        # Run network body

        if img:
            img, _ = pad_packed_images(inp["img"])
            img_size = img.shape[-2:]
            
            x = self.body(img)

        if img1:
            img1, _ = pad_packed_images(inp_1["img"])
            img_size_1 = img1.shape[-2:]
            
            x1 = self.body(img1)

        if img2:
            img2, _ = pad_packed_images(inp_2["img"])
            img_size_2 = img2.shape[-2:]
        
            x2 = self.body(img2)

        # Run local Net
        if do_loss:
            local_loss = self.local_algo.training(head=self.local_head, 
                                                  x1=x1, x2=x2, kpts=kpts,
                                                  img_size_1=img_size_1, img_size_2=img_size_2,
                                                  intrinsics1=intrinsics1, intrinsics2=intrinsics2,
                                                  extrinsics1=extrinsics1, extrinsics2=extrinsics2,
                                                  aug_trans1=inp_1["transform"], aug_trans2=inp_2["transform"])            
            local_pred = None 
        
        elif do_prediction:
            local_pred = self.local_algo.inference(self.local_head, x=x, kpts=kpts, img_size=img_size)
            local_loss = None

        else:
            local_pred, local_loss = None, None

        # Prepare outputs
        loss = OrderedDict([
            ("local_loss", local_loss)
        ])

        pred = OrderedDict([
            ("local_pred", local_pred)
        ])

        return loss, pred