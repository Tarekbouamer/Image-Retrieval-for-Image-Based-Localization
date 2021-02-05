import torch
import torch.nn as nn

urls = dict()
urls["lib"] = "https://github.com/yuruntian/SOSNet/raw/master/sosnet-weights/sosnet_32x32_liberty.pth"
urls["hp_a"] = "https://github.com/yuruntian/SOSNet/raw/master/sosnet-weights/sosnet_32x32_hpatches_a.pth"


class SOSNet(nn.Module):
    """
        128-dimensional SOSNet model definition for 32x32 patches.
        This is based on the original code from paper
        "SOSNet:Second Order Similarity Regularization for Local Descriptor Learning".
    """

    def __init__(self, pretrained):
        super(SOSNet, self).__init__()

        self.layers = nn.Sequential(
            nn.InstanceNorm2d(1, affine=False),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

        self.desc_norm = nn.Sequential(
            nn.LocalResponseNorm(256, alpha=256, beta=0.5, k=0)
        )

        # load pretrained model
        if pretrained:
            pretrained_dict = torch.hub.load_state_dict_from_url(urls['lib'], map_location=lambda storage, loc: storage)
            self.load_state_dict(pretrained_dict, strict=True)

        return

    def forward(self, input, eps=1e-10):
        desc = self.desc_norm(self.layers(input) + eps)
        desc = desc.view(desc.size(0), -1)
        return desc