from os import path, listdir
from numpy.lib.function_base import select
import torch.utils.data as data

from glob import glob

from .utils import generate_kpts

from PIL import Image

import numpy as np

HP_INPUTS = ["img1", "img2", "kpts1", "kpts2", "H", "H_inv"]


class HPacthes(data.Dataset):

    def __init__(self, root_dir, name, split="all", bbx=None, transform=None, **varargs):
        super(HPacthes, self).__init__()
        
        self.root_dir = root_dir
        self.name = name

        self.transform = transform

        # load sequences
        if split!= "all":
            self.folders = [x for x in listdir(path.join(self.root_dir, self.name)) if x.startswith(split)]
        else:
            self.folders = [x for x in listdir(path.join(self.root_dir, self.name))]

        if len(self.folders) == 0:
            raise(RuntimeError("Empty Dataset !"))
        
        # Get images
        self.images, self.homographies = self._get_image_list()

        if len(self.images) == 0:
            raise(RuntimeError("Empty Images !"))
        
        self.num_kpt = varargs["num_kpt"]
        self.kpt_type = varargs["kpt_type"]
        self.prune_kpt = varargs["prune_kpt"]

    def _get_image_list(self):
        imgs = []
        Homographies = []
        
        for folder in self.folders:
            for i in range(2, 7):
                img1_path = path.join(self.root_dir, self.name, folder, "1.ppm")
                img2_path = path.join(self.root_dir, self.name, folder, str(i) + ".ppm")

                imgs.append((img1_path, img2_path))

                H = np.loadtxt(path.join(self.root_dir, self.name, folder, "H_1_" + str(i)))
                Homographies.append(H)

        return imgs, Homographies

    def _load_item(self, img_desc):

        if path.exists(img_desc + ".png"):
            img_file = img_desc + ".png"
        elif path.exists(img_desc + ".jpg"):
            img_file = img_desc + ".jpg"
        elif path.exists(img_desc):
            img_file = img_desc
        else:
            raise IOError("Cannot find any image for id {} ".format(img_desc))

        img = Image.open(img_file).convert(mode="RGB")

        return img

    def generate_keypoints(self, img):
    
        kpts = generate_kpts(img, self.kpt_type, self.num_kpt)

        return kpts
    
    @property
    def img_sizes(self):
        """Size of each image of the dataset"""
        return [img_desc["size"] for img_desc in self.images]

    def __len__(self):

        return len(self.images)

    def __getitem__(self, item):

        out = dict()

        out["img1_path"] = self.images[item][0]
        out["img2_path"] = self.images[item][1]
        
        out["img1"] = self._load_item(out["img1_path"])
        out["img2"] = self._load_item(out["img2_path"])
        
        out["H"] = self.homographies[item]
        
        out["kpts1"] = self.generate_keypoints(out["img1"])
        out["kpts2"] = self.generate_keypoints(out["img2"])

        out = self.transform(out)

        # Extra infos about tuple
        out["img1_size"] = (out["img1"].shape[1], out["img1"].shape[2])
        out["img2_size"] = (out["img2"].shape[1], out["img2"].shape[2])
      
        return  out


