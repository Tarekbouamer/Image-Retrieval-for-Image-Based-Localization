from os import path, listdir
from collections import namedtuple, OrderedDict

import torch
import torch.utils.data as data
import numpy as np
import cv2
import skimage.io as io

from PIL import Image

from .utils import (
    skew,
    random_choice,
    rotateImage,
    generate_kpts,
    prune_kpts,
    perspective_transform
)
from tqdm import tqdm

rand = np.random.RandomState(234)

class MegaDepthDataset(data.Dataset):
    def __init__(self, root_dir, name, split, bbx=None, transform=None, **varargs):
        self.root_dir = root_dir
        self.name = name

        self.split = split
        self.transform = transform

        if bbx is not None:
            self.bbxs = bbx

        # dataset root
        self.root = path.join(self.root_dir, self.name, self.split)

        # read images descriptions from all scenes
        self.imgs_desc = self.load_img_cams()

        # read images pairs from all scenes with max pairs per scenes
        self.images = self.load_img_pairs(max_per_scene=varargs["max_per_scene"])

        self.img_1s, self.img_2s = self.images

        assert len(self.img_1s) == len(self.img_2s)

        self.num_kpt = varargs["num_kpt"]
        self.kpt_type = varargs["kpt_type"]
        self.prune_kpt = varargs["prune_kpt"]

    def __len__(self):
        if len(self.img_1s) > 0:
            return len(self.img_1s)
        else:
            raise (RuntimeError("dataset not loaded properly!"))

    def load_img_cams(self):
        img_db = OrderedDict()

        img_desc = namedtuple("Image", ["img_path", "w", "h", "fx", "fy", "cx", "cy", "rot", "trans"])

        for scene_i in sorted(listdir(self.root)):

            denses = [f for f in listdir(path.join(self.root, scene_i)) if f.startswith('dense')]

            for dense_i in denses:

                current_path = path.join(self.root, scene_i, dense_i, 'aligned')

                # camera parameters
                cam_path = path.join(current_path, 'img_cam.txt')

                with open(cam_path, "r") as fid:

                    while True:
                        line = fid.readline()

                        if not line:
                            break

                        line = line.strip()

                        if len(line) > 0 and line[0] != "#":
                            elems = line.split()

                            img_name = elems[0]

                            img_path = path.join(current_path, 'images', img_name)

                            w, h = int(elems[1]), int(elems[2])

                            fx, fy = float(elems[3]), float(elems[4])

                            cx, cy = float(elems[5]), float(elems[6])

                            rot = np.array(elems[7:16])

                            trans =np.array(elems[16:19])

                            img_db[img_name] = img_desc(img_path=img_path,
                                                        w=w, h=h,
                                                        fx=fx, fy=fy,
                                                        cx=cx, cy=cy,
                                                        rot=rot,
                                                        trans=trans)

        return img_db

    def load_img_pairs(self, max_per_scene):
        img_1s, img_2s = [], []

        for scene_i in tqdm(listdir(self.root)):

            denses = [f for f in listdir(path.join(self.root, scene_i)) if f.startswith('dense')]

            for dense_i in denses:
                img_1s_i = []
                img_2s_i = []

                folder = path.join(self.root, scene_i, dense_i, 'aligned')
                pair_path = path.join(folder, 'pairs.txt')

                if path.exists(pair_path):
                    f = open(pair_path, 'r')

                    for line in f:
                        img_pair = line.strip().split(' ')

                        img_1s_i.append(img_pair[0])
                        img_2s_i.append(img_pair[1])

                # max per scene_i/dense_i
                if len(img_1s_i) > max_per_scene:
                    index = np.arange(len(img_1s_i))
                    rand.shuffle(index)

                    img_1s_i = list(np.array(img_1s_i)[index[:max_per_scene]])
                    img_2s_i = list(np.array(img_2s_i)[index[:max_per_scene]])

                img_1s.extend(img_1s_i)
                img_2s.extend(img_2s_i)

        return img_1s, img_2s

    @staticmethod
    def get_intrinsics(desc):
        return np.array([[desc.fx, 0, desc.cx],
                         [0, desc.fy, desc.cy],
                         [0, 0, 1]])

    @staticmethod
    def get_extrinsics(desc):
        rot = desc.rot.reshape(3, 3)
        trans = desc.trans
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rot
        extrinsic[:3, 3] = trans
        return extrinsic

    @staticmethod
    def _load_img(img_path, numpy=False):

        if path.exists(img_path + ".png"):
            img_file = img_path + ".png"
        elif path.exists(img_path + ".jpg"):
            img_file = img_path + ".jpg"
        if path.exists(img_path):
            img_file = img_path
        else:
            raise IOError("Cannot find any image for id {} ".format(img_path))

        img = Image.open(img_file).convert(mode="RGB")

        if numpy:
            img = np.asarray(img)

        return img

    def get_F(self, desc_1, desc_2):
        #TODO: move this inside generator in algorithms since 
        # we perform a sequence of geometric operations 
        # afterward leading to recomputation of fundamental matrix

        k1 = self.get_intrinsics(desc_1)
        k2 = self.get_intrinsics(desc_1)

        # relative transformation
        P = T2.dot(np.linalg.inv(T1))

        rot = P[:3, :3]
        trans = P[:3, 3]

        # remove pairs that have a relative rotation angle larger than 80 degrees
        theta = np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)) * 180 / np.pi
        # if theta > 80 and self.phase == 'train':
        #   return None

        Tx = skew(trans)
        E = np.dot(Tx, rot)
        F = np.linalg.inv(k2).T.dot(E).dot(np.linalg.inv(k1))

        return F, E, P

      
        return out

    def generate_keypoints(self, img):

        kpts = generate_kpts(img, self.kpt_type, self.num_kpt)

        return kpts

    def prune_keypoints(self, keypoints, F, size_2, k1, k2, P, dmin=4, dmax=400):

        idx = prune_kpts(keypoints, F, size_2, k1, k2, P, d_min=dmin, d_max=dmax)

        if np.sum(idx) == 0:
            return None
        else:
            keypoints = keypoints[idx]

        return keypoints

    def __getitem__(self, item):
        
        out = dict()

        desc_1 = self.imgs_desc[self.img_1s[item]]
        desc_2 = self.imgs_desc[self.img_2s[item]]

        # Load images
        out["img1"] = self._load_img(desc_1.img_path)
        out["img2"] = self._load_img(desc_2.img_path)
       

        # Get intrinsics 
        out["intrinsics1"] = self.get_intrinsics(desc_1)
        out["intrinsics2"] = self.get_intrinsics(desc_2)

        # Get extrinsics 
        out["extrinsics1"] = self.get_extrinsics(desc_1)
        out["extrinsics2"] = self.get_extrinsics(desc_2)

        # Generate Keypoints on img_1
        keypoints = self.generate_keypoints(img=out["img1"])

        keypoints = random_choice(keypoints, self.num_kpt)
        
        out["kpts"] = keypoints

        # Scale images/intrinsics to desired scale range
        # form *.any to torch.Tensor 
    
        out = self.transform(out)

        # Extra infos about tuple
        out["img1_size"] = (out["img1"].shape[1], out["img1"].shape[2])
        out["img2_size"] = (out["img2"].shape[1], out["img2"].shape[2])

        out["img1_path"] = desc_1.img_path
        out["img2_path"] = desc_2.img_path

        return out

