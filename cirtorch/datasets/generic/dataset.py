from os import path
import pickle
import torch.utils.data as data
from PIL import Image


INPUTS = ["img"]


class ISSDataset(data.Dataset):

    def __init__(self, root_dir, name, images, bbx=None, transform=None):
        super(ISSDataset, self).__init__()
        self.root_dir = root_dir
        self.name = name
        self.transform = transform

        # load sequences
        self.images = [path.join(self.root_dir, images[i]) for i in range(len(images))]

        if bbx is not None:
            self.bbxs = bbx
        else:
            self.bbxs = None

        if len(self.images) == 0:
            raise(RuntimeError("Empty Dataset !"))

    def _load_item(self, item):
        img_desc = self.images[item]

        if self.bbxs is not None:
            bbx = self.bbxs[item]
        else:
            bbx = None

        if path.exists(img_desc + ".png"):
            img_file = img_desc + ".png"
        elif path.exists(img_desc + ".jpg"):
            img_file = img_desc + ".jpg"
        elif path.exists(img_desc):
            img_file = img_desc
        else:
            raise IOError("Cannot find any image for id {} ".format(img_desc))

        img = Image.open(img_file).convert(mode="RGB")

        return img, img_desc, bbx

    @property
    def img_sizes(self):
        """Size of each image of the dataset"""
        return [img_desc["size"] for img_desc in self.images]

    def __len__(self):

        return len(self.images)

    def __getitem__(self, item):

        img, img_desc, bbx = self._load_item(item)

        rec = self.transform(img, bbx)

        size = (img.size[1], img.size[0])

        img.close()

        rec["idx"] = item
        rec["img_desc"] = img_desc
        rec["size"] = size
        return rec


TEST_DATASETS = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']
_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]


def ParisOxfordTestDataset(root_dir, name=None):

    if name not in TEST_DATASETS:
        raise ValueError('Unknown dataset: {}!'.format(name))

    # Loading imlist, qimlist, and gnd, in cfg as a dict
    pkl_path = path.join(root_dir,
                         'gnd_{}.pkl'.format(name))

    with open(pkl_path, 'rb') as f:
        db = pickle.load(f)

    db['pkl_path'] = pkl_path
    db['data_path'] = path.join(root_dir)
    db['images_path'] = path.join(db['data_path'], 'jpg')
    db['_ext'] = '.jpg'

    db['n_img'] = len(db['imlist'])
    db['n_query'] = len(db['qimlist'])

    db['img_names'] = [path.join(db['images_path'], item + db['_ext']) for item in db['imlist']]
    db['query_names'] = [path.join(db['images_path'], item + db['_ext']) for item in db['qimlist']]
    db['query_bbx'] = [db['gnd'][item]['bbx'] for item in range(db['n_query'])]

    db['dataset'] = name
    return db

