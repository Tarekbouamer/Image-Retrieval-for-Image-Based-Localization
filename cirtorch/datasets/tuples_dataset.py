from os import path
from PIL import Image

import pickle
from tqdm import tqdm

import torch
import torch.distributed as distributed
import torch.utils.data as data

from cirtorch.datasets.generic.dataset import ISSDataset, INPUTS
from cirtorch.datasets.generic.transform import ISSTransform
from cirtorch.datasets.generic.sampler import DistributedARBatchSampler
from cirtorch.datasets.misc import iss_collate_fn, cid2filename
from cirtorch.utils.parallel import PackedSequence

NETWORK_INPUTS = ["img", "positive_img", "negative_img"]


class TuplesDataset(data.Dataset):
    """Data loader that loads training and validation tuples of 
        Radenovic etal ECCV16: CNN image retrieval learns from BoW

    Args:
        name (string):                  dataset name: 'retrieval-sfm-120k'
        mode (string):                  'train' or 'val' for training and validation parts of dataset
        imsize (int, Default: None):    Defines the maximum size of longer image side
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        loader (callable, optional):    A function to load an image given its path.
        nnum (int, Default:5):          Number of negatives for a query image in a training tuple
        qsize (int, Default:1000):      Number of query images, ie number of (q,p,n1,...nN) tuples, to be processed in one epoch
        poolsize (int, Default:10000):  Pool size for negative images re-mining

     Attributes:
        images (list):      List of full filenames for each image
        clusters (list):    List of clusterID per image
        qpool (list):       List of all query image indexes
        ppool (list):       List of positive image indexes, each corresponding to query at the same position in qpool

        qidxs (list): List of qsize query image indexes to be processed in an epoch
        pidxs (list): List of qsize positive image indexes, each corresponding to query at the same position in qidxs
        nidxs (list): List of qsize tuples of negative images
                        Each nidxs tuple contains nnum images corresponding to query image at the same position in qidxs

        Lists qidxs, pidxs, nidxs are refreshed by calling the ``create_epoch_tuples()`` method, 
            ie new q-p pairs are picked and negative images are remined
    """

    def __init__(self, root_dir, name, mode, batch_size=1, num_workers=1, neg_num=5, query_size=2000, pool_size=20000, transform=None):

        if not (mode == 'train' or mode == 'val'):
            raise(RuntimeError("Mode should be either train or val, passed as string"))

        if name.startswith('retrieval-SfM'):
            # setting up paths
            db_root = path.join(root_dir, 'train', name)
            ims_root = path.join(db_root, 'ims')
    
            # loading db
            db_fn = path.join(db_root, '{}.pkl'.format(name))
            with open(db_fn, 'rb') as f:
                db = pickle.load(f)[mode]
    
            # get images full path
            self.images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]

        elif name.startswith('gl'):
            # setting up paths
            db_root = '/mnt/fry2/users/datasets/landmarkscvprw18/recognition/'
            ims_root = path.join(db_root, 'images', 'train')
    
            # loading db
            db_fn = path.join(db_root, '{}.pkl'.format(name))
            with open(db_fn, 'rb') as f:
                db = pickle.load(f)[mode]
    
            # setting fullpath for images
            self.images = [path.join(ims_root, db['cids'][i]+'.jpg') for i in range(len(db['cids']))]
        else:
            raise(RuntimeError("Unknown dataset name!"))

        # initializing tuples dataset
        self.name = name
        self.mode = mode

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.clusters = db['cluster']
        self.query_pool = db['qidxs']
        self.positive_pool = db['pidxs']

        # size of training subset for an epoch
        self.neg_num = neg_num
        self.query_size = min(query_size, len(self.query_pool))
        self.pool_size = min(pool_size, len(self.images))

        self.query_indices = None
        self.positive_indices = None
        self.negative_indices = None

        self.transform = transform

    def load_img_from_desc(self, img_desc):

        if path.exists(img_desc + ".png"):
            img_file = img_desc + ".png"
        elif path.exists(img_desc + ".jpg"):
            img_file = img_desc + ".jpg"
        elif path.exists(img_desc):
            img_file = img_desc
        else:
            raise IOError("Cannot find any image for id {} ".format(img_desc))

        return Image.open(img_file).convert(mode="RGB")

    def _load_query(self, item, out):

        query_img_desc = self.images[self.query_indices[item]]

        query_img = self.load_img_from_desc(query_img_desc)

        rec = self.transform(query_img)

        query_size = (query_img.size[1], query_img.size[0])

        query_img.close()

        out["img"] = rec["img"]
        out["img_desc"] = query_img_desc
        out["size"] = query_size

        return out

    def _load_positive(self, item, out):

        positive_img_desc = self.images[self.positive_indices[item]]

        positive_img = self.load_img_from_desc(positive_img_desc)

        rec = self.transform(positive_img)

        query_size = (positive_img.size[1], positive_img.size[0])

        positive_img.close()

        out["positive_img"] = rec["img"]
        out["positive_img_desc"] = positive_img_desc
        out["positive_size"] = query_size

        return out

    def _load_negative(self, item, out):
        n_images = []
        n_images_desc = []
        n_query_size = []

        for n_idx in self.negative_indices[item]:

            negative_img_desc = self.images[n_idx]

            negative_img = self.load_img_from_desc(negative_img_desc)

            rec = self.transform(negative_img)
            query_size = (negative_img.size[1], negative_img.size[0])
            negative_img.close()

            # Append
            n_images.append(rec["img"])
            n_images_desc.append(negative_img_desc)
            n_query_size.append(query_size)

        # Re make output
        out["negative_img"] = PackedSequence(n_images)
        out["negative_img_desc"] = n_images_desc
        out["negative_size"] = n_query_size

        return out

    def __len__(self):
        return self.query_size

    def __getitem__(self, item):

        out = dict()
        out["idx"] = item

        # query image
        out = self._load_query(item, out)

        # positive image
        out = self._load_positive(item, out)

        # negative images
        out = self._load_negative(item, out)

        out["tuple_labels"] = torch.Tensor([-1, 1] + [0]*len(out["negative_img"]))
        return out

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Name and mode: {} {}\n'.format(self.name, self.mode)
        fmt_str += '    Number of images: {}\n'.format(len(self.images))
        fmt_str += '    Number of training tuples: {}\n'.format(len(self.query_pool))
        fmt_str += '    Number of negatives per tuple: {}\n'.format(self.neg_num)
        fmt_str += '    Number of tuples processed in an epoch: {}\n'.format(self.query_size)
        fmt_str += '    Pool size for negative remining: {}\n'.format(self.pool_size)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def create_epoch_tuples(self, model, log_info, log_debug, **varargs):

        log_debug('Creating tuples for an epoch of {%s}--{%s}', self.name, self.mode)

        data_config = varargs["data_config"]

        # Set model to eval mode
        model.eval()

        # Select positive pairs
        idxs2qpool = torch.randperm(len(self.query_pool))[:self.query_size]

        self.query_indices = [self.query_pool[i] for i in idxs2qpool]
        self.positive_indices = [self.positive_pool[i] for i in idxs2qpool]

        # Select negative pairs
        idxs2images = torch.randperm(len(self.images))[:self.pool_size]

        batch_size = data_config.getint("test_batch_size")
        with torch.no_grad():
            # Prepare query loader

            log_debug('Extracting descriptors for query images :')

            query_tf = ISSTransform(shortest_size=data_config.getint("train_shortest_size"),
                                    longest_max_size=data_config.getint("train_longest_max_size"),
                                    rgb_mean=data_config.getstruct("rgb_mean"),
                                    rgb_std=data_config.getstruct("rgb_std"),
                                    random_flip=data_config.getboolean("random_flip"))

            query_data = ISSDataset(root_dir='',
                                    name="query",
                                    images=[self.images[i] for i in self.query_indices],
                                    transform=query_tf)

            query_sampler = DistributedARBatchSampler(data_source=query_data,
                                                      batch_size=batch_size,
                                                      num_replicas=varargs["world_size"],
                                                      rank=varargs["rank"],
                                                      drop_last=True,
                                                      shuffle=False)

            query_dl = torch.utils.data.DataLoader(query_data,
                                                   batch_sampler=query_sampler,
                                                   collate_fn=iss_collate_fn,
                                                   pin_memory=True,
                                                   num_workers=self.num_workers,
                                                   shuffle=False)

            # Extract query vectors
            qvecs = torch.zeros(varargs["output_dim"], len(self.query_indices)).cuda()

            for it, batch in tqdm(enumerate(query_dl), total=len(query_dl)):

                # Upload batch
                batch = {k: batch[k].cuda(device=varargs["device"], non_blocking=True) for k in INPUTS}

                _, pred = model(**batch, do_prediction=True)
                distributed.barrier()

                qvecs[:, it * batch_size: (it+1) * batch_size] = pred["ret_pred"]

                del pred

            # Prepare negative pool data loader

            log_debug('Extracting descriptors for negative pool :')

            pool_tf = ISSTransform(shortest_size=data_config.getint("train_shortest_size"),
                                   longest_max_size=data_config.getint("train_longest_max_size"),
                                   rgb_mean=data_config.getstruct("rgb_mean"),
                                   rgb_std=data_config.getstruct("rgb_std"),
                                   random_flip=data_config.getboolean("random_flip"))

            pool_data = ISSDataset(root_dir='',
                                   name="negative_pool",
                                   images=[self.images[i] for i in idxs2images],
                                   transform=pool_tf)

            pool_sampler = DistributedARBatchSampler(data_source=pool_data,
                                                     batch_size=batch_size,
                                                     num_replicas=varargs["world_size"],
                                                     rank=varargs["rank"],
                                                     drop_last=True,
                                                     shuffle=False)

            pool_dl = torch.utils.data.DataLoader(pool_data,
                                                  batch_sampler=pool_sampler,
                                                  collate_fn=iss_collate_fn,
                                                  pin_memory=True,
                                                  num_workers=self.num_workers,
                                                  shuffle=False)

            # Extract negative pool vectors
            poolvecs = torch.zeros(varargs["output_dim"], len(idxs2images)).cuda()

            for it, batch in tqdm(enumerate(pool_dl), total=len(pool_dl)):

                # Upload batch
                batch = {k: batch[k].cuda(device=varargs["device"], non_blocking=True) for k in INPUTS}

                _, pred = model(**batch, do_prediction=True)
                distributed.barrier()

                poolvecs[:, it * batch_size: (it+1) * batch_size] = pred["ret_pred"]
                del pred

            log_debug('Searching for hard negatives :')

            # Compute dot product scores and ranks on GPU
            scores = torch.mm(poolvecs.t(), qvecs)

            scores, scores_indices = torch.sort(scores, dim=0, descending=True)

            average_negative_distance = torch.tensor(0).float().cuda()  # for statistics

            negative_distance = torch.tensor(0).float().cuda()  # for statistics

            # Selection of negative examples
            self.negative_indices = []

            for q in range(len(self.query_indices)):

                # Do not use query cluster those images are potentially positive
                qcluster = self.clusters[self.query_indices[q]]
                clusters = [qcluster]
                nidxs = []
                r = 0
                while len(nidxs) < self.neg_num:
                    potential = idxs2images[scores_indices[r, q]]
                    # take at most one image from the same cluster

                    if not self.clusters[potential] in clusters:
                        nidxs.append(potential)
                        clusters.append(self.clusters[potential])
                        average_negative_distance += torch.pow(qvecs[:,q]-poolvecs[:,scores_indices[r, q]]+1e-6, 2).sum(dim=0).sqrt()
                        negative_distance += 1
                    r += 1
                self.negative_indices.append(nidxs)

            del scores
            log_info('Average negative l2-distance = %f', average_negative_distance/negative_distance)

        return (average_negative_distance/negative_distance).item()  # return average negative l2-distance
