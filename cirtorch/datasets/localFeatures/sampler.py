import math

import torch
from torch import distributed
from torch.utils.data.sampler import Sampler


class ARBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last=False, epoch=0):
        super(ARBatchSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self._epoch = epoch

        # images
        self.img_sets = self.data_source.img_1s

    def _generate_batches(self):
        g = torch.Generator()
        g.manual_seed(self._epoch)

        self.img_sets = list(torch.arange(len(self.img_sets)))

        batches = []
        batch = []

        for img in self.img_sets:
            batch.append(img)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []

        # add leftovers
        if not self.drop_last:
            if len(batch) != 0:
                batches.append(batch)

        return batches

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __len__(self):
        if self.drop_last:
            return len(self.img_sets) // self.batch_size
        else:
            return (len(self.img_sets) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        batches = self._generate_batches()
        for batch in batches:
            yield batch


class DistributedARBatchSampler(ARBatchSampler):
    def __init__(self, data_source, batch_size=1, num_replicas=None, rank=None, drop_last=False, shuffle=True, epoch=0):
        super(DistributedARBatchSampler, self).__init__(data_source, batch_size, drop_last, epoch)

        # Automatically get world size and rank if not provided
        if num_replicas is None:
            num_replicas = distributed.get_world_size()
        if rank is None:
            rank = distributed.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle

        tot_batches = super(DistributedARBatchSampler, self).__len__()
        self.num_batches = int(math.ceil(tot_batches / self.num_replicas))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        batches = self._generate_batches()
        g = torch.Generator()
        g.manual_seed(self._epoch)
        if self.shuffle:
            indices = list(torch.randperm(len(batches), generator=g))
        else:
            indices = list(torch.arange(len(batches)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.num_batches * self.num_replicas - len(indices))]

        assert len(indices) == self.num_batches * self.num_replicas

        # subsample
        offset = self.num_batches * self.rank
        indices = indices[offset:offset + self.num_batches]
        assert len(indices) == self.num_batches

        for idx in indices:
            batch = sorted(batches[idx])
            yield batch
