import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist
import random
import numpy as np
import torch


class DistributedSamplerChunkByNode(torch.utils.data.Sampler):

    def __init__(self,
                 dataset,
                 all_datasets,
                 chunk_or_not,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0,
                 drop_last: bool = False,
                 node_rank=0,
                 node_number=1, process_num_per_node=1,
                 rank_within_local_node=0) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.node_number = node_number
        self.node_rank = node_rank
        self.chunk_or_not = chunk_or_not
        self.process_num_per_node = process_num_per_node
        self.rank_within_local_node = rank_within_local_node

        assert (self.process_num_per_node * self.node_number == self.num_replicas)

        # 1. divide the datasets into two parts
        normal_datasets = []
        chunked_datasets = []
        for dataset_i, chunk_i in zip(all_datasets, chunk_or_not):
            if chunk_i:
                chunked_datasets.append(dataset_i)
            else:
                normal_datasets.append(dataset_i)

        # 2. calculate dataset sizes:
        self.normal_dataset_size = sum(
            [len(i) for i in normal_datasets])  # this part we follow the conventional distributed sampler

        # 3. Divide 
        self.current_node_start_range = -1
        self.current_node_end_range = -1
        assert (len(chunked_datasets) >= self.node_number)
        chunk_size = len(chunked_datasets) // self.node_number
        current_example_num = self.normal_dataset_size

        for index in range(len(chunked_datasets)):
            if index == self.node_rank * chunk_size:
                self.current_node_start_range = current_example_num
            current_example_num += len(chunked_datasets[index])
            if index == (self.node_rank + 1) * chunk_size - 1:
                self.current_node_end_range = current_example_num

        if self.current_node_end_range == -1:  # boundary
            self.current_node_end_range = current_example_num

        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        indices = self.generate_indices_within_range_with_rank(
            seed=self.seed,
            epoch=self.epoch,

            # NOTE: Distribute among all processes
            process_num=self.num_replicas,
            rank=self.rank,
            generate_length=-1,
            valid_indices=list(range(self.normal_dataset_size)),
            prefix="Normal "
        )

        addition_indices = self.generate_indices_within_range_with_rank(
            seed=self.seed,
            epoch=self.epoch,

            # NOTE : very important arguments, distribute among local nodes
            process_num=self.process_num_per_node,
            rank=self.rank_within_local_node,

            generate_length=self.num_samples - len(indices),
            valid_indices=list(range(self.current_node_start_range, self.current_node_end_range)),
            prefix="Distribute "
        )

        indices.extend(addition_indices)
        random.seed(self.seed + self.epoch + 10 * self.rank)  # Set the seed to maximize randomness
        random.shuffle(indices)  # Reshuffle
        assert len(indices) == self.num_samples
        return iter(indices)

    def generate_indices_within_range_with_rank(self, seed, epoch, process_num, generate_length, valid_indices, rank=-1,
                                                shuffle=True, prefix=""):
        '''
        Use scenario : we want to sample 2500 examples from 10000 examples, while not sampling overlapping examples with other three process.
        Modified from DistributedSampler
        '''
        dataset_size = len(valid_indices)
        if shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(seed + epoch)
            indices = torch.randperm(dataset_size, generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(dataset_size))  # type: ignore[arg-type]

        indices = [valid_indices[i] for i in indices]

        num_samples_normal = math.ceil(
            (dataset_size - process_num) / process_num  # type: ignore[arg-type]
        )
        # remove tail of data to make it evenly divisible.
        indices = indices[:num_samples_normal * process_num]

        print("\n")
        print(prefix,
              "Global Rank {}   Local Rank {}    generate_length {}    valid_indices {}    process_num {}  indices_before_subsample {} {}".format(
                  self.rank, rank, generate_length, len(valid_indices), process_num, len(indices), indices[:10]))

        # subsample
        indices = indices[rank:num_samples_normal * process_num: process_num]

        print(prefix,
              "Global Rank {}   Local Rank {}    generate_length {}    valid_indices {}    process_num {}  indices_after_subsample {} {}".format(
                  self.rank, rank, generate_length, len(valid_indices), process_num, len(indices), indices[:10]))
        print("\n")

        if generate_length != -1:
            if len(indices) > generate_length:
                indices = indices[:generate_length]
            else:
                indices.extend(np.random.choice(valid_indices, generate_length - len(indices)).tolist())
        return indices

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
