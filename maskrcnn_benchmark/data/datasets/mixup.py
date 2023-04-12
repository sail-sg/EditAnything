"""Mixup detection dataset wrapper."""
from __future__ import absolute_import
import numpy as np
import torch
import torch.utils.data as data


class MixupDetection(data.Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset.
    Parameters
    ----------
    dataset : mx.gluon.data.Dataset
        Gluon dataset object.
    mixup : callable random generator, e.g. np.random.uniform
        A random mixup ratio sampler, preferably a random generator from numpy.random
        A random float will be sampled each time with mixup(*args).
        Use None to disable.
    *args : list
        Additional arguments for mixup random sampler.
    """
    def __init__(self, dataset, mixup=None, preproc=None, *args):
        super().__init__(dataset.input_dim)
        self._dataset = dataset
        self.preproc = preproc
        self._mixup = mixup
        self._mixup_args = args

    def set_mixup(self, mixup=None, *args):
        """Set mixup random sampler, use None to disable.
        Parameters
        ----------
        mixup : callable random generator, e.g. np.random.uniform
            A random mixup ratio sampler, preferably a random generator from numpy.random
            A random float will be sampled each time with mixup(*args)
        *args : list
            Additional arguments for mixup random sampler.
        """
        self._mixup = mixup
        self._mixup_args = args

    def __len__(self):
        return len(self._dataset)

    @Dataset.resize_getitem
    def __getitem__(self, idx):
        self._dataset._input_dim = self.input_dim
        # first image
        img1, label1, _, _= self._dataset.pull_item(idx)
        lambd = 1

        # draw a random lambda ratio from distribution
        if self._mixup is not None:
            lambd = max(0, min(1, self._mixup(*self._mixup_args)))

        if lambd >= 1:
            weights1 = np.ones((label1.shape[0], 1))
            label1 = np.hstack((label1, weights1))
            height, width, _ = img1.shape
            img_info = (width, height)
            if self.preproc is not None:
                img_o, target_o = self.preproc(img1, label1, self.input_dim)
            return img_o, target_o, img_info, idx

        # second image
        idx2 = int(np.random.choice(np.delete(np.arange(len(self)), idx)))
        img2, label2, _, _ = self._dataset.pull_item(idx2)

        # mixup two images
        height = max(img1.shape[0], img2.shape[0])
        width = max(img1.shape[1], img2.shape[1])
        mix_img = np.zeros((height, width, 3),dtype=np.float32)
        mix_img[:img1.shape[0], :img1.shape[1], :] = img1.astype(np.float32) * lambd
        mix_img[:img2.shape[0], :img2.shape[1], :] += img2.astype(np.float32) * (1. - lambd)
        mix_img = mix_img.astype(np.uint8)

        y1 = np.hstack((label1, np.full((label1.shape[0], 1), lambd)))
        y2 = np.hstack((label2, np.full((label2.shape[0], 1), 1. - lambd)))
        mix_label = np.vstack((y1, y2))
        if self.preproc is not None:
            mix_img, padded_labels = self.preproc(mix_img, mix_label, self.input_dim)

        img_info = (width, height)

        return mix_img, padded_labels, img_info , idx

    def pull_item(self, idx):
        self._dataset._input_dim = self.input_dim
        # first image
        img1, label1, _, _= self._dataset.pull_item(idx)
        lambd = 1

        # draw a random lambda ratio from distribution
        if self._mixup is not None:
            lambd = max(0, min(1, self._mixup(*self._mixup_args)))

        if lambd >= 1:
            weights1 = np.ones((label1.shape[0], 1))
            label1 = np.hstack((label1, weights1))
            height, width, _ = img1.shape
            img_info = (width, height)
            if self.preproc is not None:
                img_o, target_o = self.preproc(img1, label1, self.input_dim)
            return img_o, target_o, img_info, idx

        # second image
        idx2 = int(np.random.choice(np.delete(np.arange(len(self)), idx)))
        img2, label2 = self._dataset.pull_item(idx2)

        # mixup two images
        height = max(img1.shape[0], img2.shape[0])
        width = max(img1.shape[1], img2.shape[1])
        mix_img = np.zeros((height, width, 3),dtype=np.float32)
        mix_img[:img1.shape[0], :img1.shape[1], :] = img1.astype(np.float32) * lambd
        mix_img[:img2.shape[0], :img2.shape[1], :] += img2.astype(np.float32) * (1. - lambd)
        mix_img = mix_img.astype(np.uint8)

        y1 = np.hstack((label1, np.full((label1.shape[0], 1), lambd)))
        y2 = np.hstack((label2, np.full((label2.shape[0], 1), 1. - lambd)))
        mix_label = np.vstack((y1, y2))
        if self.preproc is not None:
            mix_img, padded_labels = self.preproc(mix_img, mix_label, self.input_dim)

        img_info = (width, height)
        return mix_img, padded_labels, img_info , idx
