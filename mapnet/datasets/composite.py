"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
Composite data-loaders derived from class specific data loaders
"""

import numpy as np
import torch
from torch.utils import data

from ..registry import DATASETS
from ..builder import build_dataset


@DATASETS.register_module(name='MF')
class MF(data.Dataset):
    """
    Returns multiple consecutive frames, and optionally VOs
    """
    def __init__(self, name, no_duplicates=False, **kwargs):
        """
        Args:
            steps: Number of frames to return on every call
            skip: Number of frames to skip
            variable_skip: If True, skip = [1, ..., skip]
            no_duplicates: if True, does not duplicate frames when len(self) is
                not a multiple of skip*steps
        """
        self.steps = kwargs.pop('steps', 2)
        self.skip = kwargs.pop('skip', 1)
        self.variable_skip = kwargs.pop('variable_skip', False)
        self.train = kwargs['train']
        self.no_duplicates = no_duplicates

        kwargs['type'] = name
        self.dset = build_dataset(kwargs)
        self.dim3 = self.dset.dim3 if hasattr(self.dset, 'dim3') else False

        self.L = self.steps * self.skip

    def get_indices(self, index):
        if self.variable_skip:
            skips = np.random.randint(1, high=self.skip+1, size=self.steps-1)
        else:
            skips = self.skip * np.ones(self.steps-1)
        # print("skips: {}".format(skips))
        offsets = np.insert(skips, 0, 0).cumsum()
        # print("offsets: {}".format(offsets))
        offsets -= offsets[len(offsets) // 2]
        # print("offsets: {}".format(offsets))
        if self.no_duplicates:
            offsets += self.steps//2 * self.skip
        offsets = offsets.astype(np.int)
        idx = index + offsets
        idx = np.minimum(np.maximum(idx, 0), len(self.dset)-1)
        assert np.all(idx >= 0), '{:d}'.format(index)
        assert np.all(idx < len(self.dset))
        return idx

    def __getitem__(self, index):
        """
        Args:
            index:
        Return:
            imgs: STEPS x 3 x H x W
            poses: STEPS x 7
        """
        idx = self.get_indices(index)
        # print("idx = ", idx)
        clip = [self.dset[i] for i in idx]
        imgs = torch.stack([c[0] for c in clip], dim=0)
        poses = torch.stack([c[1] for c in clip], dim=0)
        return imgs, poses

    def __len__(self):
        L = len(self.dset)
        if self.no_duplicates:
            L -= (self.steps-1)*self.skip
        return L
