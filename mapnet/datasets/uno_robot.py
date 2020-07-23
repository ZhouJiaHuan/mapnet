import os
import os.path as osp
import numpy as np
from torch.utils.data import Dataset

from mapnet.utils.pose_utils import process_poses
from mapnet.utils.img_utils import load_image

from ..registry import DATASETS


@DATASETS.register_module(name='UnoRobot')
class UnoRobot(Dataset):
    '''dataset for Uno robot
    the data was collected and processed according to 7-scene
    dataset, specially:

    - UnoRobot/
        - corridor/
            - seq-1/
            - seq-2/
            - ...
            - TestSplit.txt
            - TrainSplit.txt
        - ...

    for each sub-directory in scene, (eg. corridor/seq-1/), a series of
    3 kind files are included:
        - `{d}.color.png`: color frames
        - `{d}.pose.txt`: pose info represented by rotation matrix
        - `{d}.time`: (optional) timestamp
    '''

    def __init__(self, scene, data_path, train,
                 mean_t=tuple([0, 0, 0]), std_t=tuple([1, 1, 1]),
                 transform=None, target_transform=None, seed=7,
                 skip_images=False):
        self.dim3 = True  # for visualization
        self.transform = transform
        self.target_transform = target_transform
        self.skip_images = skip_images
        np.random.seed(seed)

        # directories
        base_dir = osp.join(osp.expanduser(data_path), scene)

        # decide which sequences to use
        if train:
            split_file = osp.join(base_dir, 'TrainSplit.txt')
        else:
            split_file = osp.join(base_dir, 'TestSplit.txt')
        with open(split_file, 'r') as f:
            seqs = [int(l.split('sequence')[-1]) for l in f
                    if not l.startswith('#')]

        # read poses and collect image names
        self.c_imgs = []
        self.gt_idx = np.empty((0,), dtype=np.int)
        ps = {}
        vo_stats = {}
        gt_offset = int(0)
        for seq in seqs:
            seq_dir = osp.join(base_dir, 'seq-{:02d}'.format(seq))
            p_filenames = [n for n in os.listdir(seq_dir) if
                           n.find('.pose.txt') >= 0]
            frame_idx = np.array(range(len(p_filenames)), dtype=np.int)
            pss = [np.loadtxt(osp.join(seq_dir, str(i)+'.pose.txt'))
                   .flatten()[:12] for i in frame_idx]
            ps[seq] = np.asarray(pss)
            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

            self.gt_idx = np.hstack((self.gt_idx, gt_offset+frame_idx))
            gt_offset += len(p_filenames)
            c_imgs = [osp.join(seq_dir, str(i)+'.color.png') for
                      i in frame_idx]
            self.c_imgs.extend(c_imgs)

        mean_t = np.array(mean_t)
        std_t = np.array(std_t)
        self.poses = np.empty((0, 6))
        for seq in seqs:
            vo_stat = vo_stats[seq]
            pss = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                                align_R=vo_stat['R'], align_t=vo_stat['t'],
                                align_s=vo_stat['s'])
            self.poses = np.vstack((self.poses, pss))

    def __getitem__(self, index):
        if self.skip_images:
            img = None
            pose = self.poses[index]
        else:
            img = None
            while img is None:
                img = load_image(self.c_imgs[index])
                pose = self.poses[index]
                index += 1

        if self.target_transform is not None:
            pose = self.target_transform(pose)

        if self.skip_images:
            return img, pose

        if self.transform is not None:
            img = self.transform(img)

        return img, pose

    def __len__(self):
        return self.poses.shape[0]
