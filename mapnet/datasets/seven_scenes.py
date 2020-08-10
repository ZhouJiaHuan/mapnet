import os
import os.path as osp
import numpy as np
from torch.utils.data import Dataset

from ..utils.pose_utils import process_poses
from ..utils.img_utils import load_image

from ..registry import DATASETS


@DATASETS.register_module(name='SevenScenes')
class SevenScenes(Dataset):
    '''7-scene dataset
    '''
    DATASET_NAME = 'SevenScenes'

    def __init__(self, scene, data_path, train,
                 mean_t=tuple([0, 0, 0]), std_t=tuple([1, 1, 1]),
                 transform=None, target_transform=None, mode=0,
                 seed=7, skip_images=False):
        '''
        Args:
            scene: scene name ['chess', 'pumpkin', ...]
            data_path: root 7scenes data directory.
                Usually './data/deepslam_data/7Scenes'
            mean_t, std_t: mean and std of the translate (for normalization)
                compute with `tools/dataset_pose_stat.py`
            train: if True, return the training images. If False, returns the
                testing images
            transform: transform to apply to the images
            target_transform: transform to apply to the poses
            mode: 0: just color image, 1: just depth image, 2: [c_img, d_img]
            skip_images: If True, skip loading images and return None instead
        '''
        self.dim3 = True  # for visualization
        self.mode = mode
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
        self.d_imgs = []
        self.gt_idx = np.empty((0,), dtype=np.int)
        ps = {}
        vo_stats = {}
        gt_offset = int(0)
        for seq in seqs:
            seq_dir = osp.join(base_dir, 'seq-{:02d}'.format(seq))
            p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if
                           n.find('pose') >= 0]

            frame_idx = np.array(range(len(p_filenames)), dtype=np.int)
            pss = [np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.
                   format(i))).flatten()[:12] for i in frame_idx]
            ps[seq] = np.asarray(pss)
            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

            self.gt_idx = np.hstack((self.gt_idx, gt_offset+frame_idx))
            gt_offset += len(p_filenames)
            c_imgs = [osp.join(seq_dir, 'frame-{:06d}.color.png'.format(i))
                      for i in frame_idx]
            d_imgs = [osp.join(seq_dir, 'frame-{:06d}.depth.png'.format(i))
                      for i in frame_idx]
            self.c_imgs.extend(c_imgs)
            self.d_imgs.extend(d_imgs)

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
            if self.mode == 0:
                img = None
                while img is None:
                    img = load_image(self.c_imgs[index])
                    pose = self.poses[index]
                    index += 1
                index -= 1
            elif self.mode == 1:
                img = None
                while img is None:
                    img = load_image(self.d_imgs[index])
                    pose = self.poses[index]
                    index += 1
                index -= 1
            elif self.mode == 2:
                c_img = None
                d_img = None
                while (c_img is None) or (d_img is None):
                    c_img = load_image(self.c_imgs[index])
                    d_img = load_image(self.d_imgs[index])
                    pose = self.poses[index]
                    index += 1
                img = [c_img, d_img]
                index -= 1
            else:
                raise Exception('Wrong mode {:d}'.format(self.mode))

        if self.target_transform is not None:
            pose = self.target_transform(pose)

        if self.skip_images:
            return img, pose

        if self.transform is not None:
            if self.mode == 2:
                img = [self.transform(i) for i in img]
            else:
                img = self.transform(img)

        return img, pose

    def __len__(self):
        return self.poses.shape[0]
