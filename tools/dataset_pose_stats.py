"""
compute the pose statics for dataset, only support 7-Scene-like dataset
"""

import glob
import os.path as osp
import numpy as np
import argparse

DATA_DIR = "data/deepslam_data/"


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset pose statistics')
    parser.add_argument('--dataset', type=str, choices=('SevenScenes', 'UnoRobot'),
                        help='Dataset', required=True)
    parser.add_argument('--scene', type=str,
                        help='Scene name', required=True)
    args = parser.parse_args()
    return args


def parse_poses(dataset, scene):
    data_path = osp.join(DATA_DIR, dataset, scene)
    assert osp.exists(data_path)
    split_file = osp.join(data_path, 'TrainSplit.txt')
    assert osp.exists(split_file)
    with open(split_file, 'r') as f:
        seqs = [int(l.split('sequence')[-1]) for l in f
                if not l.startswith('#')]
    pss = []
    for seq in seqs:
        seq_dir = osp.join(data_path, 'seq-{:02d}'.format(seq))
        ps = [np.loadtxt(pose_txt)[:3, 3] for pose_txt in
              glob.glob(osp.join(seq_dir, '*.pose.txt'))]
        pss.extend(ps)
    return np.array(pss)


if __name__ == "__main__":
    args = parse_args()
    pss_array = parse_poses(args.dataset, args.scene)
    print("pose shape = ", pss_array.shape)
    pss_mean = np.mean(pss_array, axis=0)
    pss_std = np.std(pss_array, axis=0)
    print("pose mean = {}".format(pss_mean))
    print("pose std = {}".format(pss_std))
