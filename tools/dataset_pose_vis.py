"""
visualize the pose infomation
"""

import glob
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset pose statistics')
    parser.add_argument('--pose_dir', type=str,
                        help='Pose txt dir', required=True)
    parser.add_argument('--show', action='store_true',
                        help='show the result or not')
    args = parser.parse_args()
    return args


def parse_poses(pose_dir):
    assert osp.exists(pose_dir)

    pss = []
    L = len(glob.glob(osp.join(pose_dir, '*.pose.txt')))
    pose_list = [osp.join(pose_dir, str(i)+'.pose.txt') for i in range(L)]
    ps = [np.loadtxt(pose_txt)[:3, 3] for pose_txt in pose_list]
    pss.extend(ps)
    return np.array(pss)


def draw_poses(pss, show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.scatter(pss[:, 0], pss[:, 1], pss[:, 2], c='g', depthshade=0)
    for i in range(0, pss.shape[0], 100):
        ax.text(pss[i][0], pss[i][1], pss[i][2], i)
    plt.grid()
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    args = parse_args()
    pss_array = parse_poses(args.pose_dir)
    print("pose shape = ", pss_array.shape)
    fig = draw_poses(pss_array, args.show)
