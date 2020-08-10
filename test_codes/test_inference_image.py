"""
one frame inference with mapnet / posenet
"""
import os
import sys
import glob
import time
import numpy as np
from mmcv import Config
sys.path.append('.')
from mapnet.apis import Inference
from mapnet.utils.img_utils import load_image


def main():
    img_dir = "data/deepslam_data/UnoRobot/corridor/seq-04/"
    img_list = glob.glob(img_dir+"*.color.png")
    weights = "logs/UnoRobot/mapnet_corridor/epoch_300.pth.tar"
    cfg_path = "configs/UnoRobot/mapnet_corridor.yaml"
    cfgs = Config.fromfile(cfg_path)
    print("config file loaded from {}".format(cfg_path))
    t0 = time.time()
    infer = Inference(cfgs, weights)
    print("model loading time: {:.3f} s".format(time.time()-t0))

    assert os.path.exists(img_dir), "{} not found!".format(img_dir)

    t_load = []
    t_pred = []
    for i, img_path in enumerate(sorted(img_list)):
        if i % 100 == 0:
            print("process {} / {} images".format(i, len(img_list)))
        t1 = time.time()
        img = load_image(img_path)
        t2 = time.time()
        pred_poses = infer.test_image(img)
        t3 = time.time()
        t_load.append(t2-t1)
        t_pred.append(t3-t2)
    print("finished.")
    print("average image loading time: {:.3f}".format(np.mean(t_load)))
    print("average prediction time: {:.3f}".format(np.mean(t_pred)))


if __name__ == "__main__":
    main()
