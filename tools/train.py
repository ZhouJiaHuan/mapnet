"""
training codes.
"""
import argparse
from mmcv import Config
import sys
sys.path.append('.')
from mapnet.apis import Trainer
from mapnet.datasets import build_transforms
from mapnet.builder import build_backbone, build_dataset, build_model
from mapnet.builder import build_loss, build_optimizer


def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', type=str,
                        help='configuration file')
    parser.add_argument('--logdir', type=str,
                        help='Experiment work directory')
    parser.add_argument('--device', type=str, default='0',
                        help='value to be set to $CUDA_VISIBLE_DEVICES')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to resume from')
    parser.add_argument('--resume_optim', action='store_true',
                        help='Resume optimization for checkpoint')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.logdir, "work dir is not specified!"
    cfgs = Config.fromfile(args.config)

    # build model with cfg
    model_cfgs = cfgs.model
    backbone_cfg = model_cfgs.backbone
    backbone = build_backbone(backbone_cfg)
    model_cfgs.backbone = backbone
    model = build_model(model_cfgs)

    # build loss with cfg
    train_loss_cfgs = cfgs.train_loss
    val_loss_cfgs = cfgs.val_loss
    train_criterion = build_loss(train_loss_cfgs)
    val_criterion = build_loss(val_loss_cfgs)

    # build optimizer with cfg
    optim_cfgs = cfgs.optim
    param_list = [{'params': model.parameters()}]
    if train_loss_cfgs.get('learn_beta', False):
        assert hasattr(train_criterion, 'sax')
        assert hasattr(train_criterion, 'saq')
        params = {'params': [train_criterion.sax, train_criterion.saq]}
        param_list.append(params)
    if train_loss_cfgs.get('learn_gamma', False):
        assert hasattr(train_criterion, 'srx')
        assert hasattr(train_criterion, 'srq')
        params = {'params': [train_criterion.srx, train_criterion.srq]}
        param_list.append(params)
    optim_cfgs.params = param_list
    optimizer = build_optimizer(optim_cfgs)

    # build dataset with cfg
    tform_cfgs = cfgs.train_transform
    data_trans, target_trans = build_transforms(tform_cfgs)
    data_cfgs = cfgs.dataset
    data_cfgs.transform = data_trans
    data_cfgs.target_transform = target_trans
    data_cfgs.train = True
    train_set = build_dataset(data_cfgs)
    data_cfgs.train = False
    val_set = build_dataset(data_cfgs)

    # configure Trainer
    train_cfgs = cfgs.common
    train_cfgs.logdir = args.logdir
    train_cfgs.model = model
    train_cfgs.optimizer = optimizer
    train_cfgs.train_criterion = train_criterion
    train_cfgs.val_criterion = val_criterion
    train_cfgs.train_set = train_set
    train_cfgs.val_set = val_set
    train_cfgs.device = args.device
    trainer = Trainer(train_cfgs,
                      checkpoint_file=args.checkpoint,
                      resume_optim=args.resume_optim)
    trainer.train_val()


if __name__ == "__main__":
    main()
