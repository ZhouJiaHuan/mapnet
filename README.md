This is the PyTorch implementation of the `MapNet` for 

"[Geometry-Aware Learning of Maps for Camera Localization](https://arxiv.org/abs/1712.03342)" - CVPR 2018 (Spotlight).

`MapNet++` and `MapNet+PGO` are **NOT** implemented in this repository. 

The code was forked from official PyTorch implementation: https://github.com/NVlabs/geomapnet

# dependences:

- Python 3.6
- PyTorch 1.5.1 + Torchvision 0.6.1
- mmcv
- numpy
- matplotlib
- scipy
- colour-demosaicing
- transforms3d

# Main work

This repository made the following modifications compared to official implements:

- update Python 2.7 to Python 3.6

- update PyTorch 0.4.1 to PyTorch 1.5.1

- Manage the classes (DATASETS, MODELS, BACKBONES, CRITERION, OPTIMIZER, PIPELINES) with a unified Register in `mmcv` for code simplicity

- Parameters are all specified with `.yaml` configure files. For example, the configure for training and testing the `mapnet` on `SevenScenes` is as follows:

  ```yaml
  common:
    n_epochs: 300
    batch_size: 20
    do_val: True
    seed: 7
    shuffle: True
    num_workers: 5
    snapshot: 50
    val_freq: 50
    max_grad_norm: 0
    print_freq: 20
    cuda: True
  
  model:
    type: MapNet
    backbone:
      type: ResNet34
      pretrained: True
    pretrained: True
    droprate: 0.5
    feat_dim: 2048
  
  train_transform:
    - type: 'Resize'
      size: 256
    - type: 'ColorJitter'
      brightness: 0.7
      contrast: 0.7
      saturation: 0.7
      hue: 0.5
    - type: 'ToTensor'
    - type: 'Normalize'
      mean:
        - 0.4943200
        - 0.4268097
        - 0.4339257
      std:
        - 0.20226764
        - 0.21839666
        - 0.20558356
  
  val_transform:
    - type: 'Resize'
      size: 256
    - type: 'ToTensor'
    - type: 'Normalize'
      mean:
        - 0.4943200
        - 0.4268097
        - 0.4339257
      std:
        - 0.20226764
        - 0.21839666
        - 0.20558356
  
  dataset:
    type: MF
    name: SevenScenes
    no_duplicates: False
    scene: chess
    data_path: data/deepslam_data/SevenScenes
    mean_t:
      - 0.0
      - 0.0
      - 0.0
    std_t:
      - 1.0
      - 1.0
      - 1.0
    mode: 0
    skip_images: False
    skip: 10
    steps: 3
    variable_skip: False
  
  train_loss:
    type: MapNetCriterion
    sax: 0.0
    saq: -3.0
    srx: 0.0
    srq: -3.0
    learn_beta: True
    learn_gamma: True
  
  val_loss:
    type: MapNetCriterion
    sax: 0.0
    saq: 0.0
    srx: 0.0
    srq: 0.0
    learn_beta: False
    learn_gamma: False
  
  optim:
    type: Optimizer
    method: adam
    base_lr: 1.0e-4
    weight_decay: 0.0005
  ```


- use `Tensorboard` for training visualization.

  ![tensorboard2](images/tensorboard2.png)

![tensorboard1](images/tensorboard1.png) 

# Experiments

For training and testing the model, using the command below:

- training `mapnet` on `SevenScenes` `chess`

```shell
python tools/train.py --config_file configs/7scenes/mapnet_chess.yaml --logdir logs/mapnet_chess/
```

The `logdir` will be created automatically which includes the following information:

1. training log file.
2. checkpoints every N epochs (default 50).
3. Tensorboard events for visualizing the training process.

- run validation process with `mapnet` on `SevenScenes` `chess`

```shell
python tools/eval.py --config_file configs/7scenes/mapnet_chess.yaml --weights logs/mapnet_chess/epoch_200.pth.tar --show --val
```

where `--show` means drawing result and `--val` means run validation on val set. For more details about the arguments of `eval.py`, please run:

```shell
python tools/eval.py --help
```

Experimental result with `mapnet` on `SevenScenes` :

| Scene       | official result      | result                            |
| ----------- | -------------------- | --------------------------------- |
| Chess       | 0.08 m, 3.25 degree  | 0.13 m, 4.87 degree (250 epochs)  |
| Fire        | 0.27 m, 11.69 degree | 0.37 m, 11.93 degree (250 epochs) |
| Heads       | 0.18 m, 13.25 degree | 0.19 m, 13.23 degree              |
| Office      | 0.17 m, 5.15 degree  | 0.22 m, 6.56 degree               |
| Pumpkin     | 0.22 m, 4.02 degree  | 0.28 m, 5.97 degree (200 epochs)  |
| Kitchen     | 0.23 m, 4.93 degree  | 0.28 m, 5.83 degree (200 epochs)  |
| Stairs      | 0.30 m, 12.08 degree | 0.35 m, 10.99 degree              |
| **Average** | 0.21 m, 7.77 degree  | 0.26 m, 8.48 degree               |

Experimental result with `AtLoc` on `SevenScenes` :

| Scene | model     | -crop, size 256 x 256 | + crop, size 224 x 224 |
| ----- | --------- | --------------------- | ---------------------- |
| Chess | AtLoc     | 0.12 m, 5.13 degree   | 0.12 m / 8.50 degree   |
| Chess | AtLocPlus | 0.10 m , 4.13 degree  | 0.12 m / 9.01 degree   |
| Heads | AtLoc     | 0.17 m, 13.88 degree  | 0.20 m / 15.35 degree  |
| Heads | AtLocPlus | 0.20 m, 13.18 degree  | 0.16 m / 14.51 degree  |



Experimental results (median / mean error) on our own data `UnoRobot`:

| Model          | Backbone  | translate_x (m) | translate_y (m) | translate (m)   | rotation (degree) |
| -------------- | --------- | --------------- | --------------- | --------------- | ----------------- |
| PoseNet        | ResNet-34 | 0.22 / 0.50     | 0.26 / 0.44     | 0.42 / 0.74     | 5.49 / 9.71       |
| MapNet         | ResNet-34 | 0.28 / 0.37     | 0.23 / 0.32     | 0.46 / 0.56     | **4.40**  / 8.27  |
| AtLoc          | ResNet-34 | 0.24 / 0.52     | 0.28 / 0.43     | 0.47 / 0.77     | 5.04 / 9.01       |
| AtLoc++        | ResNet-34 | 0.21 / 0.30     | 0.27 / 0.39     | 0.45 / 0.56     | 5.59 / 9.82       |
| PoseNet + crop | ResNet-34 | 0.30 / 0.40     | 0.27 / 0.35     | 0.54 / 0.60     | 4.82 / **6.92**   |
| MapNet + crop  | ResNet-34 | 0.22 / 0.29     | **0.21 / 0.31** | **0.40 / 0.48** | 5.55 / 7.55       |
| AtLoc + crop   | ResNet-34 | **0.20 / 0.26** | 0.29 / 0.35     | 0.46 / 0.50     | 5.25 / 7.75       |
| AtLoc++ + crop | ResNet-34 | 0.22 / 0.28     | 0.22 / 0.34     | 0.41 / 0.50     | 5.48 / 7.99       |
| PoseNet        | ResNet-18 | 0.28 / 0.42     | 0.26 / 0.40     | 0.50 /0.66      | 5.48 / 10.30      |
| MapNet         | ResNet-18 | 0.26 / 0.38     | 0.22 / 0.36     | 0.46 / 0.60     | 4.72 / 10.21      |
| AtLoc          | ResNet-18 | 0.22 / 0.36     | 0.26 / 0.41     | 0.45 / 0.63     | 5.37 / 10.24      |
| AtLoc++        | ResNet-18 | **0.20** / 0.32 | 0.26 / 0.41     | 0.41 / 0.58     | 4.87 / 9.20       |
| PoseNet        | ResNet-50 | 0.24 / 0.50     | 0.30 / 0.40     | 0.48 / 0.73     | 5.56 / 11.70      |
| MapNet         | ResNet-50 | 0.28 / 0.44     | 0.26 / 0.37     | 0.49 / 0.65     | 4.75 / 8.26       |

- PoseNet

![posenet_epoch-300_val](images/posenet_epoch-300_val.png)

- MapNet

![mapnet_epoch-300_val](images/mapnet_epoch-300_val.png)

- AtLoc![atloc_epoch-300_val](images/atloc_epoch-300_val.png)

- AtlocPlus

  ![atlocplus_epoch-300_val](images/atlocplus_epoch-300_val.png)