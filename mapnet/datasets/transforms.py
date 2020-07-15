import torch
from torchvision import transforms


def build_transforms(mean, std, size=256, color_jitter=0.7):
    '''transforms for input image and target pose
    '''
    tforms = [transforms.Resize(size)]
    if color_jitter > 0 and color_jitter <= 1.0:
        jitter_tforms = transforms.ColorJitter(brightness=color_jitter,
                                               contrast=color_jitter,
                                               saturation=color_jitter,
                                               hue=0.5)
        tforms.append(jitter_tforms)

    tforms.append(transforms.ToTensor())
    tforms.append(transforms.Normalize(mean=mean, std=std))
    data_transform = transforms.Compose(tforms)
    target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

    return data_transform, target_transform
