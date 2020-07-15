from torchvision.datasets.folder import default_loader


def load_image(filename, loader=default_loader):
    try:
        img = loader(filename)
    except Exception as e:
        print('Could not load image {:s}'.format(filename))
        print(e)
        return None
    return img
