"""Dataset loading, train/val split, and transforms."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def build_transform(cfg, train):
    """Build the image transform from the `data` section of a config.

    `train=True` returns the training transform (augmentation goes here in
    later experiments); `train=False` returns the deterministic eval transform.
    """
    size = cfg['img_size']
    aug = (cfg.get('augment') or {}) if train else {}  # never augment val/test
    t = []
    if cfg.get('grayscale', False):
        t.append(transforms.Grayscale(num_output_channels=1))
    elif cfg.get('gray3', False):
        # Gray copied into 3 channels: fits 3-channel pretrained models without the Flower color shortcut.
        t.append(transforms.Grayscale(num_output_channels=3))
    elif cfg.get('strip_color', False) and not train:
        # Analysis only: remove color but keep 3 channels, to test whether an RGB model relies on color.
        t.append(transforms.Grayscale(num_output_channels=3))

    if aug.get('rotation'):
        # Rotate at full resolution with bilinear interpolation (rotating after the 64px resize with
        # the default nearest interpolation produced jagged staircase artifacts). Corners are filled
        # with mid-gray rather than black, and the crop below usually removes most of them.
        t.append(transforms.RandomRotation(degrees=aug['rotation'],
                                           interpolation=transforms.InterpolationMode.BILINEAR,
                                           fill=128))
    if 'random_resized_crop' in aug:
        # Crop a random region covering `scale` of the image area, then resize to size×size.
        t.append(transforms.RandomResizedCrop(size, scale=tuple(aug['random_resized_crop'])))
    else:
        t.append(transforms.Resize((size, size)))
    if aug.get('hflip'):
        t.append(transforms.RandomHorizontalFlip(p=aug['hflip']))
    if aug.get('vflip'):
        t.append(transforms.RandomVerticalFlip(p=aug['vflip']))
    if aug.get('brightness') or aug.get('contrast'):
        t.append(transforms.ColorJitter(brightness=aug.get('brightness', 0),
                                        contrast=aug.get('contrast', 0)))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=cfg['mean'], std=cfg['std']))
    if 'random_erasing' in aug:
        # Cutout-style: blank out one random rectangle. Runs after Normalize, so value=0 is the
        # dataset mean (mid-gray) rather than black.
        er = aug['random_erasing']
        t.append(transforms.RandomErasing(p=er['p'], scale=tuple(er['scale']), value=0))
    return transforms.Compose(t)


def split_indices(n, val_fraction, seed):
    """Same split as `random_split` in the starter notebook (seeded randperm)."""
    val_size = int(round(n * val_fraction))
    train_size = n - val_size
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return perm[:train_size], perm[train_size:]


def build_loaders(cfg, device):
    data_cfg = cfg['data']
    root = Path(data_cfg['root'])
    train_tf = build_transform(data_cfg, train=True)
    eval_tf = build_transform(data_cfg, train=False)

    # Two views of the same folder so train and val can use different transforms.
    train_view = datasets.ImageFolder(root / 'train', transform=train_tf)
    val_view = datasets.ImageFolder(root / 'train', transform=eval_tf)
    test_set = datasets.ImageFolder(root / 'test', transform=eval_tf)

    # split_seed keeps the train/val split fixed when only the training seed is varied.
    split_seed = data_cfg.get('split_seed', cfg['seed'])
    train_idx, val_idx = split_indices(len(train_view), data_cfg['val_fraction'], split_seed)
    train_set = Subset(train_view, train_idx)
    val_set = Subset(val_view, val_idx)

    kw = dict(batch_size=cfg['train']['batch_size'],
              num_workers=data_cfg.get('num_workers', 2),
              pin_memory=device.type == 'cuda',
              persistent_workers=data_cfg.get('num_workers', 2) > 0)
    loaders = {
        'train': DataLoader(train_set, shuffle=True, **kw),
        'val': DataLoader(val_set, shuffle=False, **kw),
        'test': DataLoader(test_set, shuffle=False, **kw),
    }
    return loaders, train_view.classes
