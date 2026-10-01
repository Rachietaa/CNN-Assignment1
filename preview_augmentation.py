"""Save a grid showing original training images next to random augmented versions.

Usage:
    python preview_augmentation.py --config configs/step2b_augment.yaml
Writes runs/<name>/augment_preview.png. Use it to check that augmented images still look like real scenes.
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from torchvision import datasets

from src.data import build_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--n-images', type=int, default=6)
    parser.add_argument('--n-augments', type=int, default=5)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_cfg = cfg['data']
    plain = build_transform(data_cfg, train=False)
    augmented = build_transform(data_cfg, train=True)
    dataset = datasets.ImageFolder(Path(data_cfg['root']) / 'train')

    random.seed(cfg['seed'])
    indices = random.sample(range(len(dataset)), args.n_images)
    mean, std = data_cfg['mean'][0], data_cfg['std'][0]
    cols = args.n_augments + 1
    fig, axes = plt.subplots(args.n_images, cols, figsize=(1.8 * cols, 1.9 * args.n_images))
    for row, idx in enumerate(indices):
        image, label = dataset[idx]
        views = [plain(image)] + [augmented(image) for _ in range(args.n_augments)]
        for col, view in enumerate(views):
            ax = axes[row, col]
            ax.imshow((view * std + mean).permute(1, 2, 0).squeeze().clamp(0, 1), cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if col == 0:
                ax.set_title(f'{dataset.classes[label]}\n(original)', fontsize=8)
            elif row == 0:
                ax.set_title(f'aug {col}', fontsize=8)
    fig.tight_layout()
    out = Path('runs') / cfg['name'] / 'augment_preview.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
