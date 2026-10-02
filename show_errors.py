"""Save a grid of the images a model (or ensemble) gets wrong, for failure analysis.

Usage:
    python show_errors.py --split test --out runs/final_ensemble/test_errors.png \
        --checkpoints runs/step5c2_ls_mix/best.pt runs/step5c2_ls_mix_s1/best.pt runs/step5c2_ls_mix_s2/best.pt
Analysis only: it does not change or select any model.
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image

from ensemble import predict_probs
from src.data import build_loaders
from src.engine import get_device
from src.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--out', required=True, help='output PNG path')
    args = parser.parse_args()

    device = get_device()
    ckpts = [torch.load(p, map_location='cpu') for p in args.checkpoints]
    loaders, classes = build_loaders(ckpts[0]['config'], device)
    loader = loaders[args.split]

    probs = 0
    for ckpt in ckpts:
        model = build_model(ckpt['config'], num_classes=len(classes))
        model.load_state_dict(ckpt['state_dict'])
        p, labels = predict_probs(model.to(device), loader, device)
        probs = probs + p / len(ckpts)
    preds = probs.argmax(dim=1)
    wrong = (preds != labels).nonzero().flatten().tolist()
    print(f'{len(wrong)} errors out of {len(labels)} ({(preds == labels).float().mean():.4f} accuracy)')

    # Map loader positions back to image files (val is a Subset of the train folder).
    dataset = loader.dataset
    indices = dataset.indices if hasattr(dataset, 'indices') else range(len(dataset))
    samples = dataset.dataset.samples if hasattr(dataset, 'dataset') else dataset.samples
    files = [samples[i][0] for i in indices]

    cols = 6
    rows = max(1, (len(wrong) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.9 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(Image.open(files[i]).convert('L'), cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'true: {classes[labels[i]]}\npred: {classes[preds[i]]} ({probs[i, preds[i]]:.2f})',
                     fontsize=8)
        print(f'{files[i]}: true {classes[labels[i]]}, predicted {classes[preds[i]]} '
              f'(p={probs[i, preds[i]]:.2f}, p(true)={probs[i, labels[i]]:.2f})')
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
