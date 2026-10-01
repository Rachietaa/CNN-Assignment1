"""Evaluate a saved checkpoint.

Usage:
    python evaluate.py --checkpoint runs/baseline/best.pt               # validation split
    python evaluate.py --checkpoint runs/baseline/best.pt --split test  # final model only
"""
import argparse
import json
from pathlib import Path

import torch

from src.data import build_loaders
from src.engine import evaluate, get_device
from src.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--data-root', help='override data.root stored in the checkpoint')
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg = ckpt['config']
    if args.data_root:
        cfg['data']['root'] = args.data_root

    loaders, classes = build_loaders(cfg, device)
    model = build_model(cfg, num_classes=len(classes))
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)

    result = evaluate(model, loaders[args.split], device, num_classes=len(classes))
    confusion = result['confusion']
    per_class = (confusion.diag().float() / confusion.sum(dim=1).clamp(min=1).float()).tolist()

    print(f"{args.split} loss {result['loss']:.4f} | {args.split} accuracy {result['acc']:.4f}")
    print('Per-class accuracy:')
    for name, acc in sorted(zip(classes, per_class), key=lambda x: x[1]):
        print(f'  {name:<20s} {acc:.3f}')

    out = Path(args.checkpoint).parent / f'eval_{args.split}.json'
    out.write_text(json.dumps({'split': args.split, 'loss': result['loss'], 'acc': result['acc'],
                               'per_class': dict(zip(classes, per_class)),
                               'confusion': confusion.tolist()}, indent=2))
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
