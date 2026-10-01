"""Train a model from a YAML config.

Usage:
    python train.py --config configs/baseline.yaml
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml

from src.data import build_loaders
from src.engine import build_optimizer, get_device, set_seed, train_model
from src.models import build_model


def plot_history(history, out_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history['train_loss'], label='train')
    ax1.plot(epochs, history['val_loss'], label='val')
    ax1.set_xlabel('epoch'); ax1.set_ylabel('loss'); ax1.legend()
    ax2.plot(epochs, history['train_acc'], label='train')
    ax2.plot(epochs, history['val_acc'], label='val')
    ax2.set_xlabel('epoch'); ax2.set_ylabel('accuracy'); ax2.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-root', help='override data.root from the config')
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data_root:
        cfg['data']['root'] = args.data_root

    set_seed(cfg['seed'])
    device = get_device()
    print('Device:', device)

    loaders, classes = build_loaders(cfg, device)
    print(f'Classes ({len(classes)}): {classes}')
    print(f"Train {len(loaders['train'].dataset)} | Val {len(loaders['val'].dataset)} | "
          f"Test {len(loaders['test'].dataset)}")

    model = build_model(cfg, num_classes=len(classes))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f'Trainable parameters: {n_params:,}')

    optimizer = build_optimizer(cfg, model)
    model, history, best_val_acc, best_epoch, seconds = train_model(
        model, loaders, optimizer, cfg['train']['epochs'], device)

    # The test set is deliberately NOT evaluated here; use evaluate.py on the final model only.
    out_dir = Path('runs') / cfg['name']
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'config': cfg, 'classes': classes,
                'best_val_acc': best_val_acc, 'best_epoch': best_epoch},
               out_dir / 'best.pt')
    (out_dir / 'history.json').write_text(json.dumps(history, indent=2))
    summary = {'name': cfg['name'], 'best_val_acc': best_val_acc, 'best_epoch': best_epoch,
               'params': n_params, 'train_seconds': round(seconds, 1), 'device': str(device),
               'torch': torch.__version__}
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    plot_history(history, out_dir / 'curves.png')
    print(f'Saved checkpoint, history and curves to {out_dir}/')


if __name__ == '__main__':
    main()
