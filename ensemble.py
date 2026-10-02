"""Evaluate an ensemble: average the predicted class probabilities of several checkpoints.

Usage:
    python ensemble.py --out runs/step5e_ensemble \
        --checkpoints runs/step5c2_ls_mix/best.pt runs/step5c2_ls_mix_s1/best.pt runs/step5c2_ls_mix_s2/best.pt
    python ensemble.py ... --split test      # final model only, once

All checkpoints must use the same data settings (resolution, normalization, gray3), since one data
loader feeds all of them. Each member is also scored on its own for comparison.
"""
import argparse
import json
from pathlib import Path

import torch

from src.data import build_loaders
from src.engine import get_device
from src.models import build_model


@torch.inference_mode()
def predict_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    for images, y in loader:
        probs.append(model(images.to(device)).softmax(dim=1).cpu())
        labels.append(y)
    return torch.cat(probs), torch.cat(labels)


def scores(probs, labels, classes):
    preds = probs.argmax(dim=1)
    n = len(classes)
    confusion = torch.zeros(n, n, dtype=torch.long)
    for t, p in zip(labels, preds):
        confusion[t, p] += 1
    per_class = (confusion.diag().float() / confusion.sum(dim=1).clamp(min=1).float()).tolist()
    loss = torch.nn.functional.nll_loss(probs.clamp_min(1e-12).log(), labels).item()
    return {'acc': (preds == labels).float().mean().item(), 'loss': loss,
            'errors': int((preds != labels).sum()), 'per_class': dict(zip(classes, per_class)),
            'confusion': confusion.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--out', required=True, help='directory for the ensemble results')
    parser.add_argument('--data-root', help='override data.root stored in the checkpoints')
    args = parser.parse_args()

    device = get_device()
    ckpts = [torch.load(p, map_location='cpu') for p in args.checkpoints]
    # Compare data settings, using the effective split seed (runs trained with --seed store it as
    # data.split_seed; seed-0 runs fall back to the top-level seed).
    data_cfgs = [{**{k: v for k, v in c['config']['data'].items() if k != 'root'},
                  'split_seed': c['config']['data'].get('split_seed', c['config']['seed'])} for c in ckpts]
    assert all(d == data_cfgs[0] for d in data_cfgs), 'checkpoints use different data settings'

    cfg = ckpts[0]['config']
    if args.data_root:
        cfg['data']['root'] = args.data_root
    loaders, classes = build_loaders(cfg, device)

    member_probs, members = [], {}
    for path, ckpt in zip(args.checkpoints, ckpts):
        model = build_model(ckpt['config'], num_classes=len(classes))
        model.load_state_dict(ckpt['state_dict'])
        probs, labels = predict_probs(model.to(device), loaders[args.split], device)
        member_probs.append(probs)
        members[path] = scores(probs, labels, classes)
        print(f"{path}: {args.split} acc {members[path]['acc']:.4f} ({members[path]['errors']} errors)")
        del model

    result = scores(torch.stack(member_probs).mean(dim=0), labels, classes)
    print(f"Ensemble of {len(ckpts)}: {args.split} acc {result['acc']:.4f} ({result['errors']} errors), "
          f"loss {result['loss']:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f'eval_{args.split}.json').write_text(json.dumps(
        {'split': args.split, 'checkpoints': args.checkpoints, **result,
         'members': {p: {k: m[k] for k in ('acc', 'loss', 'errors')} for p, m in members.items()}},
        indent=2))
    print(f"Saved {out / f'eval_{args.split}.json'}")


if __name__ == '__main__':
    main()
