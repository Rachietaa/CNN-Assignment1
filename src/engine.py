"""Training / evaluation loops and shared utilities."""
import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


@torch.inference_mode()
def evaluate(model, loader, device, num_classes=None, tta=False):
    """Return loss, accuracy, and (optionally) a confusion matrix [true, pred].

    tta=True: test-time augmentation — average the predicted probabilities for each image and its
    horizontal mirror (the only flip that preserves scene meaning, see Step 2d).
    """
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long) if num_classes else None

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        probs = model(images).softmax(dim=1)
        if tta:
            probs = (probs + model(images.flip(dims=[3])).softmax(dim=1)) / 2
        loss_sum += nn.functional.nll_loss(probs.clamp_min(1e-12).log(), labels, reduction='sum').item()
        preds = probs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if confusion is not None:
            for t, p in zip(labels.cpu(), preds.cpu()):
                confusion[t, p] += 1

    return {'loss': loss_sum / total, 'acc': correct / total, 'confusion': confusion}


def build_optimizer(cfg, model):
    opt_cfg = cfg['train']
    name = opt_cfg['optimizer'].lower()
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=opt_cfg['lr'],
                                weight_decay=opt_cfg.get('weight_decay', 0.0))
    if name == 'adamw':
        # Pretrained layers get a smaller LR (backbone_lr) than the newly initialized head (lr),
        # so fine-tuning adjusts the ImageNet features gently. Frozen parameters are left out.
        trainable = [p for p in model.parameters() if p.requires_grad]
        groups = [{'params': trainable, 'lr': opt_cfg['lr']}]
        if 'backbone_lr' in opt_cfg and hasattr(model, 'head_parameters'):
            head_ids = {id(p) for p in model.head_parameters()}
            groups = [
                {'params': [p for p in trainable if id(p) in head_ids], 'lr': opt_cfg['lr']},
                {'params': [p for p in trainable if id(p) not in head_ids], 'lr': opt_cfg['backbone_lr']},
            ]
        return torch.optim.AdamW(groups, weight_decay=opt_cfg.get('weight_decay', 0.01))
    raise ValueError(f'Unknown optimizer: {name}')


def build_scheduler(cfg, optimizer, steps_per_epoch):
    """Optional LR schedule, stepped once per batch. Returns None if not configured."""
    name = cfg['train'].get('scheduler')
    if name is None:
        return None
    if name == 'cosine':
        # Decays the LR from its initial value to 0 over the whole run.
        total_steps = cfg['train']['epochs'] * steps_per_epoch
        warmup_steps = cfg['train'].get('warmup_epochs', 0) * steps_per_epoch
        if warmup_steps == 0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        # Linear warm-up from ~0 to the full LR, then cosine decay to 0. The warm-up keeps the first,
        # large updates (driven by the randomly initialized head) from damaging pretrained weights.
        def factor(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    raise ValueError(f'Unknown scheduler: {name}')


def build_mixer(mix_cfg, num_classes):
    """Mixup or CutMix (chosen at random per batch) from torchvision.transforms.v2. Returns None if unset."""
    if not mix_cfg:
        return None
    from torchvision.transforms import v2
    return v2.RandomChoice([v2.MixUp(alpha=mix_cfg['mixup_alpha'], num_classes=num_classes),
                            v2.CutMix(alpha=mix_cfg['cutmix_alpha'], num_classes=num_classes)])


def train_model(model, loaders, optimizer, epochs, device, scheduler=None,
                label_smoothing=0.0, mix_cfg=None, num_classes=16):
    """Train and keep the checkpoint with the best validation accuracy.

    label_smoothing: target probability 1-ε on the true class, ε spread over the others (train loss only).
    mix_cfg: {'p', 'mixup_alpha', 'cutmix_alpha'} — with probability p a batch is mixed with Mixup or
    CutMix, which turns its labels into soft label vectors. Train accuracy is then measured against the
    original (dominant) labels, so it is only approximate on mixed batches.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    mixer = build_mixer(mix_cfg, num_classes)
    model = model.to(device)
    best_state, best_val_acc, best_epoch = copy.deepcopy(model.state_dict()), 0.0, 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        history['lr'].append(optimizer.param_groups[0]['lr'])  # LR at the start of the epoch
        loss_sum, correct, seen = 0.0, 0, 0
        for images, labels in loaders['train']:
            targets = labels
            if mixer is not None and random.random() < mix_cfg['p']:
                images, targets = mixer(images, labels)
            images, labels, targets = images.to(device), labels.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            loss_sum += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)

        val = evaluate(model, loaders['val'], device)
        history['train_loss'].append(loss_sum / seen)
        history['train_acc'].append(correct / seen)
        history['val_loss'].append(val['loss'])
        history['val_acc'].append(val['acc'])

        if val['acc'] > best_val_acc:
            best_val_acc, best_epoch = val['acc'], epoch
            best_state = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch:03d}/{epochs} | '
              f'train loss {loss_sum / seen:.4f} acc {correct / seen:.4f} | '
              f'val loss {val["loss"]:.4f} acc {val["acc"]:.4f} | '
              f'{time.time() - start:.1f}s')

    model.load_state_dict(best_state)
    print(f'Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})')
    return model, history, best_val_acc, best_epoch, time.time() - start
