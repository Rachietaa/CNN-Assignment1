"""Training / evaluation loops and shared utilities."""
import copy
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
def evaluate(model, loader, device, num_classes=None):
    """Return loss, accuracy, and (optionally) a confusion matrix [true, pred]."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    loss_sum, correct, total = 0.0, 0, 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long) if num_classes else None

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item()
        preds = logits.argmax(dim=1)
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
    raise ValueError(f'Unknown optimizer: {name}')


def train_model(model, loaders, optimizer, epochs, device):
    """Train and keep the checkpoint with the best validation accuracy."""
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    best_state, best_val_acc, best_epoch = copy.deepcopy(model.state_dict()), 0.0, 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for images, labels in loaders['train']:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

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
