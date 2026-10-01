# ITCS 6169/8169 — Assignment 1: The CNN Challenge

16-class scene recognition from 2,400 training images using a CNN.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.14, PyTorch 2.14.1, torchvision 0.29.1 on Apple Silicon (MPS).
CUDA and CPU are picked up automatically.

## Data

Download the dataset from the course link and put it here:

```text
data/
  train/<class_name>/*.jpg
  test/<class_name>/*.jpg
```

## Usage

```bash
# Train (writes runs/<name>/{best.pt, history.json, summary.json, curves.png})
python train.py --config configs/baseline.yaml

# Repeat with another training seed (same train/val split) -> runs/<name>_s1/
python train.py --config configs/baseline.yaml --seed 1

# Evaluate on the validation split (used for model selection)
python evaluate.py --checkpoint runs/baseline/best.pt

# Evaluate on the test split (final selected model only)
python evaluate.py --checkpoint runs/baseline/best.pt --split test
```

## Validation protocol

The 2,400 training images are split 80/20 (1,920 train / 480 val) with a fixed seed (0),
using the same split as the course starter notebook. The checkpoint with the best validation
accuracy is kept. All experiment choices are made using validation accuracy only; the test
set is used once, for the final model.

## Layout

```text
configs/         one YAML per experiment
src/data.py      dataset, split, transforms
src/models.py    architectures
src/engine.py    training / evaluation loops
train.py         training entry point
evaluate.py      evaluation entry point (val or test, per-class accuracy, confusion matrix)
preview_augmentation.py  saves a grid of augmented training images for a visual check
EXPERIMENTS.md   experiment log
```

## Results

See [EXPERIMENTS.md](EXPERIMENTS.md).
