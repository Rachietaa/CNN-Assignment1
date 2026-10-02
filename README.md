# ITCS 6169/8169 — Assignment 1: The CNN Challenge

16-class scene recognition from 2,400 labelled training images with a convolutional neural network.

## Final result

| | Accuracy |
|---|---:|
| **Test (400 images, evaluated once on the final model)** | **95.75%** (383 / 400) |
| Validation (480 images, used for all model selection) | 97.7% |
| Starter baseline (TNet), validation | 48.1% |

**Final model:** an ensemble of three ConvNeXt V2-Tiny CNNs (same recipe, seeds 0/1/2), whose predicted
class probabilities are averaged. The full experimental history, including failed experiments and the
test-set failure analysis, is in [EXPERIMENTS.md](EXPERIMENTS.md).

## Final recipe

| Component | Choice |
|---|---|
| Architecture | ConvNeXt V2-Tiny (pure CNN, 27.9M parameters), new 768→16 classification layer |
| Pretraining | timm `convnextv2_tiny.fcmae_ft_in1k`: FCMAE (masked-autoencoder) self-supervised pretraining on ImageNet-1k, then supervised ImageNet-1k fine-tuning. **All layers fine-tuned.** |
| Input | 224×224, grayscale copied into 3 channels (no color: only the Flower class is in color, which the model would use as a shortcut), ImageNet mean/std |
| Augmentation (train only) | rotation ±10°, random resized crop 70–100%, horizontal flip, brightness/contrast ±20% |
| Batch mixing | Mixup (α 0.2) or CutMix (α 1.0) on 50% of batches |
| Loss | cross-entropy with label smoothing 0.1 |
| Optimizer | AdamW, weight decay 0.05; LR 1e-3 for the new head, 1e-4 for pretrained layers |
| Schedule | 2-epoch linear warm-up, cosine decay to 0, 30 epochs, batch size 64, stochastic depth 0.1 |
| Model selection | checkpoint with the best validation accuracy in each run; final model = ensemble of 3 seeds |
| Config | [configs/step5c2_ls_mix.yaml](configs/step5c2_ls_mix.yaml) |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Hardware and software used:
- Steps 0–4 and all evaluation: MacBook (Apple M5, 32 GB), PyTorch 2.14.1 on MPS, Python 3.14.
- Step 5 training: Google Colab, NVIDIA A100 40 GB, PyTorch 2.11.0 (CUDA). A control experiment showed
  both machines give the same results within seed noise (see Step 5 in EXPERIMENTS.md).

The device is chosen automatically (CUDA, then Apple MPS, then CPU). Random seeds are fixed in every config.

## Data

Download the dataset from the course link and unpack it so that:

```text
data/
  train/<class_name>/*.jpg   2,400 images, 150 per class (split 80/20 into train/val by the code)
  test/<class_name>/*.jpg    400 images, 25 per class
  test2/*.jpg                400 unlabeled images
```

## Checkpoints

The three final checkpoints (~110 MB each) are attached to the
[v1.0 GitHub release](https://github.com/Rachietaa/CNN-Assignment1/releases/tag/v1.0).
Download them into the run folders:

```bash
BASE=https://github.com/Rachietaa/CNN-Assignment1/releases/download/v1.0
mkdir -p runs/step5c2_ls_mix runs/step5c2_ls_mix_s1 runs/step5c2_ls_mix_s2
curl -L $BASE/final_seed0.pt -o runs/step5c2_ls_mix/best.pt
curl -L $BASE/final_seed1.pt -o runs/step5c2_ls_mix_s1/best.pt
curl -L $BASE/final_seed2.pt -o runs/step5c2_ls_mix_s2/best.pt
```

## Reproducing the final result

**Evaluate the final ensemble** (with the downloaded checkpoints):

```bash
CKPTS="runs/step5c2_ls_mix/best.pt runs/step5c2_ls_mix_s1/best.pt runs/step5c2_ls_mix_s2/best.pt"
python ensemble.py --split val  --out runs/step5e_ensemble --checkpoints $CKPTS   # 97.7%
python ensemble.py --split test --out runs/final_ensemble --checkpoints $CKPTS   # 95.75%
python show_errors.py --split test --out runs/final_ensemble/test_errors.png --checkpoints $CKPTS
```

**Train the final models from scratch** (≈5 min per seed on an A100; much slower on a laptop GPU):

```bash
python train.py --config configs/step5c2_ls_mix.yaml            # seed 0 -> runs/step5c2_ls_mix/
python train.py --config configs/step5c2_ls_mix.yaml --seed 1   # -> runs/step5c2_ls_mix_s1/
python train.py --config configs/step5c2_ls_mix.yaml --seed 2   # -> runs/step5c2_ls_mix_s2/
```

`--seed` changes only the training randomness (initialization, batch order, augmentation); the
train/validation split stays fixed. On Colab, `colab_step5.ipynb` runs all of Step 5 via `run_step5.py`
and stores results on Google Drive.

**Any earlier experiment** can be reproduced the same way from its config, e.g.
`python train.py --config configs/step2b_augment_60ep.yaml`, then
`python evaluate.py --checkpoint runs/step2b_augment_60ep/best.pt`.

## Validation protocol

- The 2,400 training images are split 80/20 (1,920 train / 480 validation) with a fixed seed (0),
  the same split as the course starter notebook.
- Every run keeps the epoch with the best validation accuracy. All architecture, augmentation and
  training decisions were made on validation accuracy; from Step 2c on, each configuration was run with
  3 seeds and compared by mean ± std, because single runs varied by up to ~2 points.
- The test set was used once, for the final model chosen in Step 5e.

## Experiment summary

| Step | What changed | Val acc |
|---|---|---:|
| 0 | Starter TNet (1 conv layer, gray 64px) | 48.1% |
| 1 | Deeper CNN (8 conv layers + BatchNorm) | 70.8% |
| 2a | + Cosine learning-rate decay | 77.3% |
| 2b | + Augmentation and 60 epochs | 82.8% |
| 3b | 128px input (best from scratch) | 84.7% |
| 4a | ImageNet ResNet-18, frozen | 91.7% |
| 4b | ResNet-18, fine-tuned | 94.4% |
| 5a | ConvNeXt-Tiny | 96.8% |
| 5b | ConvNeXt V2-Tiny (FCMAE) | 97.2% |
| 5c-ii | + Label smoothing + Mixup/CutMix | 97.5% |
| 5e | Ensemble of 3 seeds (final) | 97.7% → **test 95.75%** |

Experiments that did not help: Cutout, large rotation / vertical flips (−2.5 to −4.7), RGB input (a
Flower-only color shortcut), a 4× wider network, label smoothing alone, flip test-time augmentation.
Details, per-class results and failure analyses: [EXPERIMENTS.md](EXPERIMENTS.md).

## Pretrained weights used

| Model | Weights | Pretraining | Fine-tuned parameters |
|---|---|---|---|
| ResNet-18 (Step 4a) | torchvision `ResNet18_Weights.IMAGENET1K_V1` | ImageNet-1k, supervised | new final layer only (8,208) |
| ResNet-18 (Step 4b) | same | same | all layers (11.2M) |
| ConvNeXt-Tiny (Step 5a) | torchvision `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` | ImageNet-1k, supervised | all layers (27.8M) |
| ConvNeXt V2-Tiny (Steps 5b–5e, final) | timm `convnextv2_tiny.fcmae_ft_in1k` | FCMAE self-supervised + supervised fine-tuning, ImageNet-1k | all layers (27.9M) |

Weights are downloaded automatically by torchvision / timm on first use. No vision-language models,
DINO-style embeddings or Vision Transformers are used.

## Repository layout

```text
configs/                 one YAML file per experiment
src/data.py              dataset, fixed train/val split, transforms and augmentation
src/models.py            TNet, SceneCNN, pretrained ResNet-18, ConvNeXt / ConvNeXt V2
src/engine.py            training loop, evaluation, optimizer, LR schedule, Mixup/CutMix
train.py                 train one model from a config (optionally --seed)
evaluate.py              evaluate one checkpoint on val or test (--tta for flip averaging)
ensemble.py              evaluate an ensemble of checkpoints (averaged probabilities)
show_errors.py           grid of misclassified images for failure analysis
preview_augmentation.py  grid of augmented training images for a visual check
run_step5.py             runs all Step 5 experiments in order (used on Colab)
colab_step5.ipynb        Colab notebook for Step 5 (A100)
runs/                    per-run results: summary.json, history.json, eval_*.json, curves.png
EXPERIMENTS.md           full experiment log
AI_USAGE.md              how AI coding tools were used
```
