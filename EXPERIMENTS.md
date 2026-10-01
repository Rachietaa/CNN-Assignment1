# Experiment Log

All numbers are on the fixed 20% validation split (seed 0, 480 images).
The test set is only evaluated once, on the final selected model.

## Dataset notes

- 16 classes, 150 train / 25 test images per class (balanced). `test2/` holds 400 unlabeled images.
- **Only the `Flower` class is in color.** 2,250 of 2,400 train images are grayscale (`L`); the 150 RGB
  images are all `Flower` (same in test: 25 RGB, all `Flower`). Color input therefore gives a shortcut
  for one class rather than general color information.
- Image sizes vary (292 distinct sizes); half are 256×256, most others ~220px tall.
- No duplicate files between train, test and test2 (MD5 check).

## Results summary

| # | Experiment | Config | What changed | Why | Val acc | Observation |
|---|---|---|---|---|---:|---|
| 0 | Baseline | `configs/baseline.yaml` | Starter TNet, grayscale 64×64, Adam 2e-3, 20 epochs | Sanity check / reference point | 48.1% | Heavy overfitting: train acc 99% vs val 47%; val loss rises after epoch 7. Weakest classes: InsideCity, Kitchen (20%), Industrial (31%). |

---

## Step 0 — Baseline (starter TNet)

**Goal.** Reproduce the course starter model exactly, inside our own config-driven pipeline, to get a
reference number that every later experiment is compared against. Nothing was tuned.

**Reproduce.**
```bash
python train.py --config configs/baseline.yaml
python evaluate.py --checkpoint runs/baseline/best.pt     # validation split
```
Outputs: `runs/baseline/{summary.json, history.json, curves.png, eval_val.json}`.

### Setup

| Item | Value |
|---|---|
| Data split | 2,400 train images → 1,920 train / 480 val. Random (not stratified) split from `torch.randperm` with seed 0, the same split as the starter notebook's `random_split`. |
| Preprocessing | Convert to grayscale (1 channel) → resize to 64×64 (aspect ratio not kept) → tensor → normalize with mean 0.5, std 0.5 (range [-1, 1]) |
| Augmentation | None |
| Model | `TNet` ([src/models.py](src/models.py)) |
| Loss | Cross-entropy |
| Optimizer | Adam, lr 0.002, no weight decay, no LR schedule |
| Batch size / epochs | 64 / 20 (30 iterations per epoch) |
| Model selection | After every epoch, evaluate on val; keep the weights from the epoch with the highest val accuracy |
| Seed / hardware | Seed 0; Apple Silicon GPU (MPS), PyTorch 2.14.1, torchvision 0.29.1 |
| Training time | 9.9 s total |

### Architecture and parameter count

| Layer | Output shape | Parameters |
|---|---|---:|
| Input (grayscale) | 1 × 64 × 64 | – |
| Conv2d 3×3, 16 filters, no padding | 16 × 62 × 62 | 1·16·3·3 + 16 = **160** |
| ReLU | 16 × 62 × 62 | 0 |
| MaxPool 4×4, stride 4 | 16 × 15 × 15 | 0 |
| Flatten | 3,600 | 0 |
| Linear 3,600 → 16 | 16 | 3,600·16 + 16 = **57,616** |
| **Total** | | **57,776** |

The model has only one convolutional layer, so it sees 3×3 patterns and nothing larger. Almost all of
its parameters (99.7%) are in the final linear layer, which effectively memorizes where those small
patterns appear in each image.

### How the baseline number is calculated

Validation accuracy = (validation images whose highest-scoring class equals the true label) ÷ 480.
For each image the model outputs 16 scores, and the predicted class is the one with the highest score.

- The best epoch was **epoch 10**, with **231 / 480 correct = 0.48125 → 48.1%**.
- Chance level for 16 balanced classes is 1/16 = 6.25%, so the pipeline is clearly learning.
- The handout says the starter gets "less than approximately 50%", which this matches.

Note: the epoch is chosen on the same validation set we report, so 48.1% is slightly optimistic. The
same rule is used for every experiment, so comparisons between them remain fair.

### Training curve

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---:|---:|---:|---:|---:|
| 1 | 2.559 | 19.7% | 2.229 | 32.5% |
| 3 | 1.664 | 50.0% | 1.873 | 41.3% |
| 5 | 1.153 | 66.5% | 1.804 | 46.5% |
| 7 | 0.815 | 78.4% | **1.742** (lowest) | 46.9% |
| **10** | 0.516 | 88.4% | 1.827 | **48.1%** (best) |
| 15 | 0.241 | 96.2% | 2.137 | 46.7% |
| 20 | 0.118 | 99.2% | 2.329 | 47.3% |

![Baseline curves](runs/baseline/curves.png)

### Per-class validation accuracy (best checkpoint)

The val split is random rather than stratified, so classes have 24–39 val images each.

| Class | Correct / total | Acc | Most confused with |
|---|---:|---:|---|
| InsideCity | 6 / 30 | 20.0% | Store (8) |
| Kitchen | 6 / 30 | 20.0% | InsideCity (4) |
| Industrial | 9 / 29 | 31.0% | Bedroom (6) |
| Forest | 11 / 29 | 37.9% | TallBuilding (4) |
| LivingRoom | 10 / 26 | 38.5% | Kitchen (5) |
| Bedroom | 16 / 39 | 41.0% | LivingRoom (7) |
| Store | 14 / 31 | 45.2% | InsideCity (6) |
| Flower | 15 / 31 | 48.4% | Store (5) |
| Mountain | 14 / 28 | 50.0% | OpenCountry (4) |
| Office | 12 / 24 | 50.0% | LivingRoom (3) |
| OpenCountry | 14 / 28 | 50.0% | Coast (7) |
| Highway | 16 / 28 | 57.1% | Coast (5) |
| Suburb | 21 / 34 | 61.8% | Mountain (4) |
| Coast | 22 / 33 | 66.7% | OpenCountry (7) |
| TallBuilding | 19 / 28 | 67.9% | Industrial (3) |
| Street | 26 / 32 | 81.2% | Bedroom (1) |

### Observations

1. **Severe overfitting.** Training accuracy climbs to 99% while validation stays around 47%. Validation
   loss is lowest at epoch 7 and then rises steadily, so after that point the model is memorizing the
   1,920 training images rather than learning general features.
2. **The model is too shallow, so it can't be called underfit or well fit.** One 3×3 conv layer can only
   detect edges and small textures. The confusions match this: classes that differ in layout but share
   textures get mixed up (indoor scenes Bedroom/LivingRoom/Kitchen; outdoor Coast/OpenCountry/Mountain;
   InsideCity/Store).
3. **Classes with distinctive structure do best.** Street (81%) and TallBuilding (68%) have strong,
   consistent lines; the cluttered indoor classes do worst.
4. **Flower gets only 48%** even though it is the only color class, because the baseline converts every
   image to grayscale and throws that information away (see Dataset notes).
5. **Low resolution.** 64×64 removes most fine detail, which matters for indoor scenes defined by
   objects (stove, bed, shelves).

### What this suggests for the next step

- Add more conv layers with padding and BatchNorm, so the model learns larger patterns (objects, layout)
  instead of memorizing positions in a large linear layer → **Step 1**.
- Fight overfitting with augmentation and regularization → **Step 2**.
- Revisit resolution (64 → 128) and the gray-vs-color question in later steps.
