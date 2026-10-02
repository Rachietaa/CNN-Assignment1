# AI Usage

## 1. AI coding tool used

**Claude Code** (Anthropic), used inside VS Code as a pair programmer for the whole project.

I used it the way the assignment describes: the AI wrote boilerplate, implemented features I asked for,
debugged errors, built plotting/analysis utilities and **suggested** experiments. I decided which
experiments to run and why, checked whether the generated code was correct, interpreted the results, and
made the final decisions. I also ran the Colab GPU jobs and made every Git commit myself, step by step.

---

## 2. Representative examples of how AI assisted me

**1. Boilerplate and refactoring.** I asked the AI to turn the starter notebook into a reproducible
project. It built a config-driven pipeline (one YAML file per experiment, `train.py`, `evaluate.py`, saved
curves and per-class results), which let me change exactly one thing per experiment and compare runs on
the same fixed validation split.

**2. Implementing augmentation and a visual check.** I asked whether we would use rotation, Cutout and
Mixup. The AI implemented these together with cropping and flips, and a utility
(`preview_augmentation.py`) that shows original images next to augmented ones. I used it before every
augmentation experiment, and it caught a real bug (Section 3).

**3. Suggesting experiments, which I then decided on.** When augmentation first seemed not to help, the AI
suggested a control run (longer training without augmentation). I ran it because otherwise "more
augmentation" and "more epochs" were mixed up in one comparison; it showed the two only help together.
When single runs started disagreeing by up to ~2 points, I adopted the AI's suggestion of running every
setting with 3 random seeds and comparing mean ± std.

**4. Debugging.** The AI diagnosed several failures: a check in `ensemble.py` that wrongly rejected
compatible checkpoints, a macOS multiprocessing error in an analysis script, and a Colab notebook that kept
disconnecting because its output was too large (fixed by saving runs directly to Google Drive and printing
only one line per run).

**5. Analysis utilities.** The AI wrote `show_errors.py`, which displays every misclassified image. I used
it to analyse the test errors: 10 of 17 were wooded hillsides and meadows labelled Mountain/Forest but
predicted as OpenCountry.

---

## 3. An incorrect, ineffective or questionable AI suggestion

**Incorrect: validation images would have been augmented.** When implementing augmentation, the AI wrote:

```python
aug = cfg.get('augment') or {} if train else {}
```

It reads as "augment only during training", but Python evaluates it as
`cfg.get('augment') or ({} if train else {})`, so whenever the config contains augmentation settings,
**validation images get augmented too**. Validation accuracy would have been measured on randomly cropped,
flipped and rotated images. Nothing would crash, but every comparison after that point would be wrong.

**Incorrect: augmented images with artifacts.** The first augmentation pipeline rotated images *after*
shrinking them to 64×64, using nearest-neighbour interpolation. This produced jagged "staircase" edges and
black corners that never appear in real photos, so the model could have learned from artifacts.

**Ineffective: flip test-time augmentation.** The AI proposed averaging each prediction with the prediction
on the mirrored image. On every ConvNeXt model it gave no gain (−0.1 to −0.4 points), because training with
random flips had already made the models flip-invariant. I did not use it in the final system.

**Questionable: training the final model without validation data** (see Section 5).

Smaller errors: wrong numbers in drafted text (e.g. "Forest errors 0 → 6" when it was 2 → 6), shell
commands with `#` comments that my zsh does not ignore (which produced two mislabelled commits), and a
training-time estimate for ConvNeXt that was far too low (~70 s per epoch on my laptop).

---

## 4. How I verified or modified the AI-generated solutions

| Issue | How it was verified | Change made |
|---|---|---|
| Augmented validation data | Printed the training and evaluation transforms side by side: evaluation must contain only resize, tensor conversion and normalization, with no random operations. Repeated whenever augmentation code changed. | Fixed the precedence: `(cfg.get('augment') or {}) if train else {}` |
| Rotation artifacts | Looked at the augmentation preview grid **before** training | Rotate at full resolution with bilinear interpolation and grey corner fill |
| Flip TTA | Evaluated every Step 5 model with and without it | Not used in the final system |
| Wrong numbers in text | Recomputed every claim from the saved JSON result files before committing | Corrected the text |
| Mislabelled commits | Compared the GitHub history with the work actually done | Rewrote the two commits with correct messages |

**Checks I relied on throughout:**
- **One change per experiment**, so any difference can be traced to that change.
- **3 seeds per setting** once differences became small, so a lucky run could not decide an experiment.
- **Control experiments**: the 60-epoch run without augmentation (Step 2b), and re-running the previous
  best model on the Colab GPU before comparing new models there (94.4% on my laptop vs 94.1% on Colab, i.e.
  the hardware does not change results).
- **Reproducing results**: Colab-trained checkpoints were re-evaluated on my laptop and gave exactly the
  same validation accuracy; the published checkpoints were downloaded again and checksum-compared.
- **Test set used once**, on a final model fixed beforehand.

---

## 5. A decision I made rather than accepting the AI's recommendation

### Keeping a validation set for the final model

Before the final test, I asked how to get beyond 97.7% validation accuracy. The AI's top recommendation
was to retrain the final recipe on **all 2,400 labelled images** (merging the 480 validation images into
training) and keep the last epoch, arguing that 25% more data usually helps.

I rejected it. Without validation data, nothing could detect overfitting or select the best epoch, so the
final model would go into the one-time test with no measured evidence that it works, and the validation
accuracy in my report would describe a different model from the one tested. I chose a slightly smaller
training set and a final model whose performance was actually measured.

### Running the full Step 5 on a Colab A100

The AI advised a reduced Step 5 (one stronger model only), and later advised staying on my laptop rather
than moving to Colab. I chose to run the **complete** comparison (ConvNeXt vs ConvNeXt V2, label smoothing,
Mixup/CutMix, 3 seeds each) on a Colab A100. This gave the best result of the project (97.5% mean
validation for the final recipe) and finished in about an hour instead of an estimated 3–5 hours locally.

---

## Justifying the final system (not "the AI suggested it")

Every component of the final model is backed by an experiment in [EXPERIMENTS.md](EXPERIMENTS.md):

| Component | Evidence |
|---|---|
| ImageNet-pretrained CNN, all layers fine-tuned | From-scratch models plateaued at 83–85% and a 4× wider network did not help (data, not capacity, was the limit); a frozen pretrained ResNet-18 reached 91.7%, full fine-tuning 94.4% |
| ConvNeXt V2-Tiny backbone | ConvNeXt beat ResNet-18 by 2.7 points with almost no seed variation (96.8% ± 0.1 vs 94.1% ± 0.1) |
| Grey input copied to 3 channels | Colour is a shortcut in this data: an RGB model's Flower accuracy fell from 89% to 22% without colour. The grey model reached 100% on Flower |
| Mild augmentation (±10° rotation, crop, horizontal flip, brightness/contrast) | +5.4 points with longer training; large rotations and vertical flips *lost* 2.5–4.7 points because scenes have a fixed "up" |
| Cosine learning-rate schedule | Removed ±15-point epoch-to-epoch swings in validation accuracy |
| Label smoothing + Mixup/CutMix | Highest mean validation accuracy (97.5%) and fewer Coast/OpenCountry confusions; within seed noise, so a weak preference |
| Ensemble of 3 seeds | The best single seed (98.3% on validation) was likely lucky; on test it scored the same as the ensemble (95.75%) |
