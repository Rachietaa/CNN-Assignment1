"""Run all Step 5 experiments in order on one machine (written for a CUDA GPU such as Colab's A100).

    python run_step5.py

1. Control: Step 4b (ResNet-18 fine-tune) re-run on this hardware, to compare with Step 4 fairly.
2. Step 5a: ConvNeXt-Tiny (supervised ImageNet weights).
3. Step 5b: ConvNeXt V2-Tiny (FCMAE + supervised ImageNet weights).
4. Picks the better of 5a/5b by mean best validation accuracy over 3 seeds, then runs on top of it:
   Step 5c-i:  + label smoothing 0.1
   Step 5c-ii: + label smoothing 0.1 + Mixup/CutMix
Every run is evaluated on the validation split with and without horizontal-flip TTA (Step 5d).
The test split is never touched. Finished runs are skipped, so the script can be re-started.
"""
import json
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

SEEDS = [0, 1, 2]


def run_name(name, seed):
    return name if seed == 0 else f'{name}_s{seed}'


def train_and_eval(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    for seed in SEEDS:
        run = Path('runs') / run_name(cfg['name'], seed)
        if not (run / 'eval_val_tta.json').exists():
            cmd = [sys.executable, 'train.py', '--config', str(config_path)]
            if seed != 0:
                cmd += ['--seed', str(seed)]
            subprocess.run(cmd, check=True)
            for extra in ([], ['--tta']):
                subprocess.run([sys.executable, 'evaluate.py', '--checkpoint', str(run / 'best.pt'), *extra],
                               check=True)
        summary = json.loads((run / 'summary.json').read_text())
        print(f"== {run.name}: best val {summary['best_val_acc']:.4f} in {summary['train_seconds']} s", flush=True)
    return statistics.mean(json.loads((Path('runs') / run_name(cfg['name'], s) / 'summary.json').read_text())
                           ['best_val_acc'] for s in SEEDS)


def derived_config(base_path, name, comment, **train_updates):
    """Write a new config that equals `base_path` except for the given train settings."""
    cfg = yaml.safe_load(Path(base_path).read_text())
    cfg['name'] = name
    cfg['train'].update(train_updates)
    out = Path('configs') / f'{name}.yaml'
    out.write_text(f'# {comment}\n' + yaml.safe_dump(cfg, sort_keys=False))
    return out


def main():
    ctrl = derived_config('configs/step4b_resnet18_finetune.yaml', 'step5_ctrl_resnet18_cuda',
                          'Step 5 control: Step 4b re-run unchanged on the Step 5 hardware (CUDA GPU).')
    train_and_eval(ctrl)

    means = {path: train_and_eval(path)
             for path in ['configs/step5a_convnext_tiny.yaml', 'configs/step5b_convnextv2_tiny.yaml']}
    best_base = max(means, key=means.get)
    print(f'== Mean best val: {means}. Step 5c builds on {best_base}', flush=True)

    ls = derived_config(best_base, 'step5c1_label_smoothing',
                        f'Step 5c-i: {best_base} + label smoothing 0.1.', label_smoothing=0.1)
    train_and_eval(ls)
    mix = derived_config(best_base, 'step5c2_ls_mix',
                         f'Step 5c-ii: {best_base} + label smoothing 0.1 + Mixup/CutMix (p 0.5).',
                         label_smoothing=0.1, mix={'p': 0.5, 'mixup_alpha': 0.2, 'cutmix_alpha': 1.0})
    train_and_eval(mix)
    print('== Step 5 finished.', flush=True)


if __name__ == '__main__':
    main()
