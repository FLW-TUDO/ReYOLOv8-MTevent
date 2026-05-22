"""
Optuna hyperparameter tuning for RVT on MTevent.

Each trial runs RVT training for a limited number of steps, triggers one validation pass
at the end, and reports val/AP (COCO mAP) extracted from the saved checkpoint filename.

Usage:
  cd /home/loki/event/ReYOLOv8
  python scripts/rvt/tune_optuna_rvt.py \
    --data_path /home/loki/event/ReYOLOv8/preprocessed_datasets/rvt_mtevent \
    --rvt_dir /home/loki/event/RVT \
    --n_trials 30 \
    --trial_steps 5000 \
    --device 0
"""

import argparse
import glob
import json
import os
import re
import subprocess
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

RVT_PYTHON = '/home/loki/venvs/rvt/bin/python'


def get_best_val_ap_from_ckpts(rvt_dir: Path, log_text: str) -> float:
    """
    Extract best val/AP from ModelCheckpoint filenames.
    Checkpoints land in {rvt_dir}/dummy/{run_id}/checkpoints/ because:
    - wandb disabled run returns project_name()='dummy', id=run_id
    - PL saves to {cwd}/{logger.name}/{logger.version}/checkpoints/
    Parse the run_id from the 'generating id' log line to find the right directory.
    """
    m = re.search(r'generating id ([a-z0-9]+)', log_text)
    if m:
        run_id = m.group(1)
        ckpt_dir = rvt_dir / 'dummy' / run_id / 'checkpoints'
        best_ap = 0.0
        for ckpt in ckpt_dir.glob('*.ckpt'):
            vm = re.search(r'val_AP=([0-9.]+)', ckpt.stem)
            if vm:
                best_ap = max(best_ap, float(vm.group(1)))
        return best_ap
    # Fallback: search all of dummy/ for recently modified val_AP checkpoints
    best_ap = 0.0
    for ckpt in (rvt_dir / 'dummy').glob('*/checkpoints/*.ckpt'):
        vm = re.search(r'val_AP=([0-9.]+)', ckpt.stem)
        if vm:
            best_ap = max(best_ap, float(vm.group(1)))
    return best_ap


def parse_val_ap_from_log(log_text: str) -> float:
    """Parse val/AP from PL ModelCheckpoint verbose stdout (exact precision)."""
    # PL prints: "val/AP' reached 0.00640 (best 0.00640)"
    patterns = [
        r"'val/AP'\s+reached\s+([0-9.]+)",
        r"val/AP['\s=:]+([0-9.]+)",
    ]
    best = 0.0
    for pat in patterns:
        for m in re.findall(pat, log_text):
            best = max(best, float(m))
    return best


def run_trial(trial: optuna.Trial, args) -> float:
    lr = trial.suggest_float('learning_rate', 5e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.05)
    grad_clip = trial.suggest_float('gradient_clip_val', 0.5, 2.0)
    batch_size = trial.suggest_categorical('batch_size_train', [4, 8])
    pct_start = trial.suggest_float('pct_start', 0.002, 0.05)

    trial_name = f'trial_{trial.number:03d}'
    output_dir = Path(args.output_dir) / trial_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        RVT_PYTHON, str(Path(args.rvt_dir) / 'train.py'),
        '+experiment/mtevent=small',
        'dataset=mtevent',
        f'dataset.path={args.data_path}',
        f'training.learning_rate={lr}',
        f'training.weight_decay={weight_decay}',
        f'training.gradient_clip_val={grad_clip}',
        f'training.max_steps={args.trial_steps}',
        f'training.lr_scheduler.total_steps={args.trial_steps}',
        f'training.lr_scheduler.pct_start={pct_start}',
        f'batch_size.train={batch_size}',
        'batch_size.eval=4',
        f'hardware.gpus={args.device}',
        'hardware.num_workers.train=4',
        'hardware.num_workers.eval=2',
        'wandb.group_name=optuna_rvt',
        f'hydra.run.dir={output_dir}',
        # Validate once at the end of trial (step-based, not epoch-based)
        f'validation.val_check_interval={args.trial_steps}',
        'validation.check_val_every_n_epoch=null',
        # Disable expensive high-dim visualisation
        'logging.train.high_dim.enable=False',
        'logging.validation.high_dim.enable=False',
    ]

    env = os.environ.copy()
    env['WANDB_MODE'] = 'disabled'

    result = subprocess.run(
        cmd,
        cwd=args.rvt_dir,
        capture_output=True,
        text=True,
        env=env,
    )

    log_text = result.stdout + result.stderr

    # Primary: exact value from PL ModelCheckpoint verbose stdout
    val_ap = parse_val_ap_from_log(log_text)
    # Fallback: parse checkpoint filename (lower precision, format=:.2f)
    if val_ap == 0.0:
        val_ap = get_best_val_ap_from_ckpts(Path(args.rvt_dir), log_text)

    # Persist trial artifacts
    (output_dir / 'stdout.txt').write_text(result.stdout)
    (output_dir / 'stderr.txt').write_text(result.stderr)
    (output_dir / 'params.json').write_text(json.dumps(trial.params, indent=2))
    (output_dir / 'result.json').write_text(json.dumps({'val_ap': val_ap}, indent=2))

    status = 'OK' if result.returncode == 0 else f'EXIT={result.returncode}'
    print(f'Trial {trial.number:03d} [{status}]: '
          f'lr={lr:.2e} wd={weight_decay:.4f} clip={grad_clip:.2f} '
          f'bs={batch_size} pct={pct_start:.3f} → val/AP={val_ap:.4f}')

    if result.returncode != 0:
        # Print last 1 KB of stderr for quick diagnosis
        print('  Last stderr:\n', result.stderr[-1000:])

    return val_ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='/home/loki/event/ReYOLOv8/preprocessed_datasets/rvt_mtevent')
    parser.add_argument('--rvt_dir', type=str, default='/home/loki/event/RVT')
    parser.add_argument('--output_dir', type=str,
                        default='/home/loki/event/ReYOLOv8/runs/rvt_optuna')
    parser.add_argument('--n_trials', type=int, default=30)
    parser.add_argument('--trial_steps', type=int, default=5000)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--db', type=str,
                        default='/home/loki/event/ReYOLOv8/optuna_rvt.db')
    args = parser.parse_args()

    study = optuna.create_study(
        study_name='rvt_mtevent',
        storage=f'sqlite:///{args.db}',
        direction='maximize',
        sampler=TPESampler(seed=42),
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: run_trial(trial, args),
        n_trials=args.n_trials,
        catch=(Exception,),
    )

    print('\nBest trial:')
    print(f'  val/AP: {study.best_value:.4f}')
    print(f'  Params: {study.best_params}')


if __name__ == '__main__':
    main()
