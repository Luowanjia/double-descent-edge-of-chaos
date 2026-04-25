# Double Descent and Edge of Chaos

AIS5281 final project code for studying the relationship between double descent
and edge-of-chaos metrics in neural networks.

The repository contains training code, model definitions, Lyapunov/FTLE logging,
and plotting scripts for post-processing experiment outputs.

## Repository layout

```text
.
|-- main.py                         # Main training and chaos-metric entry point
|-- models.py                       # MLP/CNN/CNN-dynamics model definitions
|-- utils.py                        # Dataset loading and augmentation helpers
|-- utils/
|   `-- loggingreporter.py          # Lyapunov, FTLE, and loss logging callback
`-- scripts/
    |-- plot_modelwise_double_descent.py          # Plot double-descent curves from metrics.csv
    |-- plot_double_descent_vs_ftle.py            # Combine validation error with FTLE outputs
    |-- plot_double_descent_vs_jacobian_norm.py   # Combine validation error with Lyapunov outputs
    `-- plot_validation_loss_by_width.py          # Plot validation-loss summaries
```

Generated experiment data is intentionally ignored by Git:

- `results_dd/`: CSV metrics and summary plots
- `rawdata/`: Lyapunov, FTLE, and loss pickle outputs

These folders are created automatically when the training code writes outputs.
Existing folders can be reused; new runs are placed in run-specific subfolders
based on architecture, optimizer, width, training budget, noise level, and repeat
index.

## Setup

Use Python 3.10 or 3.11. TensorFlow installation can be platform-specific, so
choose the TensorFlow package that matches your machine.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick smoke run

This runs a short CNN-dynamics experiment without chaos logging. It writes
`metrics.csv` under `results_dd/`.

```bash
python main.py \
  --architecture cnn_dynamics \
  --skip-chaos \
  --epochs 1 \
  --num-repeats 1 \
  --batch-size 128 \
  --base-width 2 \
  --train-subset 512
```

## Main experiment examples

Fixed-step CNN-dynamics training without chaos logging:

```bash
python main.py \
  --architecture cnn_dynamics \
  --skip-chaos \
  --optimizer SGD \
  --momentum 0.0 \
  --lr 0.1 \
  --lr-schedule inverse_sqrt \
  --lr-decay-steps 512 \
  --batch-size 128 \
  --base-width 8 \
  --label-noise 0.15 \
  --max-train-steps 50000 \
  --eval-every-steps 391 \
  --num-repeats 3
```

The same setup with FTLE/Lyapunov logging enabled:

```bash
python main.py \
  --architecture cnn_dynamics \
  --optimizer SGD \
  --momentum 0.0 \
  --lr 0.1 \
  --lr-schedule inverse_sqrt \
  --lr-decay-steps 512 \
  --batch-size 128 \
  --base-width 8 \
  --label-noise 0.15 \
  --max-train-steps 50000 \
  --eval-every-steps 391 \
  --num-repeats 3
```

Use `--results-root` and `--rawdata-root` if you want outputs somewhere other
than `results_dd/` and `rawdata/`.

## Plotting

Plot model-wise double-descent curves from existing `metrics.csv` files:

```bash
python scripts/plot_modelwise_double_descent.py \
  --architecture cnn_dynamics \
  --run-name-prefix sgd_mom0.0_lr0.1_lrschinverse_sqrt_decay512_bw \
  --required-substring _bs128_wd0_noise0.15_full_relu_steps50000_iter100_skipchaos \
  --double-arch-subdir
```

The remaining plotting scripts are also configurable from the command line, so
new users should not need to edit paths inside the source files.

Validation-loss summary by width:

```bash
python scripts/plot_validation_loss_by_width.py \
  --architecture cnn_dynamics \
  --run-prefix sgd_mom0.0_lr0.1_lrschinverse_sqrt_decay512_bw \
  --run-suffix _bs128_wd0_noise0.15_full_relu_steps50000_iter100_withchaos \
  --outdir plots
```

Double descent vs FTLE:

```bash
python scripts/plot_double_descent_vs_ftle.py \
  --architecture cnn_dynamics \
  --run-prefix sgd_mom0.0_lr0.1_lrschinverse_sqrt_decay512_bw \
  --run-suffix _bs128_wd0_noise0.15_full_relu_steps50000_iter100_withchaos \
  --outdir plots
```

Double descent vs Jacobian norm:

```bash
python scripts/plot_double_descent_vs_jacobian_norm.py \
  --architecture cnn_dynamics \
  --run-prefix sgd_mom0.0_lr0.1_lrschinverse_sqrt_decay512_bw \
  --run-suffix _bs128_wd0_noise0.15_full_relu_steps50000_iter100_withchaos \
  --repeat-id 0 \
  --max-width 24 \
  --outdir plots
```

For these scripts, widths and repeat ids are auto-discovered from the output
folders when possible. Pass `--widths` or `--repeat-ids` only when you want a
specific subset.

## Code reading guide

Start with these files in order:

1. `main.py`: parses experiment arguments, loads data, trains models, and writes
   metrics under `results_dd/`.
2. `models.py`: defines the MLP, standard CNN, and `cnn_dynamics` architectures.
3. `utils/loggingreporter.py`: computes and saves FTLE, Lyapunov/Jacobian-norm,
   and loss traces when `--skip-chaos` is not used.
4. `scripts/`: post-processing scripts that assume training outputs already
   exist under `results_dd/` and, for chaos plots, `rawdata/`.

## Notes for handoff

- Run commands from the repository root.
- Keras downloads CIFAR-10/Fashion-MNIST on first use if the datasets are not
  already cached locally.
- `--skip-chaos` is much faster and only writes training metrics.
- Chaos logging writes larger pickle files under `rawdata/`; keep these out of
  Git unless a downstream user explicitly needs a small sample.
