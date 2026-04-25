# Double Descent and Edge-of-Chaos Dynamics

This repository contains the code for the AIS5281 final project:

**Double Descent and Edge-of-Chaos Dynamics in Deep Neural Networks**

The project studies whether model-wise double descent is associated with changes
in dynamical regimes, measured with finite-time Lyapunov exponents (FTLE) and
Jacobian-norm-based indicators.

## Repository Layout

```text
.
|-- main.py
|-- models.py
|-- utils.py
|-- utils/
|   `-- loggingreporter.py
|-- scripts/
|   |-- plot_modelwise_double_descent.py
|   |-- plot_validation_loss_by_width.py
|   |-- plot_double_descent_vs_ftle.py
|   `-- plot_double_descent_vs_jacobian_norm.py
|-- requirements.txt
`-- README.md
```

Main files:

- `main.py`: main training entry point, argument parsing, output directory naming, and experiment control.
- `models.py`: standard CNN and CNN-dynamics architecture definitions.
- `utils.py`: dataset loading and preprocessing utilities.
- `utils/loggingreporter.py`: loss logging plus FTLE and Lyapunov/Jacobian-norm computations.
- `scripts/`: plotting and post-processing scripts used for report figures.

## Data

The main experiments use CIFAR-10 from `tensorflow.keras.datasets`.

CIFAR-10 is downloaded automatically by Keras on first use. No manual dataset
preparation is required.

## Environment Setup

Python 3.10 or 3.11 is recommended.

```bash
pip install -r requirements.txt
```

## Output Directories

Generated experiment data is intentionally ignored by Git:

- `results_dd/`: training metrics and plotting summaries
- `rawdata/`: FTLE, Lyapunov/Jacobian-norm, and loss pickle files
- `plots/`: recommended location for generated figures

These folders are created automatically when the code writes outputs. Existing
folders can be reused; new runs are placed in run-specific subfolders based on
architecture, optimizer, width, training budget, noise level, and repeat index.

Use `--results-root` and `--rawdata-root` if you want training outputs somewhere
other than `results_dd/` and `rawdata/`.

## Main CNN-Dynamics Sweep with FTLE Logging

This is the main dense width sweep used for the CNN-dynamics experiments. Run it
from the repository root.

```bash
nohup bash -c '
for bw in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18 20 24 32; do
  echo "=============================="
  echo "Starting bw=${bw} at $(date)"
  python -u main.py \
    --architecture cnn_dynamics \
    --optimizer sgd \
    --lr 0.1 \
    --lr-schedule inverse_sqrt \
    --lr-decay-steps 512 \
    --momentum 0.0 \
    --batch-size 128 \
    --base-width ${bw} \
    --max-train-steps 50000 \
    --eval-every-steps 391 \
    --epochs 130 \
    --label-noise 0.15 \
    --num-iterations 100 \
    --num-repeats 3
  echo "Finished bw=${bw} at $(date)"
done
' > sweep_50k_cnndyn_withftle_dense_r3.log 2>&1 &
```

This writes training metrics under `results_dd/` and dynamical quantities under
`rawdata/`. The most relevant raw outputs are:

- `rawdata/ftle_benettin/`
- `rawdata/lyapunov1s/`
- `rawdata/losses/`

## Faster Training without Chaos Logging

For a faster sweep that only records training and validation metrics, add
`--skip-chaos`.

```bash
nohup bash -c '
for bw in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18 20 24 32; do
  echo "=============================="
  echo "Starting bw=${bw} at $(date)"
  python -u main.py \
    --architecture cnn_dynamics \
    --optimizer sgd \
    --lr 0.1 \
    --lr-schedule inverse_sqrt \
    --lr-decay-steps 512 \
    --momentum 0.0 \
    --batch-size 128 \
    --base-width ${bw} \
    --max-train-steps 50000 \
    --eval-every-steps 391 \
    --epochs 130 \
    --label-noise 0.15 \
    --num-iterations 100 \
    --num-repeats 3 \
    --skip-chaos
  echo "Finished bw=${bw} at $(date)"
done
' > sweep_50k_cnndyn_skipchaos_dense_r3.log 2>&1 &
```

## Quick Smoke Test

Use this command only to check that the code runs.

```bash
python main.py \
  --architecture cnn_dynamics \
  --optimizer sgd \
  --lr 0.1 \
  --lr-schedule inverse_sqrt \
  --lr-decay-steps 512 \
  --momentum 0.0 \
  --batch-size 128 \
  --base-width 2 \
  --epochs 1 \
  --label-noise 0.15 \
  --num-iterations 100 \
  --num-repeats 1 \
  --train-subset 512 \
  --skip-chaos
```

## Plotting

Run plotting scripts from the repository root.

The main sweep above omits `--weight-decay`, so the generated run suffix uses
`wd0`. If you explicitly run training with `--weight-decay 0.0`, the suffix uses
`wd0.0`; use the suffix that matches your output directories.

The analysis distinguishes between final-epoch and best-epoch evaluation.
Model-wise double descent is primarily defined using final validation error,
following standard practice. However, models of different widths may reach their
best generalization performance at different training stages. Therefore, the
repository reports both final-vs-final and best-vs-best comparisons.

FTLE uses 0 as the edge-of-chaos threshold: positive values indicate chaotic
behavior, while negative values indicate ordered behavior. The
Jacobian-norm-based indicator uses 1 as the edge-of-chaos threshold.

### Model-wise Double Descent

```bash
python scripts/plot_modelwise_double_descent.py \
  --architecture cnn_dynamics \
  --run-name-prefix sgd_mom0.0_lr0.1_lrschinverse_sqrt_decay512_bw \
  --required-substring _bs128_wd0_noise0.15_full_relu_steps50000_iter100_withchaos \
  --double-arch-subdir \
  --outdir plots/modelwise_dd
```

For a skip-chaos sweep, use:

```bash
--required-substring _bs128_wd0_noise0.15_full_relu_steps50000_iter100_skipchaos
```

### Validation-Loss Summary by Width

```bash
python scripts/plot_validation_loss_by_width.py \
  --architecture cnn_dynamics \
  --run-prefix sgd_mom0.0_lr0.1_lrschinverse_sqrt_decay512_bw \
  --run-suffix _bs128_wd0_noise0.15_full_relu_steps50000_iter100_withchaos \
  --outdir plots/validation_loss_by_width
```

### Double Descent vs FTLE

```bash
python scripts/plot_double_descent_vs_ftle.py \
  --architecture cnn_dynamics \
  --run-prefix sgd_mom0.0_lr0.1_lrschinverse_sqrt_decay512_bw \
  --run-suffix _bs128_wd0_noise0.15_full_relu_steps50000_iter100_withchaos \
  --outdir plots/dd_vs_ftle
```

This produces two main comparisons:

- final validation error vs final FTLE
- best validation error vs best-epoch FTLE

The first comparison follows the standard final-epoch definition of model-wise
double descent. The second comparison evaluates both generalization and dynamics
at the best-validation epoch, providing a stage-aligned comparison across model
widths.

An additional diagnostic plot, final validation error vs best-epoch FTLE, may
also be generated for exploratory analysis, but it is not used as the main
comparison in the report.

For fair visual comparison, all FTLE curve plots use a shared FTLE y-axis range.

### Double Descent vs Jacobian Norm

```bash
python scripts/plot_double_descent_vs_jacobian_norm.py \
  --architecture cnn_dynamics \
  --run-prefix sgd_mom0.0_lr0.1_lrschinverse_sqrt_decay512_bw \
  --run-suffix _bs128_wd0_noise0.15_full_relu_steps50000_iter100_withchaos \
  --repeat-id 0 \
  --max-width 24 \
  --outdir plots/dd_vs_jacobian_norm
```

This produces:

- final validation error vs final Jacobian norm
- final validation error vs best-epoch Jacobian norm
- best validation error vs best-epoch Jacobian norm

The Jacobian-norm plots use a shared right-axis range for fair visual
comparison.

Widths and repeat ids are auto-discovered from the output folders when possible.
Pass `--widths` or `--repeat-ids` only when you want a specific subset.

## Notes on Directory Names

Run directories are generated automatically from the training configuration. A
typical run name from the main sweep is:

```text
sgd_mom0.0_lr0.1_lrschinverse_sqrt_decay512_bw{WIDTH}_bs128_wd0_noise0.15_full_relu_steps50000_iter100_withchaos
```

The plotting scripts rely on this naming pattern. If the training command uses
`--weight-decay 0.0`, the directory contains `wd0.0`. If weight decay is omitted,
as in the main sweep above, the directory contains `wd0`.

## Code Reading Guide

A recommended order for reading the code is:

1. `main.py`: training loop, argument parsing, output directory naming, and experiment control.
2. `models.py`: standard CNN and CNN-dynamics architecture definitions.
3. `utils/loggingreporter.py`: FTLE, Lyapunov/Jacobian-norm, and loss logging.
4. `scripts/`: post-processing and figure generation.

## Project Summary

The main experimental pipeline is:

1. Train width-scaled CNN-dynamics models on CIFAR-10 with 15% label noise.
2. Measure model-wise double descent using validation error.
3. Measure dynamical behavior using FTLE.
4. Compare generalization curves with dynamical indicators across model width.
5. Use Jacobian norm as an additional appendix analysis.

The project finds that the apparent relationship between double descent and
dynamical behavior depends on the evaluation stage, and does not support a fixed
one-to-one correspondence between the double descent peak and the edge-of-chaos
transition.
