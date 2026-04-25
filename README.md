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
    |-- plot_model_wise_dd.py       # Plot double-descent curves from metrics.csv
    |-- plot_dd_vs_ftle.py          # Combine validation error with FTLE outputs
    |-- plot_dd_vs_chaos.py         # Combine validation error with Lyapunov outputs
    `-- plot_val_loss_latest.py     # Plot validation-loss summaries
```

Generated experiment data is intentionally ignored by Git:

- `results_dd/`: CSV metrics and summary plots
- `rawdata/`: Lyapunov, FTLE, and loss pickle outputs

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
python scripts/plot_model_wise_dd.py \
  --architecture cnn_dynamics \
  --run-name-prefix sgd_mom0.0_lr0.1_lrschinverse_sqrt_decay512_bw \
  --required-substring _bs128_wd0_noise0.15_full_relu_steps50000_iter100_skipchaos
```

The other plotting scripts contain configuration constants near the top of each
file. Update `RESULTS_ROOT`, `RAW_ROOT`, `WIDTHS`, and `REPEAT_IDS` to match the
experiment outputs before running them.

## Notes for handoff

- Run commands from the repository root.
- Keras downloads CIFAR-10/Fashion-MNIST on first use if the datasets are not
  already cached locally.
- `--skip-chaos` is much faster and only writes training metrics.
- Chaos logging writes larger pickle files under `rawdata/`; keep these out of
  Git unless a downstream user explicitly needs a small sample.
