import os
import re
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_base_width(run_dir_name: str):
    match = re.search(r'_bw(\d+)_', run_dir_name)
    if match:
        return int(match.group(1))
    return None


def collect_run_metrics(
    metrics_root: str,
    architecture: str,
    run_name_prefix: str,
    required_substring: str = None,
    double_arch_subdir: bool = False,
    verbose: bool = True,
):
    """
    Search run directories and collect one summary row per run_dir.

    Supported directory layouts:

    1) Standard:
       {metrics_root}/{architecture}/{run_name_prefix}*

    2) Double-architecture subdir:
       {metrics_root}/{architecture}/{architecture}/{run_name_prefix}*

    Returns one row per matched run_dir, not yet merged by width.
    """
    if double_arch_subdir:
        base_dir = os.path.join(metrics_root, architecture, architecture)
    else:
        base_dir = os.path.join(metrics_root, architecture)

    pattern = os.path.join(base_dir, f"{run_name_prefix}*")
    run_dirs = sorted(glob.glob(pattern))

    if required_substring is not None:
        run_dirs = [
            d for d in run_dirs
            if required_substring in os.path.basename(d)
        ]

    if verbose:
        print(f"[info] base_dir = {base_dir}")
        print(f"[info] glob pattern = {pattern}")
        print(f"[info] required_substring = {required_substring}")
        print(f"[info] matched run_dirs = {len(run_dirs)}")
        for d in run_dirs:
            print(f"  - {os.path.basename(d)}")

    results = []

    for run_dir in run_dirs:
        run_dir_name = os.path.basename(run_dir)
        bw = extract_base_width(run_dir_name)
        if bw is None:
            print(f"[skip] cannot parse bw from: {run_dir_name}")
            continue

        metrics_files = sorted(glob.glob(os.path.join(run_dir, "repeat_*", "metrics.csv")))
        if len(metrics_files) == 0:
            print(f"[skip] no metrics.csv found in {run_dir}")
            continue

        final_val_errors = []
        best_val_errors = []
        final_val_losses = []
        best_val_losses = []
        gap_val_errors = []
        gap_val_losses = []

        for mf in metrics_files:
            df = pd.read_csv(mf)

            required_cols = ["val_acc", "val_loss"]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"{mf} missing required column: {col}")

            val_acc = df["val_acc"].astype(float).values
            val_loss = df["val_loss"].astype(float).values
            val_error = 1.0 - val_acc

            final_err = float(val_error[-1])
            best_err = float(np.min(val_error))
            final_loss = float(val_loss[-1])
            best_loss = float(np.min(val_loss))

            final_val_errors.append(final_err)
            best_val_errors.append(best_err)
            final_val_losses.append(final_loss)
            best_val_losses.append(best_loss)
            gap_val_errors.append(final_err - best_err)
            gap_val_losses.append(final_loss - best_loss)

        results.append({
            "base_width": bw,
            "run_dir_name": run_dir_name,
            "num_repeats": len(metrics_files),

            "final_val_error_mean": float(np.mean(final_val_errors)),
            "final_val_error_std": float(np.std(final_val_errors)),
            "best_val_error_mean": float(np.mean(best_val_errors)),
            "best_val_error_std": float(np.std(best_val_errors)),

            "final_val_loss_mean": float(np.mean(final_val_losses)),
            "final_val_loss_std": float(np.std(final_val_losses)),
            "best_val_loss_mean": float(np.mean(best_val_losses)),
            "best_val_loss_std": float(np.std(best_val_losses)),

            "gap_val_error_mean": float(np.mean(gap_val_errors)),
            "gap_val_error_std": float(np.std(gap_val_errors)),
            "gap_val_loss_mean": float(np.mean(gap_val_losses)),
            "gap_val_loss_std": float(np.std(gap_val_losses)),
        })

    if len(results) == 0:
        raise ValueError(
            f"No valid runs found.\n"
            f"base_dir={base_dir}\n"
            f"pattern={pattern}\n"
            f"required_substring={required_substring}\n"
            f"Please check your arguments."
        )

    df = pd.DataFrame(results).sort_values(["base_width", "run_dir_name"]).reset_index(drop=True)
    return df


def merge_duplicate_widths(df: pd.DataFrame):
    """
    Merge multiple run_dirs with the same base_width.
    This should usually NOT happen if filtering is correct, but when it does,
    we average them and also warn the user.
    """
    dup_counts = df.groupby("base_width").size()
    duplicated_widths = dup_counts[dup_counts > 1]

    if len(duplicated_widths) > 0:
        print("\n[warning] multiple run_dirs found for the same base_width.")
        print("This usually means multiple experiment configs were matched.")
        print("Consider tightening --required-substring.")
        print(duplicated_widths.to_string())

    agg_df = (
        df.groupby("base_width", as_index=False)
        .agg({
            "num_repeats": "sum",

            "final_val_error_mean": "mean",
            "final_val_error_std": "mean",
            "best_val_error_mean": "mean",
            "best_val_error_std": "mean",

            "final_val_loss_mean": "mean",
            "final_val_loss_std": "mean",
            "best_val_loss_mean": "mean",
            "best_val_loss_std": "mean",

            "gap_val_error_mean": "mean",
            "gap_val_error_std": "mean",
            "gap_val_loss_mean": "mean",
            "gap_val_loss_std": "mean",
        })
        .sort_values("base_width")
        .reset_index(drop=True)
    )

    return agg_df


def save_plot(x, y, yerr, xlabel, ylabel, title, outpath):
    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker='o')
    if yerr is not None:
        y = np.asarray(y)
        yerr = np.asarray(yerr)
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_all_formats(x, y, yerr, xlabel, ylabel, title, outdir, stem):
    save_plot(
        x, y, yerr,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        outpath=os.path.join(outdir, f"{stem}.png")
    )
    save_plot(
        x, y, yerr,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        outpath=os.path.join(outdir, f"{stem}.pdf")
    )


def main():
    parser = argparse.ArgumentParser(description="Plot model-wise double descent curves from metrics.csv")
    parser.add_argument("--metrics-root", type=str, default="results_dd",
                        help="root directory containing results")
    parser.add_argument("--architecture", type=str, required=True,
                        help="e.g. cnn")
    parser.add_argument("--run-name-prefix", type=str, required=True,
                        help=("prefix of run directories.\n"
                              "Example:\n"
                              "sgd_mom0.0_lr0.1_lrschinverse_sqrt_decay512_bw"))
    parser.add_argument("--required-substring", type=str, default=None,
                        help=("extra substring that must appear in run_dir_name.\n"
                              "Use this to isolate one exact experiment config.\n"
                              "Example:\n"
                              "_bs128_wd0_noise0.15_full_relu_steps50000_iter100_skipchaos"))
    parser.add_argument("--double-arch-subdir", action="store_true",
                        help=("set this if your directory structure is like:\n"
                              "results_dd/cnn/cnn/<run_dir>\n"
                              "instead of results_dd/cnn/<run_dir>"))
    parser.add_argument("--outdir", type=str, default=None,
                        help="output directory for plots and csv")
    parser.add_argument("--primary-metric", type=str, default="final_val_error",
                        choices=[
                            "final_val_error",
                            "best_val_error",
                            "final_val_loss",
                            "best_val_loss",
                            "gap_val_error",
                            "gap_val_loss",
                        ],
                        help="metric to highlight in terminal output; for paper alignment use final_val_error")
    parser.add_argument("--verbose", action="store_true",
                        help="print matched run directories")
    args = parser.parse_args()

    raw_df = collect_run_metrics(
        metrics_root=args.metrics_root,
        architecture=args.architecture,
        run_name_prefix=args.run_name_prefix,
        required_substring=args.required_substring,
        double_arch_subdir=args.double_arch_subdir,
        verbose=args.verbose,
    )

    print("\nCollected raw run-wise summary:")
    print(raw_df.to_string(index=False))

    df = merge_duplicate_widths(raw_df)

    print("\nMerged model-wise summary:")
    print(df.to_string(index=False))

    if args.outdir is None:
        outdir = os.path.join(
            args.metrics_root,
            args.architecture,
            "model_wise_plots"
        )
    else:
        outdir = args.outdir

    os.makedirs(outdir, exist_ok=True)

    raw_csv_path = os.path.join(outdir, "model_wise_raw_run_summary.csv")
    merged_csv_path = os.path.join(outdir, "model_wise_summary.csv")

    raw_df.to_csv(raw_csv_path, index=False)
    df.to_csv(merged_csv_path, index=False)

    x = df["base_width"].values

    final_err = df["final_val_error_mean"].values
    final_err_std = df["final_val_error_std"].values

    best_err = df["best_val_error_mean"].values
    best_err_std = df["best_val_error_std"].values

    final_loss = df["final_val_loss_mean"].values
    final_loss_std = df["final_val_loss_std"].values

    best_loss = df["best_val_loss_mean"].values
    best_loss_std = df["best_val_loss_std"].values

    gap_err = df["gap_val_error_mean"].values
    gap_err_std = df["gap_val_error_std"].values

    gap_loss = df["gap_val_loss_mean"].values
    gap_loss_std = df["gap_val_loss_std"].values

    save_all_formats(
        x, final_err, final_err_std,
        xlabel="Base width",
        ylabel="Final validation error",
        title="Model-wise curve: Final Validation Error vs Base Width",
        outdir=outdir,
        stem="model_wise_final_val_error"
    )
    save_all_formats(
        x, best_err, best_err_std,
        xlabel="Base width",
        ylabel="Best validation error",
        title="Model-wise curve: Best Validation Error vs Base Width",
        outdir=outdir,
        stem="model_wise_best_val_error"
    )
    save_all_formats(
        x, final_loss, final_loss_std,
        xlabel="Base width",
        ylabel="Final validation loss",
        title="Model-wise curve: Final Validation Loss vs Base Width",
        outdir=outdir,
        stem="model_wise_final_val_loss"
    )
    save_all_formats(
        x, best_loss, best_loss_std,
        xlabel="Base width",
        ylabel="Best validation loss",
        title="Model-wise curve: Best Validation Loss vs Base Width",
        outdir=outdir,
        stem="model_wise_best_val_loss"
    )
    save_all_formats(
        x, gap_err, gap_err_std,
        xlabel="Base width",
        ylabel="Final - Best validation error",
        title="Model-wise curve: Validation Error Gap vs Base Width",
        outdir=outdir,
        stem="model_wise_gap_val_error"
    )
    save_all_formats(
        x, gap_loss, gap_loss_std,
        xlabel="Base width",
        ylabel="Final - Best validation loss",
        title="Model-wise curve: Validation Loss Gap vs Base Width",
        outdir=outdir,
        stem="model_wise_gap_val_loss"
    )

    print("\n[primary metric view]")
    cols_to_show = ["base_width", f"{args.primary_metric}_mean"]
    std_col = f"{args.primary_metric}_std"
    if std_col in df.columns:
        cols_to_show.append(std_col)
    print(df[cols_to_show].to_string(index=False))

    print(f"\nSaved raw run summary CSV to: {raw_csv_path}")
    print(f"Saved merged summary CSV to: {merged_csv_path}")
    print(f"Saved plots to: {outdir}")


if __name__ == "__main__":
    main()
