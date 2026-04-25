import os
import re
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# Default config. Override from the command line; do not edit these for routine use.
# =========================================================
RESULTS_ROOT = "results_dd"
RAW_ROOT = "rawdata/ftle_benettin"

ARCH = "cnn_dynamics"
RUN_PREFIX = None
RUN_SUFFIX = None

# Dense FTLE sweep widths
WIDTHS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 24, 32]

# repeats to aggregate
REPEAT_IDS = [0, 1, 2]

ALLOW_FALLBACK_TO_PREV_LOGGED_EPOCH = True
OUT_PREFIX = "dd_vs_ftle_benettin_repeats"
OUTDIR = "."


# =========================================================
# Helpers
# =========================================================
def parse_int_list(value):
    if isinstance(value, list):
        return value
    items = [item.strip() for item in value.split(",") if item.strip()]
    if len(items) == 0:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return [int(item) for item in items]


def make_output_path(filename):
    return os.path.join(OUTDIR, filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot double-descent metrics against FTLE outputs."
    )
    parser.add_argument("--results-root", default=RESULTS_ROOT,
                        help="root directory containing metrics.csv outputs")
    parser.add_argument("--raw-root", default=RAW_ROOT,
                        help="root directory containing FTLE pickle outputs")
    parser.add_argument("--architecture", default=ARCH,
                        help="architecture directory name, e.g. cnn_dynamics")
    parser.add_argument("--run-prefix", required=True,
                        help="run directory prefix before the base width")
    parser.add_argument("--run-suffix", required=True,
                        help="run directory suffix after the base width")
    parser.add_argument("--widths", type=parse_int_list, default=WIDTHS,
                        help="comma-separated base widths, e.g. 2,4,8,16")
    parser.add_argument("--repeat-ids", type=parse_int_list, default=REPEAT_IDS,
                        help="comma-separated repeat ids, e.g. 0,1,2")
    parser.add_argument("--out-prefix", default=OUT_PREFIX,
                        help="prefix for generated CSV/PNG/PDF files")
    parser.add_argument("--outdir", default=OUTDIR,
                        help="directory for generated CSV/PNG/PDF files")
    parser.add_argument("--no-fallback-to-prev-epoch", action="store_true",
                        help="require exact best-epoch FTLE files")
    return parser.parse_args()


def configure_from_args(args):
    global RESULTS_ROOT, RAW_ROOT, ARCH, RUN_PREFIX, RUN_SUFFIX
    global WIDTHS, REPEAT_IDS, ALLOW_FALLBACK_TO_PREV_LOGGED_EPOCH
    global OUT_PREFIX, OUTDIR

    RESULTS_ROOT = args.results_root
    RAW_ROOT = args.raw_root
    ARCH = args.architecture
    RUN_PREFIX = args.run_prefix
    RUN_SUFFIX = args.run_suffix
    WIDTHS = args.widths
    REPEAT_IDS = args.repeat_ids
    ALLOW_FALLBACK_TO_PREV_LOGGED_EPOCH = not args.no_fallback_to_prev_epoch
    OUT_PREFIX = args.out_prefix
    OUTDIR = args.outdir
    os.makedirs(OUTDIR, exist_ok=True)


def get_run_name(bw):
    return f"{RUN_PREFIX}{bw}{RUN_SUFFIX}"


def get_ftle_dir(bw, repeat_id):
    return os.path.join(
        RAW_ROOT,
        ARCH,
        get_run_name(bw),
        f"repeat_{repeat_id}",
    )


def get_metrics_csv_path(bw, repeat_id):
    return os.path.join(
        RESULTS_ROOT,
        ARCH,
        ARCH,
        get_run_name(bw),
        f"repeat_{repeat_id}",
        "metrics.csv"
    )


def parse_epoch_from_filename(path):
    fname = os.path.basename(path)
    m = re.match(r"epoch(\d+)\.pkl$", fname)
    if m is None:
        return None
    return int(m.group(1))


def list_epoch_files(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "epoch*.pkl")))
    pairs = []
    for f in files:
        ep = parse_epoch_from_filename(f)
        if ep is not None:
            pairs.append((ep, f))
    pairs.sort(key=lambda x: x[0])
    return pairs


def load_pkl_mean(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    arr = np.asarray(data)
    return float(np.mean(arr))


def choose_final_epoch_file(epoch_files):
    if len(epoch_files) == 0:
        return None, None
    return epoch_files[-1]


def choose_best_epoch_file(epoch_files, target_epoch, allow_fallback=True):
    if len(epoch_files) == 0:
        return None, None

    for ep, fp in epoch_files:
        if ep == target_epoch:
            return ep, fp

    if not allow_fallback:
        return None, None

    candidates = [(ep, fp) for ep, fp in epoch_files if ep <= target_epoch]
    if len(candidates) == 0:
        return None, None
    return candidates[-1]


def load_metrics_csv(metrics_csv_path):
    if not os.path.exists(metrics_csv_path):
        return None

    df = pd.read_csv(metrics_csv_path)

    required_cols = ["epoch", "val_acc", "val_loss"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"{metrics_csv_path} missing required column: {c}")

    df = df.dropna(subset=["epoch", "val_acc", "val_loss"]).copy()
    if len(df) == 0:
        return None

    df["epoch"] = df["epoch"].astype(int)
    df["val_acc"] = df["val_acc"].astype(float)
    df["val_loss"] = df["val_loss"].astype(float)
    df["val_error"] = 1.0 - df["val_acc"]

    return df.sort_values("epoch").reset_index(drop=True)


def get_final_metrics_from_metrics_csv(metrics_csv_path):
    df = load_metrics_csv(metrics_csv_path)
    if df is None or len(df) == 0:
        return None

    final_row = df.iloc[-1]
    return {
        "final_epoch": int(final_row["epoch"]),
        "final_val_acc": float(final_row["val_acc"]),
        "final_val_loss": float(final_row["val_loss"]),
        "final_val_error": float(final_row["val_error"]),
    }


def get_best_metrics_from_metrics_csv(metrics_csv_path):
    df = load_metrics_csv(metrics_csv_path)
    if df is None or len(df) == 0:
        return None

    best_val_error = df["val_error"].min()
    best_rows = df[df["val_error"] == best_val_error].sort_values("epoch")
    best_row = best_rows.iloc[0]

    return {
        "best_epoch": int(best_row["epoch"]),
        "best_val_acc": float(best_row["val_acc"]),
        "best_val_loss": float(best_row["val_loss"]),
        "best_val_error": float(best_row["val_error"]),
    }


# =========================================================
# Build aggregated records
# =========================================================
def build_repeat_level_records():
    rows = []

    print("Building repeat-level records from latest metrics.csv + raw FTLE files")
    print(f"Using widths: {WIDTHS}")
    print(f"Using repeats: {REPEAT_IDS}")

    for bw in WIDTHS:
        for repeat_id in REPEAT_IDS:
            metrics_csv = get_metrics_csv_path(bw, repeat_id)
            ftle_dir = get_ftle_dir(bw, repeat_id)

            print("=" * 80)
            print(f"[info] bw={bw} repeat={repeat_id}")
            print(f"       metrics_csv={metrics_csv}")
            print(f"       ftle_dir={ftle_dir}")

            final_info = get_final_metrics_from_metrics_csv(metrics_csv)
            if final_info is None:
                print(f"[skip] bw={bw} repeat={repeat_id}: cannot read final metrics")
                continue

            best_info = get_best_metrics_from_metrics_csv(metrics_csv)
            if best_info is None:
                print(f"[skip] bw={bw} repeat={repeat_id}: cannot read best metrics")
                continue

            epoch_files = list_epoch_files(ftle_dir)
            if len(epoch_files) == 0:
                print(f"[skip] bw={bw} repeat={repeat_id}: no FTLE files found")
                continue

            final_logged_epoch, final_ftle_file = choose_final_epoch_file(epoch_files)
            if final_ftle_file is None:
                print(f"[skip] bw={bw} repeat={repeat_id}: no final FTLE file")
                continue
            final_ftle = load_pkl_mean(final_ftle_file)

            best_epoch = best_info["best_epoch"]
            best_logged_epoch, best_ftle_file = choose_best_epoch_file(
                epoch_files,
                target_epoch=best_epoch,
                allow_fallback=ALLOW_FALLBACK_TO_PREV_LOGGED_EPOCH
            )
            if best_ftle_file is None:
                print(f"[skip] bw={bw} repeat={repeat_id}: no FTLE file for best epoch {best_epoch}")
                continue
            best_ftle = load_pkl_mean(best_ftle_file)

            print(
                f"[ok] bw={bw} repeat={repeat_id}: "
                f"final_dd={final_info['final_val_error']:.6f}, "
                f"best_dd={best_info['best_val_error']:.6f}, "
                f"best_epoch={best_epoch}, "
                f"final_ftle={final_ftle:.6f}, "
                f"best_ftle={best_ftle:.6f}"
            )

            rows.append({
                "base_width": bw,
                "repeat_id": repeat_id,

                "final_dd": final_info["final_val_error"],
                "final_val_acc": final_info["final_val_acc"],
                "final_val_loss": final_info["final_val_loss"],
                "final_epoch": final_info["final_epoch"],

                "best_dd": best_info["best_val_error"],
                "best_val_acc": best_info["best_val_acc"],
                "best_val_loss": best_info["best_val_loss"],
                "best_epoch": best_info["best_epoch"],

                "final_ftle": final_ftle,
                "final_ftle_abs": abs(final_ftle),
                "final_ftle_logged_epoch": final_logged_epoch,

                "best_ftle": best_ftle,
                "best_ftle_abs": abs(best_ftle),
                "best_ftle_logged_epoch": best_logged_epoch,
            })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(["base_width", "repeat_id"]).reset_index(drop=True)
    return df


def aggregate_by_width(df_repeat):
    if len(df_repeat) == 0:
        return pd.DataFrame()

    agg_rows = []
    grouped = df_repeat.groupby("base_width", sort=True)

    for bw, g in grouped:
        agg_rows.append({
            "base_width": bw,
            "num_repeats_found": len(g),

            "final_dd_mean": g["final_dd"].mean(),
            "final_dd_std": g["final_dd"].std(ddof=1) if len(g) > 1 else 0.0,

            "best_dd_mean": g["best_dd"].mean(),
            "best_dd_std": g["best_dd"].std(ddof=1) if len(g) > 1 else 0.0,

            "final_ftle_mean": g["final_ftle"].mean(),
            "final_ftle_std": g["final_ftle"].std(ddof=1) if len(g) > 1 else 0.0,

            "best_ftle_mean": g["best_ftle"].mean(),
            "best_ftle_std": g["best_ftle"].std(ddof=1) if len(g) > 1 else 0.0,

            "final_ftle_abs_mean": g["final_ftle_abs"].mean(),
            "final_ftle_abs_std": g["final_ftle_abs"].std(ddof=1) if len(g) > 1 else 0.0,

            "best_ftle_abs_mean": g["best_ftle_abs"].mean(),
            "best_ftle_abs_std": g["best_ftle_abs"].std(ddof=1) if len(g) > 1 else 0.0,
        })

    df_agg = pd.DataFrame(agg_rows)
    if len(df_agg) > 0:
        df_agg = df_agg.sort_values("base_width").reset_index(drop=True)
    return df_agg


# =========================================================
# Plotting
# =========================================================
def save_curve_outputs(
    df_agg,
    dd_mean_col,
    dd_std_col,
    ftle_mean_col,
    ftle_std_col,
    combo_name,
    dd_label,
    ftle_label,
    title
):
    out_csv = make_output_path(f"{OUT_PREFIX}_{combo_name}.csv")
    out_png = make_output_path(f"{OUT_PREFIX}_{combo_name}.png")
    out_pdf = make_output_path(f"{OUT_PREFIX}_{combo_name}.pdf")

    out_df = df_agg[[
        "base_width",
        "num_repeats_found",
        dd_mean_col,
        dd_std_col,
        ftle_mean_col,
        ftle_std_col,
    ]].copy()

    out_df.to_csv(out_csv, index=False)
    print(f"Saved aggregated CSV to: {out_csv}")

    widths = out_df["base_width"].to_numpy()
    dd_mean = out_df[dd_mean_col].to_numpy()
    dd_std = out_df[dd_std_col].to_numpy()
    ftle_mean = out_df[ftle_mean_col].to_numpy()
    ftle_std = out_df[ftle_std_col].to_numpy()

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(widths, dd_mean, marker="o", color="blue", label=dd_label)
    ax1.fill_between(widths, dd_mean - dd_std, dd_mean + dd_std, color="blue", alpha=0.15)
    ax1.set_xlabel("Base width")
    ax1.set_ylabel(dd_label, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(widths, ftle_mean, marker="x", color="orange", label=ftle_label)
    ax2.fill_between(widths, ftle_mean - ftle_std, ftle_mean + ftle_std, color="orange", alpha=0.15)
    ax2.set_ylabel("FTLE (Benettin)", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    ax2.axhline(
        y=0.0,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Edge of chaos (=0)"
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()

    print(f"Saved curve figure to: {out_png}")
    print(f"Saved curve figure to: {out_pdf}")


def main():
    configure_from_args(parse_args())

    df_repeat = build_repeat_level_records()
    if len(df_repeat) == 0:
        raise RuntimeError("No valid repeat-level records were built. Please check paths and files.")

    repeat_csv = make_output_path(f"{OUT_PREFIX}_repeat_level_records.csv")
    df_repeat.to_csv(repeat_csv, index=False)
    print(f"Saved repeat-level CSV to: {repeat_csv}")

    df_agg = aggregate_by_width(df_repeat)
    if len(df_agg) == 0:
        raise RuntimeError("Aggregation by width failed.")

    agg_csv = make_output_path(f"{OUT_PREFIX}_aggregated_by_width.csv")
    df_agg.to_csv(agg_csv, index=False)
    print(f"Saved aggregated-by-width CSV to: {agg_csv}")

    print("\nFinal widths included in aggregated curve plots:")
    print(df_agg["base_width"].tolist())

    # 1) final DD + final FTLE
    save_curve_outputs(
        df_agg=df_agg,
        dd_mean_col="final_dd_mean",
        dd_std_col="final_dd_std",
        ftle_mean_col="final_ftle_mean",
        ftle_std_col="final_ftle_std",
        combo_name="final_dd_final_ftle",
        dd_label="Final validation error (mean ± std)",
        ftle_label="Final FTLE (mean ± std)",
        title="Final DD vs Final FTLE (aggregated over repeats)"
    )

    # 2) final DD + best FTLE
    save_curve_outputs(
        df_agg=df_agg,
        dd_mean_col="final_dd_mean",
        dd_std_col="final_dd_std",
        ftle_mean_col="best_ftle_mean",
        ftle_std_col="best_ftle_std",
        combo_name="final_dd_best_ftle",
        dd_label="Final validation error (mean ± std)",
        ftle_label="Best-epoch FTLE (mean ± std)",
        title="Final DD vs Best-Epoch FTLE (aggregated over repeats)"
    )

    # 3) best DD + best FTLE
    save_curve_outputs(
        df_agg=df_agg,
        dd_mean_col="best_dd_mean",
        dd_std_col="best_dd_std",
        ftle_mean_col="best_ftle_mean",
        ftle_std_col="best_ftle_std",
        combo_name="best_dd_best_ftle",
        dd_label="Best validation error (mean ± std)",
        ftle_label="Best-epoch FTLE (mean ± std)",
        title="Best DD vs Best-Epoch FTLE (aggregated over repeats)"
    )


if __name__ == "__main__":
    main()
