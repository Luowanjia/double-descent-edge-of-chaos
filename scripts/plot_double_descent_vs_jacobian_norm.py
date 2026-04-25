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
RAW_ROOT = "rawdata/lyapunov1s"

ARCH = "cnn_dynamics"
RUN_PREFIX = None
RUN_SUFFIX = None

REPEAT_ID = 0

# If None, auto-discover widths from RAW_ROOT
WIDTHS = None

# Optional cap to mimic your old plot style
MAX_WIDTH = 24

# If best epoch does not have an exact chaos file,
# allow fallback to the latest logged epoch <= best epoch.
ALLOW_FALLBACK_TO_PREV_LOGGED_EPOCH = True

# Output prefix
OUT_PREFIX = "dd_vs_chaos_lyapunov1"
OUTDIR = "."

# Shared y-axis across all JN plots
USE_SHARED_CHAOS_YLIM = True
CHAOS_YLIM_MARGIN_RATIO = 0.05


# =========================================================
# Helpers
# =========================================================
def parse_int_list(value):
    if value is None:
        return None
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
        description="Plot double-descent metrics against Jacobian-norm/Lyapunov outputs."
    )
    parser.add_argument("--results-root", default=RESULTS_ROOT,
                        help="root directory containing metrics.csv outputs")
    parser.add_argument("--raw-root", default=RAW_ROOT,
                        help="root directory containing Lyapunov/Jacobian-norm pickle outputs")
    parser.add_argument("--architecture", default=ARCH,
                        help="architecture directory name, e.g. cnn_dynamics")
    parser.add_argument("--run-prefix", required=True,
                        help="run directory prefix before the base width")
    parser.add_argument("--run-suffix", required=True,
                        help="run directory suffix after the base width")
    parser.add_argument("--repeat-id", type=int, default=REPEAT_ID,
                        help="repeat id to plot")
    parser.add_argument("--widths", type=parse_int_list, default=WIDTHS,
                        help="comma-separated base widths; omit to auto-discover from raw-root")
    parser.add_argument("--max-width", type=int, default=MAX_WIDTH,
                        help="maximum auto-discovered width; use -1 for no cap")
    parser.add_argument("--out-prefix", default=OUT_PREFIX,
                        help="prefix for generated CSV/PNG/PDF files")
    parser.add_argument("--outdir", default=OUTDIR,
                        help="directory for generated CSV/PNG/PDF files")
    parser.add_argument("--no-fallback-to-prev-epoch", action="store_true",
                        help="require exact best-epoch chaos files")
    parser.add_argument("--no-shared-chaos-ylim", action="store_true",
                        help="use independent y-limits for each chaos axis")
    parser.add_argument("--chaos-ylim-margin-ratio", type=float,
                        default=CHAOS_YLIM_MARGIN_RATIO,
                        help="margin ratio for shared chaos y-axis limits")
    return parser.parse_args()


def configure_from_args(args):
    global RESULTS_ROOT, RAW_ROOT, ARCH, RUN_PREFIX, RUN_SUFFIX
    global REPEAT_ID, WIDTHS, MAX_WIDTH, ALLOW_FALLBACK_TO_PREV_LOGGED_EPOCH
    global OUT_PREFIX, OUTDIR, USE_SHARED_CHAOS_YLIM, CHAOS_YLIM_MARGIN_RATIO

    RESULTS_ROOT = args.results_root
    RAW_ROOT = args.raw_root
    ARCH = args.architecture
    RUN_PREFIX = args.run_prefix
    RUN_SUFFIX = args.run_suffix
    REPEAT_ID = args.repeat_id
    WIDTHS = args.widths
    MAX_WIDTH = None if args.max_width < 0 else args.max_width
    ALLOW_FALLBACK_TO_PREV_LOGGED_EPOCH = not args.no_fallback_to_prev_epoch
    OUT_PREFIX = args.out_prefix
    OUTDIR = args.outdir
    USE_SHARED_CHAOS_YLIM = not args.no_shared_chaos_ylim
    CHAOS_YLIM_MARGIN_RATIO = args.chaos_ylim_margin_ratio
    os.makedirs(OUTDIR, exist_ok=True)


def get_run_name(bw):
    return f"{RUN_PREFIX}{bw}{RUN_SUFFIX}"


def get_lyapunov1_dir(bw):
    return os.path.join(
        RAW_ROOT,
        ARCH,
        get_run_name(bw),
        f"repeat_{REPEAT_ID}",
    )


def get_metrics_csv_path(bw):
    return os.path.join(
        RESULTS_ROOT,
        ARCH,
        ARCH,
        get_run_name(bw),
        f"repeat_{REPEAT_ID}",
        "metrics.csv"
    )


def parse_epoch_from_filename(path):
    fname = os.path.basename(path)
    m = re.match(r"epoch(\d+)\.pkl$", fname)
    if m is None:
        return None
    return int(m.group(1))


def parse_bw_from_run_dirname(name):
    m = re.search(r"_bw(\d+)_", name)
    if m is None:
        return None
    return int(m.group(1))


def auto_discover_widths():
    root = os.path.join(RAW_ROOT, ARCH)
    if not os.path.exists(root):
        return []

    discovered = []
    for name in os.listdir(root):
        bw = parse_bw_from_run_dirname(name)
        if bw is not None:
            discovered.append(bw)

    discovered = sorted(set(discovered))
    if MAX_WIDTH is not None:
        discovered = [bw for bw in discovered if bw <= MAX_WIDTH]
    return discovered


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
    """
    Prefer exact epoch match.
    If not found and allow_fallback=True, choose latest logged epoch <= target_epoch.
    """
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
    """
    Define best epoch by minimum validation error = 1 - val_acc,
    i.e. maximum val_acc.

    If there are ties, choose the earliest epoch.
    """
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


def compute_shared_chaos_ylim(df_records, margin_ratio=0.05):
    """
    Compute one shared y-axis range across all JN/chaos curve plots.
    Uses both final_chaos and best_chaos.
    """
    global_min = min(df_records["final_chaos"].min(), df_records["best_chaos"].min())
    global_max = max(df_records["final_chaos"].max(), df_records["best_chaos"].max())

    if np.isclose(global_min, global_max):
        margin = 0.1
    else:
        margin = margin_ratio * (global_max - global_min)

    return float(global_min - margin), float(global_max + margin)


# =========================================================
# Build records directly from raw files
# =========================================================
def build_width_records(widths):
    """
    Build one record per width by combining:
    - final DD from metrics.csv
    - best DD from metrics.csv
    - final chaos from last logged pkl
    - best chaos from best-epoch matched pkl
    """
    records = []

    print("Building JN records directly from metrics.csv + raw lyapunov1 files")
    print(f"Using widths: {widths}")
    print(f"Using repeat: {REPEAT_ID}")

    for bw in widths:
        metrics_csv = get_metrics_csv_path(bw)
        lyp_dir = get_lyapunov1_dir(bw)
        epoch_files = list_epoch_files(lyp_dir)

        print("=" * 80)
        print(f"[info] bw={bw}")
        print(f"       metrics_csv={metrics_csv}")
        print(f"       lyp_dir={lyp_dir}")

        if len(epoch_files) == 0:
            print(f"[skip] bw={bw}: no chaos files found in {lyp_dir}")
            continue

        final_info = get_final_metrics_from_metrics_csv(metrics_csv)
        if final_info is None:
            print(f"[skip] bw={bw}: cannot read final metrics from {metrics_csv}")
            continue

        best_info = get_best_metrics_from_metrics_csv(metrics_csv)
        if best_info is None:
            print(f"[skip] bw={bw}: cannot read best metrics from {metrics_csv}")
            continue

        # Final chaos
        final_logged_epoch, final_chaos_file = choose_final_epoch_file(epoch_files)
        if final_chaos_file is None:
            print(f"[skip] bw={bw}: no final chaos file")
            continue
        final_chaos = load_pkl_mean(final_chaos_file)

        # Best chaos
        best_epoch = best_info["best_epoch"]
        best_logged_epoch, best_chaos_file = choose_best_epoch_file(
            epoch_files,
            target_epoch=best_epoch,
            allow_fallback=ALLOW_FALLBACK_TO_PREV_LOGGED_EPOCH
        )
        if best_chaos_file is None:
            print(
                f"[skip] bw={bw}: no chaos file for best epoch {best_epoch} "
                f"(and no earlier fallback found)"
            )
            continue
        best_chaos = load_pkl_mean(best_chaos_file)

        print(
            f"[ok] bw={bw}: "
            f"final_dd={final_info['final_val_error']:.6f}, "
            f"best_dd={best_info['best_val_error']:.6f}, "
            f"best_epoch={best_epoch}, "
            f"final_chaos={final_chaos:.6f} (logged_epoch={final_logged_epoch}), "
            f"best_chaos={best_chaos:.6f} (used_logged_epoch={best_logged_epoch})"
        )

        records.append({
            "base_width": bw,

            "final_dd": final_info["final_val_error"],
            "final_val_acc": final_info["final_val_acc"],
            "final_val_loss": final_info["final_val_loss"],
            "final_epoch": final_info["final_epoch"],

            "best_dd": best_info["best_val_error"],
            "best_val_acc": best_info["best_val_acc"],
            "best_val_loss": best_info["best_val_loss"],
            "best_epoch": best_info["best_epoch"],

            "final_chaos": final_chaos,
            "final_chaos_logged_epoch": final_logged_epoch,

            "best_chaos": best_chaos,
            "best_chaos_logged_epoch": best_logged_epoch,
        })

    return pd.DataFrame(records)


def save_combo_outputs(
    df_records,
    dd_col,
    chaos_col,
    combo_name,
    dd_label,
    chaos_label,
    title,
    shared_chaos_ylim=None
):
    out_csv = make_output_path(f"{OUT_PREFIX}_{combo_name}.csv")
    out_png = make_output_path(f"{OUT_PREFIX}_{combo_name}.png")
    out_pdf = make_output_path(f"{OUT_PREFIX}_{combo_name}.pdf")

    out_df = df_records[[
        "base_width",
        dd_col,
        chaos_col,
        "final_epoch",
        "best_epoch",
        "best_val_acc",
        "best_val_loss",
        "final_chaos_logged_epoch",
        "best_chaos_logged_epoch",
    ]].copy()

    out_df.to_csv(out_csv, index=False)
    print(f"Saved merged CSV to: {out_csv}")

    widths = out_df["base_width"].tolist()
    dd_vals = out_df[dd_col].tolist()
    chaos_vals = out_df[chaos_col].tolist()

    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Left axis: DD
    ax1.plot(
        widths,
        dd_vals,
        marker="o",
        color="blue",
        label=dd_label
    )
    ax1.set_xlabel("Base width")
    ax1.set_ylabel(dd_label, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Right axis: chaos
    ax2 = ax1.twinx()
    ax2.plot(
        widths,
        chaos_vals,
        marker="x",
        color="orange",
        label=chaos_label
    )
    ax2.set_ylabel("Normalized Jacobian norm", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    if shared_chaos_ylim is not None:
        ax2.set_ylim(shared_chaos_ylim[0], shared_chaos_ylim[1])

    # Edge of chaos threshold
    ax2.axhline(
        y=1.0,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Edge of chaos (=1)"
    )

    # Combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()

    print(f"Saved figure to: {out_png}")
    print(f"Saved figure to: {out_pdf}")


# =========================================================
# Main
# =========================================================
def main():
    configure_from_args(parse_args())

    widths = WIDTHS
    if widths is None:
        widths = auto_discover_widths()

    if len(widths) == 0:
        raise RuntimeError(
            f"No widths found under {os.path.join(RAW_ROOT, ARCH)}. "
            "Please check whether old lyapunov1 files still exist."
        )

    df_records = build_width_records(widths)

    if len(df_records) == 0:
        raise RuntimeError("No valid width records were built. Please check paths and files.")

    df_records = df_records.sort_values("base_width").reset_index(drop=True)

    print("\nFinal widths included in JN plots:")
    print(df_records["base_width"].tolist())

    master_csv = make_output_path(f"{OUT_PREFIX}_all_records.csv")
    df_records.to_csv(master_csv, index=False)
    print(f"Saved master merged CSV to: {master_csv}")

    shared_chaos_ylim = None
    if USE_SHARED_CHAOS_YLIM:
        shared_chaos_ylim = compute_shared_chaos_ylim(
            df_records,
            margin_ratio=CHAOS_YLIM_MARGIN_RATIO
        )
        print(f"\nUsing shared JN/chaos y-limits across all curve plots: {shared_chaos_ylim}")

    # 1) final DD + final lyp
    save_combo_outputs(
        df_records=df_records,
        dd_col="final_dd",
        chaos_col="final_chaos",
        combo_name="final_dd_final_chaos",
        dd_label="Final validation error",
        chaos_label="Normalized Jacobian norm (final logged epoch)",
        title="Final DD vs Final Chaos Metric",
        shared_chaos_ylim=shared_chaos_ylim
    )

    # 2) final DD + best lyp
    save_combo_outputs(
        df_records=df_records,
        dd_col="final_dd",
        chaos_col="best_chaos",
        combo_name="final_dd_best_chaos",
        dd_label="Final validation error",
        chaos_label="Normalized Jacobian norm (best-epoch matched)",
        title="Final DD vs Best-Epoch Chaos Metric",
        shared_chaos_ylim=shared_chaos_ylim
    )

    # 3) best DD + best lyp
    save_combo_outputs(
        df_records=df_records,
        dd_col="best_dd",
        chaos_col="best_chaos",
        combo_name="best_dd_best_chaos",
        dd_label="Best validation error",
        chaos_label="Normalized Jacobian norm (best-epoch matched)",
        title="Best DD vs Best-Epoch Chaos Metric",
        shared_chaos_ylim=shared_chaos_ylim
    )


if __name__ == "__main__":
    main()
