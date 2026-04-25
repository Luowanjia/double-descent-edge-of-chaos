import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Default config. Override from the command line; do not edit these for routine use.
# =========================================================
RESULTS_ROOT = "results_dd"

ARCH = "cnn_dynamics"
RUN_PREFIX = None
RUN_SUFFIX = None

WIDTHS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 24, 32]
REPEAT_IDS = [0, 1, 2]

OUT_PREFIX = "val_loss_repeats"
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
        description="Plot validation-loss summaries by model width."
    )
    parser.add_argument("--results-root", default=RESULTS_ROOT,
                        help="root directory containing metrics.csv outputs")
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
    return parser.parse_args()


def configure_from_args(args):
    global RESULTS_ROOT, ARCH, RUN_PREFIX, RUN_SUFFIX
    global WIDTHS, REPEAT_IDS, OUT_PREFIX, OUTDIR

    RESULTS_ROOT = args.results_root
    ARCH = args.architecture
    RUN_PREFIX = args.run_prefix
    RUN_SUFFIX = args.run_suffix
    WIDTHS = args.widths
    REPEAT_IDS = args.repeat_ids
    OUT_PREFIX = args.out_prefix
    OUTDIR = args.outdir
    os.makedirs(OUTDIR, exist_ok=True)


def get_run_name(bw):
    return f"{RUN_PREFIX}{bw}{RUN_SUFFIX}"


def get_metrics_csv_path(bw, repeat_id):
    return os.path.join(
        RESULTS_ROOT,
        ARCH,
        ARCH,
        get_run_name(bw),
        f"repeat_{repeat_id}",
        "metrics.csv"
    )


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


def get_final_metrics(df):
    final_row = df.iloc[-1]
    return {
        "final_epoch": int(final_row["epoch"]),
        "final_val_acc": float(final_row["val_acc"]),
        "final_val_error": float(final_row["val_error"]),
        "final_val_loss": float(final_row["val_loss"]),
    }


def get_best_metrics(df):
    # keep consistent with your current pipeline:
    # "best" is defined by minimum validation error (= max val_acc)
    best_val_error = df["val_error"].min()
    best_rows = df[df["val_error"] == best_val_error].sort_values("epoch")
    best_row = best_rows.iloc[0]

    return {
        "best_epoch": int(best_row["epoch"]),
        "best_val_acc": float(best_row["val_acc"]),
        "best_val_error": float(best_row["val_error"]),
        "best_val_loss_at_best_error_epoch": float(best_row["val_loss"]),
    }


def get_min_val_loss_metrics(df):
    min_loss = df["val_loss"].min()
    min_rows = df[df["val_loss"] == min_loss].sort_values("epoch")
    min_row = min_rows.iloc[0]

    return {
        "min_loss_epoch": int(min_row["epoch"]),
        "min_val_loss": float(min_row["val_loss"]),
        "val_error_at_min_loss_epoch": float(min_row["val_error"]),
        "val_acc_at_min_loss_epoch": float(min_row["val_acc"]),
    }


# =========================================================
# Build repeat-level records
# =========================================================
def build_repeat_level_records():
    rows = []

    print("Building validation-loss records from latest metrics.csv")
    print(f"Using widths: {WIDTHS}")
    print(f"Using repeats: {REPEAT_IDS}")

    for bw in WIDTHS:
        for repeat_id in REPEAT_IDS:
            metrics_csv = get_metrics_csv_path(bw, repeat_id)

            print("=" * 80)
            print(f"[info] bw={bw} repeat={repeat_id}")
            print(f"       metrics_csv={metrics_csv}")

            df = load_metrics_csv(metrics_csv)
            if df is None:
                print(f"[skip] bw={bw} repeat={repeat_id}: cannot read metrics")
                continue

            final_info = get_final_metrics(df)
            best_info = get_best_metrics(df)
            min_loss_info = get_min_val_loss_metrics(df)

            print(
                f"[ok] bw={bw} repeat={repeat_id}: "
                f"final_val_loss={final_info['final_val_loss']:.6f}, "
                f"best_val_loss_at_best_error_epoch={best_info['best_val_loss_at_best_error_epoch']:.6f}, "
                f"min_val_loss={min_loss_info['min_val_loss']:.6f}"
            )

            rows.append({
                "base_width": bw,
                "repeat_id": repeat_id,

                "final_epoch": final_info["final_epoch"],
                "final_val_loss": final_info["final_val_loss"],
                "final_val_error": final_info["final_val_error"],

                "best_epoch": best_info["best_epoch"],
                "best_val_loss_at_best_error_epoch": best_info["best_val_loss_at_best_error_epoch"],
                "best_val_error": best_info["best_val_error"],

                "min_loss_epoch": min_loss_info["min_loss_epoch"],
                "min_val_loss": min_loss_info["min_val_loss"],
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

            "final_val_loss_mean": g["final_val_loss"].mean(),
            "final_val_loss_std": g["final_val_loss"].std(ddof=1) if len(g) > 1 else 0.0,

            "best_val_loss_at_best_error_epoch_mean": g["best_val_loss_at_best_error_epoch"].mean(),
            "best_val_loss_at_best_error_epoch_std": g["best_val_loss_at_best_error_epoch"].std(ddof=1) if len(g) > 1 else 0.0,

            "min_val_loss_mean": g["min_val_loss"].mean(),
            "min_val_loss_std": g["min_val_loss"].std(ddof=1) if len(g) > 1 else 0.0,
        })

    df_agg = pd.DataFrame(agg_rows)
    if len(df_agg) > 0:
        df_agg = df_agg.sort_values("base_width").reset_index(drop=True)
    return df_agg


# =========================================================
# Plotting
# =========================================================
def save_single_curve(df_agg, mean_col, std_col, ylabel, title, out_name):
    out_csv = make_output_path(f"{OUT_PREFIX}_{out_name}.csv")
    out_png = make_output_path(f"{OUT_PREFIX}_{out_name}.png")
    out_pdf = make_output_path(f"{OUT_PREFIX}_{out_name}.pdf")

    out_df = df_agg[["base_width", "num_repeats_found", mean_col, std_col]].copy()
    out_df.to_csv(out_csv, index=False)

    x = out_df["base_width"].to_numpy()
    y = out_df[mean_col].to_numpy()
    s = out_df[std_col].to_numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.fill_between(x, y - s, y + s, alpha=0.15)
    plt.xlabel("Base width")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def save_combined_curve(df_agg):
    out_csv = make_output_path(f"{OUT_PREFIX}_combined.csv")
    out_png = make_output_path(f"{OUT_PREFIX}_combined.png")
    out_pdf = make_output_path(f"{OUT_PREFIX}_combined.pdf")

    out_df = df_agg.copy()
    out_df.to_csv(out_csv, index=False)

    x = out_df["base_width"].to_numpy()

    y1 = out_df["final_val_loss_mean"].to_numpy()
    s1 = out_df["final_val_loss_std"].to_numpy()

    y2 = out_df["best_val_loss_at_best_error_epoch_mean"].to_numpy()
    s2 = out_df["best_val_loss_at_best_error_epoch_std"].to_numpy()

    y3 = out_df["min_val_loss_mean"].to_numpy()
    s3 = out_df["min_val_loss_std"].to_numpy()

    plt.figure(figsize=(8, 5))

    plt.plot(x, y1, marker="o", label="Final val loss")
    plt.fill_between(x, y1 - s1, y1 + s1, alpha=0.12)

    plt.plot(x, y2, marker="x", label="Val loss at best-error epoch")
    plt.fill_between(x, y2 - s2, y2 + s2, alpha=0.12)

    plt.plot(x, y3, marker="s", label="Minimum val loss")
    plt.fill_between(x, y3 - s3, y3 + s3, alpha=0.12)

    plt.xlabel("Base width")
    plt.ylabel("Validation loss")
    plt.title("Validation loss vs width (aggregated over repeats)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


# =========================================================
# Main
# =========================================================
def main():
    configure_from_args(parse_args())

    df_repeat = build_repeat_level_records()
    if len(df_repeat) == 0:
        raise RuntimeError("No valid repeat-level records were built. Please check paths and files.")

    repeat_csv = make_output_path(f"{OUT_PREFIX}_repeat_level_records.csv")
    df_repeat.to_csv(repeat_csv, index=False)
    print(f"Saved: {repeat_csv}")

    df_agg = aggregate_by_width(df_repeat)
    if len(df_agg) == 0:
        raise RuntimeError("Aggregation by width failed.")

    agg_csv = make_output_path(f"{OUT_PREFIX}_aggregated_by_width.csv")
    df_agg.to_csv(agg_csv, index=False)
    print(f"Saved: {agg_csv}")

    print("\nFinal widths included in loss plots:")
    print(df_agg["base_width"].tolist())

    save_single_curve(
        df_agg,
        mean_col="final_val_loss_mean",
        std_col="final_val_loss_std",
        ylabel="Final validation loss (mean ± std)",
        title="Final validation loss vs width (aggregated over repeats)",
        out_name="final_val_loss"
    )

    save_single_curve(
        df_agg,
        mean_col="best_val_loss_at_best_error_epoch_mean",
        std_col="best_val_loss_at_best_error_epoch_std",
        ylabel="Validation loss at best-error epoch (mean ± std)",
        title="Validation loss at best-error epoch vs width (aggregated over repeats)",
        out_name="val_loss_at_best_error_epoch"
    )

    save_single_curve(
        df_agg,
        mean_col="min_val_loss_mean",
        std_col="min_val_loss_std",
        ylabel="Minimum validation loss (mean ± std)",
        title="Minimum validation loss vs width (aggregated over repeats)",
        out_name="min_val_loss"
    )

    save_combined_curve(df_agg)


if __name__ == "__main__":
    main()
