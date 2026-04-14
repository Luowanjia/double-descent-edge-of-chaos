import os
import re
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# Config
# =========================================================
RESULT_CSV = "plots_50k_cnndyn_withchaos_fixed/model_wise_summary.csv"
RESULTS_ROOT = "results_dd"
RAW_ROOT = "rawdata/lyapunov1s"

ARCH = "cnn_dynamics"
RUN_PREFIX = "sgd_mom0.0_lr0.1_lrschinverse_sqrt_decay512_bw"
RUN_SUFFIX = "_bs128_wd0_noise0.15_full_relu_steps50000_iter100_withchaos"

REPEAT_ID = 0
MAX_WIDTH = 24

# If best epoch does not have an exact chaos file,
# allow fallback to the latest logged epoch <= best epoch.
ALLOW_FALLBACK_TO_PREV_LOGGED_EPOCH = True

# Output prefix
OUT_PREFIX = "dd_vs_chaos_lyapunov1"


# =========================================================
# Helpers
# =========================================================
def load_summary_csv(path):
    df = pd.read_csv(path)
    df = df.sort_values("base_width").reset_index(drop=True)

    if MAX_WIDTH is not None:
        df = df[df["base_width"] <= MAX_WIDTH].reset_index(drop=True)

    return df


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


def get_best_metrics_from_metrics_csv(metrics_csv_path):
    """
    Define best epoch by minimum validation error = 1 - val_acc,
    i.e. maximum val_acc.

    If there are ties, choose the earliest epoch.
    """
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

    best_val_error = df["val_error"].min()
    best_rows = df[df["val_error"] == best_val_error].sort_values("epoch")
    best_row = best_rows.iloc[0]

    return {
        "best_epoch": int(best_row["epoch"]),
        "best_val_acc": float(best_row["val_acc"]),
        "best_val_loss": float(best_row["val_loss"]),
        "best_val_error": float(best_row["val_error"]),
    }


def build_width_records(summary_df):
    """
    Build one record per width by combining:
    - final DD from summary csv
    - best DD from per-run metrics.csv
    - final chaos from last logged pkl
    - best chaos from best-epoch matched pkl
    """
    records = []

    for _, row in summary_df.iterrows():
        bw = int(row["base_width"])
        final_dd = float(row["final_val_error_mean"])

        metrics_csv = get_metrics_csv_path(bw)
        lyp_dir = get_lyapunov1_dir(bw)
        epoch_files = list_epoch_files(lyp_dir)

        if len(epoch_files) == 0:
            print(f"[skip] bw={bw}: no chaos files found in {lyp_dir}")
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
            f"final_dd={final_dd:.6f}, "
            f"best_dd={best_info['best_val_error']:.6f}, "
            f"best_epoch={best_epoch}, "
            f"final_chaos={final_chaos:.6f} (logged_epoch={final_logged_epoch}), "
            f"best_chaos={best_chaos:.6f} (used_logged_epoch={best_logged_epoch})"
        )

        records.append({
            "base_width": bw,

            "final_dd": final_dd,

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
    title
):
    out_csv = f"{OUT_PREFIX}_{combo_name}.csv"
    out_png = f"{OUT_PREFIX}_{combo_name}.png"
    out_pdf = f"{OUT_PREFIX}_{combo_name}.pdf"

    out_df = df_records[[
        "base_width",
        dd_col,
        chaos_col,
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
    summary_df = load_summary_csv(RESULT_CSV)

    if "base_width" not in summary_df.columns:
        raise ValueError("RESULT_CSV must contain column: base_width")

    if "final_val_error_mean" not in summary_df.columns:
        raise ValueError("RESULT_CSV must contain column: final_val_error_mean")

    df_records = build_width_records(summary_df)

    if len(df_records) == 0:
        raise RuntimeError("No valid width records were built. Please check paths and files.")

    # 1) final DD + final lyp
    save_combo_outputs(
        df_records=df_records,
        dd_col="final_dd",
        chaos_col="final_chaos",
        combo_name="final_dd_final_chaos",
        dd_label="Final validation error",
        chaos_label="Normalized Jacobian norm (final logged epoch)",
        title="Final DD vs Final Chaos Metric"
    )

    # 2) final DD + best lyp
    save_combo_outputs(
        df_records=df_records,
        dd_col="final_dd",
        chaos_col="best_chaos",
        combo_name="final_dd_best_chaos",
        dd_label="Final validation error",
        chaos_label="Normalized Jacobian norm (best-epoch matched)",
        title="Final DD vs Best-Epoch Chaos Metric"
    )

    # 3) best DD + best lyp
    save_combo_outputs(
        df_records=df_records,
        dd_col="best_dd",
        chaos_col="best_chaos",
        combo_name="best_dd_best_chaos",
        dd_label="Best validation error",
        chaos_label="Normalized Jacobian norm (best-epoch matched)",
        title="Best DD vs Best-Epoch Chaos Metric"
    )


if __name__ == "__main__":
    main()
