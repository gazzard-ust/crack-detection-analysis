"""
Multi-Seed Results Analysis & Figure Generation
================================================
Reads multiseed_results.json and produces:
  1. Console summary with mean +/- std
  2. LaTeX-ready comparison table (printed + saved to .tex)
  3. fig_model_comparison.png — dot plot with error bars
  4. Statistical significance tests (Welch's t-test, 5 seeds per model)
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats


# ============================================
# CONFIGURATION
# ============================================

RESULTS_FILE = "multiseed_results.json"
FIGURE_PATH = "fig_model_comparison.png"
LATEX_PATH = "table_model_comparison.tex"
MARKDOWN_PATH = "multiseed_results_summary.md"
SPLIT = "test"  # Primary split for reporting

MODEL_LABELS = {
    "yoloworld_xl": "YOLO-World XL",
    "yolov8x": "YOLOv8x",
}

MAIN_METRICS = ["mAP50-95", "mAP50", "precision", "recall", "f1_score"]

METRIC_DISPLAY = {
    "mAP50-95": "mAP@50-95",
    "mAP50": "mAP@50",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1 Score",
}


# ============================================
# DATA LOADING
# ============================================

def load_results(path=RESULTS_FILE):
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run multiseed_runner.py first.")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    return data


def get_metric_arrays(data, model_name, split, metric):
    """Extract a numpy array of metric values across seeds for a given model/split."""
    runs = [r for r in data["runs"] if r["model"] == model_name]
    values = [r[split][metric] for r in runs if split in r and metric in r[split]]
    return np.array(values)


# ============================================
# CONSOLE SUMMARY
# ============================================

def print_summary(data, split=SPLIT):
    print("=" * 70)
    print(f"MULTI-SEED RESULTS SUMMARY  ({split.upper()} set, {len(data.get('seeds', []))} seeds)")
    print("=" * 70)

    for model in ["yoloworld_xl", "yolov8x"]:
        label = MODEL_LABELS[model]
        print(f"\n  {label}")
        print(f"  {'─' * 40}")

        for metric in MAIN_METRICS:
            vals = get_metric_arrays(data, model, split, metric)
            if len(vals) == 0:
                continue
            print(f"    {METRIC_DISPLAY[metric]:<15} {np.mean(vals):.4f} +/- {np.std(vals, ddof=1):.4f}  (n={len(vals)})")

        # Per-class AP
        runs = [r for r in data["runs"] if r["model"] == model and split in r]
        if runs:
            class_keys = [k for k in runs[0][split] if k.startswith("AP50-95_")]
            if class_keys:
                print(f"\n    Per-class AP@50-95:")
                for key in sorted(class_keys):
                    vals = np.array([r[split][key] for r in runs])
                    class_name = key.replace("AP50-95_", "")
                    print(f"      {class_name:<20} {np.mean(vals):.4f} +/- {np.std(vals, ddof=1):.4f}")

    print()


# ============================================
# STATISTICAL SIGNIFICANCE
# ============================================

def run_significance_tests(data, split=SPLIT):
    """Paired t-test: same test set evaluated across seeds, so observations are paired by seed."""
    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE (Paired t-test)")
    print("=" * 70)
    print(f"  H0: No difference between YOLO-World XL and YOLOv8x")
    print(f"  Note: Paired by seed (same fixed test set across all runs)")
    print(f"  Split: {split.upper()}, Seeds: {data.get('seeds', [])}\n")

    results = {}

    for metric in MAIN_METRICS:
        yw_vals = get_metric_arrays(data, "yoloworld_xl", split, metric)
        v8_vals = get_metric_arrays(data, "yolov8x", split, metric)

        if len(yw_vals) < 2 or len(v8_vals) < 2:
            print(f"  {METRIC_DISPLAY[metric]:<15} — insufficient data for t-test")
            continue

        t_stat, p_value = scipy_stats.ttest_rel(yw_vals, v8_vals)
        diff = np.mean(yw_vals) - np.mean(v8_vals)
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."

        results[metric] = {"t_stat": t_stat, "p_value": p_value, "diff": diff, "sig": sig}

        print(f"  {METRIC_DISPLAY[metric]:<15}  diff={diff:+.4f}  t={t_stat:.3f}  p={p_value:.4f}  {sig}")

    print(f"\n  Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant\n")
    return results


# ============================================
# LATEX TABLE
# ============================================

def generate_latex_table(data, split=SPLIT, output_path=LATEX_PATH):
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comparison of YOLO-World XL and YOLOv8x on the test set (mean $\pm$ std over 5 seeds).}")
    lines.append(r"\label{tab:model_comparison}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Metric & YOLO-World XL & YOLOv8x \\")
    lines.append(r"\midrule")

    for metric in MAIN_METRICS:
        yw_vals = get_metric_arrays(data, "yoloworld_xl", split, metric)
        v8_vals = get_metric_arrays(data, "yolov8x", split, metric)

        yw_str = f"${np.mean(yw_vals):.4f} \\pm {np.std(yw_vals, ddof=1):.4f}$" if len(yw_vals) > 0 else "—"
        v8_str = f"${np.mean(v8_vals):.4f} \\pm {np.std(v8_vals, ddof=1):.4f}$" if len(v8_vals) > 0 else "—"

        # Bold the better result
        if len(yw_vals) > 0 and len(v8_vals) > 0:
            if np.mean(yw_vals) > np.mean(v8_vals):
                yw_str = r"\textbf{" + yw_str + "}"
            elif np.mean(v8_vals) > np.mean(yw_vals):
                v8_str = r"\textbf{" + v8_str + "}"

        display = METRIC_DISPLAY[metric]
        lines.append(f"{display} & {yw_str} & {v8_str} \\\\")

    # Per-class AP
    runs_yw = [r for r in data["runs"] if r["model"] == "yoloworld_xl" and split in r]
    runs_v8 = [r for r in data["runs"] if r["model"] == "yolov8x" and split in r]

    if runs_yw and runs_v8:
        class_keys = sorted([k for k in runs_yw[0][split] if k.startswith("AP50-95_")])
        if class_keys:
            lines.append(r"\midrule")
            for key in class_keys:
                class_name = key.replace("AP50-95_", "")
                yw_vals = np.array([r[split][key] for r in runs_yw])
                v8_vals = np.array([r[split][key] for r in runs_v8])

                yw_str = f"${np.mean(yw_vals):.4f} \\pm {np.std(yw_vals, ddof=1):.4f}$"
                v8_str = f"${np.mean(v8_vals):.4f} \\pm {np.std(v8_vals, ddof=1):.4f}$"

                if np.mean(yw_vals) > np.mean(v8_vals):
                    yw_str = r"\textbf{" + yw_str + "}"
                elif np.mean(v8_vals) > np.mean(yw_vals):
                    v8_str = r"\textbf{" + v8_str + "}"

                lines.append(f"AP@50-95 {class_name} & {yw_str} & {v8_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(latex)

    print(f"LaTeX table saved to: {output_path}")
    print()
    print(latex)
    print()


# ============================================
# FIGURE: DOT PLOT WITH ERROR BARS
# ============================================

def generate_comparison_figure(data, split=SPLIT, output_path=FIGURE_PATH):
    metrics_to_plot = MAIN_METRICS

    yw_means, yw_stds = [], []
    v8_means, v8_stds = [], []
    labels = []

    for metric in metrics_to_plot:
        yw_vals = get_metric_arrays(data, "yoloworld_xl", split, metric)
        v8_vals = get_metric_arrays(data, "yolov8x", split, metric)

        if len(yw_vals) == 0 or len(v8_vals) == 0:
            continue

        labels.append(METRIC_DISPLAY[metric])
        yw_means.append(np.mean(yw_vals))
        yw_stds.append(np.std(yw_vals, ddof=1))
        v8_means.append(np.mean(v8_vals))
        v8_stds.append(np.std(v8_vals, ddof=1))

    if not labels:
        print("WARNING: No data available for figure generation.")
        return

    y = np.arange(len(labels))
    offset = 0.15  # Vertical offset between the two models

    fig, ax = plt.subplots(figsize=(8, 5))

    # YOLO-World XL dots
    ax.errorbar(
        yw_means, y + offset, xerr=yw_stds, fmt='o',
        color="#2196F3", ecolor="#2196F3", elinewidth=1.5, capsize=4,
        capthick=1.5, markersize=8, markeredgecolor="white", markeredgewidth=1,
        label="YOLO-World XL", zorder=3,
    )
    # YOLOv8x dots
    ax.errorbar(
        v8_means, y - offset, xerr=v8_stds, fmt='s',
        color="#FF9800", ecolor="#FF9800", elinewidth=1.5, capsize=4,
        capthick=1.5, markersize=7, markeredgecolor="white", markeredgewidth=1,
        label="YOLOv8x", zorder=3,
    )

    # Value labels positioned past the error bar tips
    for i in range(len(labels)):
        yw_tip = yw_means[i] + yw_stds[i]
        v8_tip = v8_means[i] + v8_stds[i]
        ax.annotate(
            f"{yw_means[i]:.4f}", xy=(yw_tip, y[i] + offset),
            xytext=(6, 0), textcoords="offset points",
            ha="left", va="center", fontsize=8, color="#1565C0",
        )
        ax.annotate(
            f"{v8_means[i]:.4f}", xy=(v8_tip, y[i] - offset),
            xytext=(6, 0), textcoords="offset points",
            ha="left", va="center", fontsize=8, color="#E65100",
        )

    # Compute a zoomed x-axis range
    all_vals = yw_means + v8_means
    all_stds = yw_stds + v8_stds
    x_min = min(v - s for v, s in zip(all_vals, all_stds)) - 0.008
    x_max = max(v + s for v, s in zip(all_vals, all_stds)) + 0.025

    ax.set_xlim(x_min, x_max)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_title(
        f"YOLO-World XL vs YOLOv8x — {split.capitalize()} Set (5 seeds)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Light horizontal bands for readability
    for i in range(len(labels)):
        if i % 2 == 0:
            ax.axhspan(y[i] - 0.4, y[i] + 0.4, color="#f5f5f5", zorder=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure saved to: {output_path}")


# ============================================
# FIGURE: PER-CLASS AP DOT PLOT
# ============================================

PERCLASS_FIGURE_PATH = "fig_perclass_comparison.png"

def generate_perclass_figure(data, split=SPLIT, output_path=PERCLASS_FIGURE_PATH):
    runs_yw = [r for r in data["runs"] if r["model"] == "yoloworld_xl" and split in r]
    runs_v8 = [r for r in data["runs"] if r["model"] == "yolov8x" and split in r]

    if not runs_yw or not runs_v8:
        print("WARNING: No data available for per-class figure.")
        return

    class_keys = sorted([k for k in runs_yw[0][split] if k.startswith("AP50-95_")])
    if not class_keys:
        print("WARNING: No per-class AP keys found.")
        return

    class_names = [k.replace("AP50-95_", "") for k in class_keys]
    yw_means, yw_stds = [], []
    v8_means, v8_stds = [], []

    for key in class_keys:
        yw_vals = np.array([r[split][key] for r in runs_yw])
        v8_vals = np.array([r[split][key] for r in runs_v8])
        yw_means.append(np.mean(yw_vals))
        yw_stds.append(np.std(yw_vals, ddof=1))
        v8_means.append(np.mean(v8_vals))
        v8_stds.append(np.std(v8_vals, ddof=1))

    y = np.arange(len(class_names))
    offset = 0.15

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.errorbar(
        yw_means, y + offset, xerr=yw_stds, fmt='o',
        color="#2196F3", ecolor="#2196F3", elinewidth=1.5, capsize=4,
        capthick=1.5, markersize=8, markeredgecolor="white", markeredgewidth=1,
        label="YOLO-World XL", zorder=3,
    )
    ax.errorbar(
        v8_means, y - offset, xerr=v8_stds, fmt='s',
        color="#FF9800", ecolor="#FF9800", elinewidth=1.5, capsize=4,
        capthick=1.5, markersize=7, markeredgecolor="white", markeredgewidth=1,
        label="YOLOv8x", zorder=3,
    )

    for i in range(len(class_names)):
        yw_tip = yw_means[i] + yw_stds[i]
        v8_tip = v8_means[i] + v8_stds[i]
        ax.annotate(
            f"{yw_means[i]:.4f}", xy=(yw_tip, y[i] + offset),
            xytext=(6, 0), textcoords="offset points",
            ha="left", va="center", fontsize=8, color="#1565C0",
        )
        ax.annotate(
            f"{v8_means[i]:.4f}", xy=(v8_tip, y[i] - offset),
            xytext=(6, 0), textcoords="offset points",
            ha="left", va="center", fontsize=8, color="#E65100",
        )

    all_vals = yw_means + v8_means
    all_stds = yw_stds + v8_stds
    x_min = min(v - s for v, s in zip(all_vals, all_stds)) - 0.01
    x_max = max(v + s for v, s in zip(all_vals, all_stds)) + 0.03

    ax.set_xlim(x_min, x_max)
    ax.set_yticks(y)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("AP@50-95", fontsize=12)
    ax.set_title(
        f"Per-Class AP@50-95 — {split.capitalize()} Set (5 seeds)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11, loc="lower center")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    for i in range(len(class_names)):
        if i % 2 == 0:
            ax.axhspan(y[i] - 0.4, y[i] + 0.4, color="#f5f5f5", zorder=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure saved to: {output_path}")


# ============================================
# MARKDOWN REPORT
# ============================================

def generate_markdown_report(data, sig_results, split=SPLIT, output_path=MARKDOWN_PATH):
    """Generate a comprehensive markdown summary of all results."""

    seeds = data.get("seeds", [])
    lines = []

    lines.append("# Multi-Seed Baseline Comparison Results")
    lines.append("")
    lines.append("## Experimental Setup")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append("| Dataset | Pipe Crack Detection (3 classes: Dummy crack, PVC pipe crack, Paper crack) |")
    lines.append("| Train / Val / Test | 2292 / 217 / 108 images |")
    lines.append(f"| Seeds | {seeds} |")
    lines.append("| Epochs | 100 (early stopping, patience=20) |")
    lines.append("| Optimizer | AdamW (lr=2e-4, lrf=0.01, weight_decay=0.05) |")
    lines.append("| Image Size | 640 |")
    lines.append("| Augmentation | HSV, flip, mosaic, mixup, degrees, scale, shear |")
    lines.append("| Evaluation | conf=0.25, iou=0.45 |")
    lines.append("")
    lines.append("| Model | Architecture | Params | Batch Size |")
    lines.append("|-------|-------------|--------|------------|")
    lines.append("| YOLO-World XL | YOLOv8x-worldv2 (with CLIP text encoder) | ~72.9M | 8 |")
    lines.append("| YOLOv8x | Standard YOLOv8x (detection only) | ~68.2M | 16 |")
    lines.append("")

    # --- Aggregate comparison table ---
    lines.append(f"## Model Comparison ({split.capitalize()} Set)")
    lines.append("")
    lines.append("| Metric | YOLO-World XL | YOLOv8x | Diff |")
    lines.append("|--------|--------------|---------|------|")

    for metric in MAIN_METRICS:
        yw_vals = get_metric_arrays(data, "yoloworld_xl", split, metric)
        v8_vals = get_metric_arrays(data, "yolov8x", split, metric)

        yw_str = f"{np.mean(yw_vals):.4f} +/- {np.std(yw_vals, ddof=1):.4f}" if len(yw_vals) > 0 else "---"
        v8_str = f"{np.mean(v8_vals):.4f} +/- {np.std(v8_vals, ddof=1):.4f}" if len(v8_vals) > 0 else "---"

        if len(yw_vals) > 0 and len(v8_vals) > 0:
            diff = np.mean(yw_vals) - np.mean(v8_vals)
            diff_str = f"{diff:+.4f}"
            better = "**" if diff > 0 else ""
            yw_str = f"{better}{yw_str}{better}"
            if diff < 0:
                v8_str = f"**{v8_str}**"
        else:
            diff_str = "---"

        lines.append(f"| {METRIC_DISPLAY[metric]} | {yw_str} | {v8_str} | {diff_str} |")

    lines.append("")

    # --- Per-class AP ---
    runs_yw = [r for r in data["runs"] if r["model"] == "yoloworld_xl" and split in r]
    runs_v8 = [r for r in data["runs"] if r["model"] == "yolov8x" and split in r]

    if runs_yw and runs_v8:
        class_keys = sorted([k for k in runs_yw[0][split] if k.startswith("AP50-95_")])
        if class_keys:
            lines.append("### Per-Class AP@50-95")
            lines.append("")
            lines.append("| Class | YOLO-World XL | YOLOv8x | Diff |")
            lines.append("|-------|--------------|---------|------|")

            for key in class_keys:
                class_name = key.replace("AP50-95_", "")
                yw_vals = np.array([r[split][key] for r in runs_yw])
                v8_vals = np.array([r[split][key] for r in runs_v8])
                diff = np.mean(yw_vals) - np.mean(v8_vals)

                yw_str = f"{np.mean(yw_vals):.4f} +/- {np.std(yw_vals, ddof=1):.4f}"
                v8_str = f"{np.mean(v8_vals):.4f} +/- {np.std(v8_vals, ddof=1):.4f}"

                if diff > 0:
                    yw_str = f"**{yw_str}**"
                elif diff < 0:
                    v8_str = f"**{v8_str}**"

                lines.append(f"| {class_name} | {yw_str} | {v8_str} | {diff:+.4f} |")

            lines.append("")

    # --- Statistical significance ---
    if sig_results:
        lines.append("## Statistical Significance (Welch's t-test)")
        lines.append("")
        lines.append("H0: No difference between YOLO-World XL and YOLOv8x.")
        lines.append("")
        lines.append("| Metric | t-statistic | p-value | Significance |")
        lines.append("|--------|------------|---------|--------------|")

        for metric in MAIN_METRICS:
            if metric in sig_results:
                r = sig_results[metric]
                lines.append(f"| {METRIC_DISPLAY[metric]} | {r['t_stat']:.3f} | {r['p_value']:.4f} | {r['sig']} |")

        lines.append("")
        lines.append("Significance levels: \\*\\*\\* p<0.001, \\*\\* p<0.01, \\* p<0.05, n.s. not significant")
        lines.append("")

    # --- Per-seed results ---
    lines.append("## Per-Seed Results")
    lines.append("")

    for model in ["yoloworld_xl", "yolov8x"]:
        label = MODEL_LABELS[model]
        model_runs = sorted(
            [r for r in data["runs"] if r["model"] == model],
            key=lambda r: r["seed"],
        )

        if not model_runs:
            continue

        lines.append(f"### {label}")
        lines.append("")
        header = "| Seed | mAP@50-95 | mAP@50 | Precision | Recall | F1 |"
        sep = "|------|-----------|--------|-----------|--------|----|"
        lines.append(header)
        lines.append(sep)

        for r in model_runs:
            m = r[split]
            lines.append(
                f"| {r['seed']} | {m['mAP50-95']:.4f} | {m['mAP50']:.4f} | "
                f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1_score']:.4f} |"
            )

        # Mean +/- std row
        if len(model_runs) > 1:
            vals = {k: [r[split][k] for r in model_runs] for k in MAIN_METRICS}
            lines.append(
                f"| **Mean+/-Std** | **{np.mean(vals['mAP50-95']):.4f}+/-{np.std(vals['mAP50-95'], ddof=1):.4f}** | "
                f"**{np.mean(vals['mAP50']):.4f}+/-{np.std(vals['mAP50'], ddof=1):.4f}** | "
                f"**{np.mean(vals['precision']):.4f}+/-{np.std(vals['precision'], ddof=1):.4f}** | "
                f"**{np.mean(vals['recall']):.4f}+/-{np.std(vals['recall'], ddof=1):.4f}** | "
                f"**{np.mean(vals['f1_score']):.4f}+/-{np.std(vals['f1_score'], ddof=1):.4f}** |"
            )

        lines.append("")

    # --- Validation set results ---
    lines.append("## Validation Set Results")
    lines.append("")
    lines.append("| Metric | YOLO-World XL | YOLOv8x |")
    lines.append("|--------|--------------|---------|")

    for metric in MAIN_METRICS:
        yw_vals = get_metric_arrays(data, "yoloworld_xl", "val", metric)
        v8_vals = get_metric_arrays(data, "yolov8x", "val", metric)

        yw_str = f"{np.mean(yw_vals):.4f} +/- {np.std(yw_vals, ddof=1):.4f}" if len(yw_vals) > 0 else "---"
        v8_str = f"{np.mean(v8_vals):.4f} +/- {np.std(v8_vals, ddof=1):.4f}" if len(v8_vals) > 0 else "---"

        lines.append(f"| {METRIC_DISPLAY[metric]} | {yw_str} | {v8_str} |")

    lines.append("")

    # --- Generated files ---
    lines.append("## Generated Artifacts")
    lines.append("")
    lines.append(f"- `{RESULTS_FILE}` — Raw results (per-seed metrics + aggregate statistics)")
    lines.append(f"- `{FIGURE_PATH}` — Grouped bar chart with error bars")
    lines.append(f"- `{LATEX_PATH}` — LaTeX table for paper inclusion")
    lines.append(f"- `{output_path}` — This summary report")
    lines.append("")

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    print(f"Markdown report saved to: {output_path}")
    return report


# ============================================
# MAIN
# ============================================

def main():
    data = load_results()

    n_runs = len(data.get("runs", []))
    print(f"Loaded {n_runs} runs from {RESULTS_FILE}\n")

    # Console summary
    print_summary(data, split=SPLIT)

    # Also show val set summary
    print_summary(data, split="val")

    # Statistical tests
    sig_results = run_significance_tests(data, split=SPLIT)

    # LaTeX table
    generate_latex_table(data, split=SPLIT)

    # Comparison figure
    generate_comparison_figure(data, split=SPLIT)

    # Per-class figure
    generate_perclass_figure(data, split=SPLIT)

    # Markdown report
    generate_markdown_report(data, sig_results, split=SPLIT)

    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  {FIGURE_PATH}")
    print(f"  {LATEX_PATH}")
    print(f"  {MARKDOWN_PATH}")
    print(f"  {RESULTS_FILE}")


if __name__ == "__main__":
    main()
