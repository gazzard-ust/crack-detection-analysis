"""
Mitigation Strategy Results Analysis & Figure Generation
=========================================================
Reads multiseed_results.json (baseline) + mitigation_results.json and produces:
  1. Console summary comparing baseline vs mitigation strategies
  2. LaTeX table: baseline vs 3 strategies (table_mitigation.tex)
  3. fig_mitigation_comparison.png — grouped bar chart
  4. Statistical significance tests (paired t-test with Bonferroni correction)
  5. mitigation_results_summary.md — full report
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
import numpy as np
from scipy import stats as scipy_stats


# ============================================
# CONFIGURATION
# ============================================

BASELINE_FILE = "multiseed_results.json"
MITIGATION_FILE = "mitigation_results.json"
FIGURE_PATH = "fig_mitigation_comparison.png"
LATEX_PATH = "table_mitigation.tex"
MARKDOWN_PATH = "mitigation_results_summary.md"
SPLIT = "test"

STRATEGY_LABELS = {
    "baseline": "Baseline",
    "cutmix": "CutMix",
    "multi_scale": "Multi-Scale",
    "combined": "Combined",
}

STRATEGY_ORDER = ["baseline", "cutmix", "multi_scale", "combined"]

MAIN_METRICS = ["mAP50-95", "mAP50", "precision", "recall", "f1_score"]

METRIC_DISPLAY = {
    "mAP50-95": "mAP@50-95",
    "mAP50": "mAP@50",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1 Score",
}

CLASS_NAMES = ["Dummy crack", "Paper crack", "PVC pipe crack"]

# Bonferroni correction: alpha=0.05 / 3 strategies
BONFERRONI_ALPHA = 0.05 / 3


# ============================================
# DATA LOADING
# ============================================

def load_all_data():
    """Load baseline and mitigation results, returning a unified data structure."""

    if not os.path.exists(BASELINE_FILE):
        print(f"ERROR: {BASELINE_FILE} not found. Run multiseed_runner.py first.")
        sys.exit(1)

    if not os.path.exists(MITIGATION_FILE):
        print(f"ERROR: {MITIGATION_FILE} not found. Run mitigation_runner.py first.")
        sys.exit(1)

    with open(BASELINE_FILE) as f:
        baseline_data = json.load(f)

    with open(MITIGATION_FILE) as f:
        mitigation_data = json.load(f)

    # Extract YOLO-World XL baseline runs and tag them
    baseline_runs = []
    for r in baseline_data["runs"]:
        if r["model"] == "yoloworld_xl":
            run = dict(r)
            run["strategy"] = "baseline"
            baseline_runs.append(run)

    mitigation_runs = mitigation_data["runs"]

    all_runs = baseline_runs + mitigation_runs

    return {
        "runs": all_runs,
        "baseline_data": baseline_data,
        "mitigation_data": mitigation_data,
    }


def get_metric_arrays(runs, strategy, split, metric):
    """Extract metric values across seeds for a given strategy/split."""
    strategy_runs = [r for r in runs if r["strategy"] == strategy]
    values = [r[split][metric] for r in strategy_runs if split in r and metric in r[split]]
    return np.array(values)


def get_seed_paired_arrays(runs, strategy, split, metric):
    """Extract metric values paired by seed for baseline vs strategy comparison."""
    baseline_runs = sorted(
        [r for r in runs if r["strategy"] == "baseline" and split in r],
        key=lambda r: r["seed"],
    )
    strategy_runs = sorted(
        [r for r in runs if r["strategy"] == strategy and split in r],
        key=lambda r: r["seed"],
    )

    # Match by seed
    baseline_seeds = {r["seed"]: r[split][metric] for r in baseline_runs if metric in r[split]}
    strategy_seeds = {r["seed"]: r[split][metric] for r in strategy_runs if metric in r[split]}

    common_seeds = sorted(set(baseline_seeds.keys()) & set(strategy_seeds.keys()))

    baseline_vals = np.array([baseline_seeds[s] for s in common_seeds])
    strategy_vals = np.array([strategy_seeds[s] for s in common_seeds])

    return baseline_vals, strategy_vals, common_seeds


# ============================================
# CONSOLE SUMMARY
# ============================================

def print_summary(data, split=SPLIT):
    runs = data["runs"]

    print("=" * 70)
    print(f"MITIGATION STRATEGY RESULTS SUMMARY  ({split.upper()} set)")
    print("=" * 70)

    for strategy in STRATEGY_ORDER:
        label = STRATEGY_LABELS.get(strategy, strategy)
        vals = get_metric_arrays(runs, strategy, split, "mAP50-95")
        if len(vals) == 0:
            continue

        print(f"\n  {label} (n={len(vals)})")
        print(f"  {'─' * 45}")

        for metric in MAIN_METRICS:
            vals = get_metric_arrays(runs, strategy, split, metric)
            if len(vals) == 0:
                continue
            print(f"    {METRIC_DISPLAY[metric]:<15} {np.mean(vals):.4f} +/- {np.std(vals, ddof=1):.4f}")

        # Per-class AP
        strategy_runs = [r for r in runs if r["strategy"] == strategy and split in r]
        if strategy_runs:
            print(f"\n    Per-class AP@50-95:")
            for cls in CLASS_NAMES:
                key = f"AP50-95_{cls}"
                cls_vals = np.array([r[split][key] for r in strategy_runs if key in r[split]])
                if len(cls_vals) > 0:
                    print(f"      {cls:<20} {np.mean(cls_vals):.4f} +/- {np.std(cls_vals, ddof=1):.4f}")

            # Substrate gap
            dummy_key = "AP50-95_Dummy crack"
            paper_key = "AP50-95_Paper crack"
            dummy_vals = np.array([r[split][dummy_key] for r in strategy_runs if dummy_key in r[split]])
            paper_vals = np.array([r[split][paper_key] for r in strategy_runs if paper_key in r[split]])
            if len(dummy_vals) > 0 and len(paper_vals) > 0:
                gap = np.mean(dummy_vals) - np.mean(paper_vals)
                print(f"\n    Substrate gap (Dummy - Paper): {gap:.4f} ({gap*100:.1f} pp)")

    print()


# ============================================
# STATISTICAL SIGNIFICANCE WITH BONFERRONI
# ============================================

def run_significance_tests(data, split=SPLIT):
    """Paired t-tests (by seed) for each mitigation strategy vs baseline.
    Uses Bonferroni correction: alpha = 0.05/3."""

    runs = data["runs"]

    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE (Paired t-test, Bonferroni-corrected)")
    print("=" * 70)
    print(f"  Bonferroni-corrected alpha: {BONFERRONI_ALPHA:.4f} (0.05 / 3 strategies)")
    print(f"  Split: {split.upper()}\n")

    all_results = {}

    for strategy in ["cutmix", "multi_scale", "combined"]:
        label = STRATEGY_LABELS[strategy]
        print(f"\n  {label} vs Baseline:")
        print(f"  {'─' * 55}")

        strategy_results = {}

        for metric in MAIN_METRICS:
            b_vals, s_vals, seeds = get_seed_paired_arrays(runs, strategy, split, metric)

            if len(b_vals) < 2 or len(s_vals) < 2:
                print(f"    {METRIC_DISPLAY[metric]:<15} — insufficient data")
                continue

            t_stat, p_value = scipy_stats.ttest_rel(s_vals, b_vals)
            diff = np.mean(s_vals) - np.mean(b_vals)

            sig = "***" if p_value < 0.001 else "**" if p_value < BONFERRONI_ALPHA else "*" if p_value < 0.05 else "n.s."
            bonf_sig = p_value < BONFERRONI_ALPHA

            strategy_results[metric] = {
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "diff": float(diff),
                "sig": sig,
                "bonferroni_significant": bonf_sig,
                "n_seeds": len(seeds),
            }

            bonf_marker = " (Bonf.)" if bonf_sig else ""
            print(f"    {METRIC_DISPLAY[metric]:<15}  diff={diff:+.4f}  t={t_stat:.3f}  p={p_value:.4f}  {sig}{bonf_marker}")

        # Per-class AP paired tests
        print(f"\n    Per-class AP@50-95:")
        for cls in CLASS_NAMES:
            key = f"AP50-95_{cls}"
            b_vals, s_vals, seeds = get_seed_paired_arrays(runs, strategy, split, key)
            if len(b_vals) < 2:
                continue
            t_stat, p_value = scipy_stats.ttest_rel(s_vals, b_vals)
            diff = np.mean(s_vals) - np.mean(b_vals)
            sig = "***" if p_value < 0.001 else "**" if p_value < BONFERRONI_ALPHA else "*" if p_value < 0.05 else "n.s."

            strategy_results[key] = {
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "diff": float(diff),
                "sig": sig,
            }
            print(f"      {cls:<20}  diff={diff:+.4f}  t={t_stat:.3f}  p={p_value:.4f}  {sig}")

        # Substrate gap test
        dummy_key = "AP50-95_Dummy crack"
        paper_key = "AP50-95_Paper crack"

        b_dummy, s_dummy, _ = get_seed_paired_arrays(runs, strategy, split, dummy_key)
        b_paper, s_paper, _ = get_seed_paired_arrays(runs, strategy, split, paper_key)

        if len(b_dummy) >= 2 and len(b_paper) >= 2:
            b_gap = b_dummy - b_paper
            s_gap = s_dummy - s_paper
            t_stat, p_value = scipy_stats.ttest_rel(s_gap, b_gap)
            diff = np.mean(s_gap) - np.mean(b_gap)
            sig = "***" if p_value < 0.001 else "**" if p_value < BONFERRONI_ALPHA else "*" if p_value < 0.05 else "n.s."

            strategy_results["substrate_gap"] = {
                "baseline_gap_mean": float(np.mean(b_gap)),
                "strategy_gap_mean": float(np.mean(s_gap)),
                "gap_change": float(diff),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "sig": sig,
            }
            print(f"\n    Substrate gap change:  {diff:+.4f}  t={t_stat:.3f}  p={p_value:.4f}  {sig}")
            print(f"      Baseline gap: {np.mean(b_gap)*100:.1f} pp  →  {STRATEGY_LABELS[strategy]} gap: {np.mean(s_gap)*100:.1f} pp")

        all_results[strategy] = strategy_results

    print(f"\n  Significance: *** p<0.001, ** p<{BONFERRONI_ALPHA:.4f} (Bonferroni), * p<0.05 (uncorrected), n.s. not significant\n")
    return all_results


# ============================================
# DELTA ANALYSIS
# ============================================

def print_delta_analysis(data, split=SPLIT):
    """Per-class AP change from baseline for each strategy."""

    runs = data["runs"]

    print("=" * 70)
    print("DELTA ANALYSIS: Per-Class AP Change from Baseline")
    print("=" * 70)

    for strategy in ["cutmix", "multi_scale", "combined"]:
        label = STRATEGY_LABELS[strategy]
        print(f"\n  {label}:")

        for cls in CLASS_NAMES:
            key = f"AP50-95_{cls}"
            b_vals, s_vals, _ = get_seed_paired_arrays(runs, strategy, split, key)
            if len(b_vals) == 0:
                continue
            delta = np.mean(s_vals) - np.mean(b_vals)
            print(f"    {cls:<20}  baseline={np.mean(b_vals):.4f}  →  {np.mean(s_vals):.4f}  (delta={delta:+.4f}, {delta*100:+.1f} pp)")

        # Localization gap decomposition
        print(f"\n    Localization gap (AP@50 - AP@50-95):")
        for cls in CLASS_NAMES:
            ap50_key = f"AP50_{cls}"
            ap5095_key = f"AP50-95_{cls}"

            strategy_runs = [r for r in runs if r["strategy"] == strategy and split in r]
            baseline_runs_s = [r for r in runs if r["strategy"] == "baseline" and split in r]

            if strategy_runs and baseline_runs_s:
                s_ap50 = np.mean([r[split][ap50_key] for r in strategy_runs if ap50_key in r[split]])
                s_ap5095 = np.mean([r[split][ap5095_key] for r in strategy_runs if ap5095_key in r[split]])
                b_ap50 = np.mean([r[split][ap50_key] for r in baseline_runs_s if ap50_key in r[split]])
                b_ap5095 = np.mean([r[split][ap5095_key] for r in baseline_runs_s if ap5095_key in r[split]])

                s_gap = s_ap50 - s_ap5095
                b_gap = b_ap50 - b_ap5095

                print(f"      {cls:<20}  baseline={b_gap*100:.1f} pp  →  {s_gap*100:.1f} pp  (change={((s_gap - b_gap)*100):+.1f} pp)")

    print()


# ============================================
# LATEX TABLE
# ============================================

def generate_latex_table(data, split=SPLIT, output_path=LATEX_PATH):
    runs = data["runs"]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Mitigation strategy evaluation on the test set (mean $\pm$ std, 5 seeds). Strategies applied to YOLO-World XL only.}")
    lines.append(r"\label{tab:mitigation}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Metric & Baseline & CutMix & Multi-Scale & Combined \\")
    lines.append(r"\midrule")

    for metric in MAIN_METRICS:
        row_vals = []
        row_means = []

        for strategy in STRATEGY_ORDER:
            vals = get_metric_arrays(runs, strategy, split, metric)
            if len(vals) > 0:
                mean = np.mean(vals)
                std = np.std(vals, ddof=1)
                row_vals.append(f"${mean:.4f} \\pm {std:.4f}$")
                row_means.append(mean)
            else:
                row_vals.append("---")
                row_means.append(-1)

        # Bold the best result
        if all(m >= 0 for m in row_means):
            best_idx = np.argmax(row_means)
            row_vals[best_idx] = r"\textbf{" + row_vals[best_idx] + "}"

        display = METRIC_DISPLAY[metric]
        lines.append(f"{display} & {' & '.join(row_vals)} \\\\")

    # Per-class AP
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{l}{\textit{Per-class AP@50-95}} \\")
    lines.append(r"\midrule")

    for cls in CLASS_NAMES:
        key = f"AP50-95_{cls}"
        row_vals = []
        row_means = []

        for strategy in STRATEGY_ORDER:
            vals = get_metric_arrays(runs, strategy, split, key)
            if len(vals) > 0:
                mean = np.mean(vals)
                std = np.std(vals, ddof=1)
                row_vals.append(f"${mean:.4f} \\pm {std:.4f}$")
                row_means.append(mean)
            else:
                row_vals.append("---")
                row_means.append(-1)

        if all(m >= 0 for m in row_means):
            best_idx = np.argmax(row_means)
            row_vals[best_idx] = r"\textbf{" + row_vals[best_idx] + "}"

        lines.append(f"{cls} & {' & '.join(row_vals)} \\\\")

    # Substrate gap row
    lines.append(r"\midrule")
    gap_vals = []
    for strategy in STRATEGY_ORDER:
        dummy_vals = get_metric_arrays(runs, strategy, split, "AP50-95_Dummy crack")
        paper_vals = get_metric_arrays(runs, strategy, split, "AP50-95_Paper crack")
        if len(dummy_vals) > 0 and len(paper_vals) > 0:
            gap = np.mean(dummy_vals) - np.mean(paper_vals)
            gap_vals.append(f"${gap*100:.1f}$ pp")
        else:
            gap_vals.append("---")

    lines.append(f"Substrate gap & {' & '.join(gap_vals)} \\\\")

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
# FIGURE: GROUPED BAR CHART
# ============================================

def generate_comparison_figure(data, split=SPLIT, output_path=FIGURE_PATH):
    """Grouped bar chart showing per-class AP and substrate gap for each strategy."""

    runs = data["runs"]

    # Collect per-class AP means and stds for each strategy
    strategies_present = [s for s in STRATEGY_ORDER if any(r["strategy"] == s for r in runs)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1]})

    # --- Left panel: Per-class AP@50-95 grouped bar chart ---
    ax = axes[0]
    x = np.arange(len(CLASS_NAMES))
    n_strategies = len(strategies_present)
    width = 0.8 / n_strategies

    colors = ["#607D8B", "#2196F3", "#4CAF50", "#FF9800"]

    for i, strategy in enumerate(strategies_present):
        means = []
        stds = []
        for cls in CLASS_NAMES:
            key = f"AP50-95_{cls}"
            vals = get_metric_arrays(runs, strategy, split, key)
            if len(vals) > 0:
                means.append(np.mean(vals))
                stds.append(np.std(vals, ddof=1))
            else:
                means.append(0)
                stds.append(0)

        offset = (i - (n_strategies - 1) / 2) * width
        bars = ax.bar(
            x + offset, means, width, yerr=stds,
            label=STRATEGY_LABELS.get(strategy, strategy),
            color=colors[i % len(colors)],
            edgecolor="white", linewidth=0.5,
            capsize=3, error_kw={"linewidth": 1},
        )

        # Value labels on top
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=7, rotation=45,
            )

    ax.set_ylabel("AP@50-95", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_title("Per-Class AP@50-95 by Strategy", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0.6, 1.05)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # --- Right panel: Substrate gap ---
    ax2 = axes[1]
    gap_means = []
    gap_labels = []

    for strategy in strategies_present:
        dummy_vals = get_metric_arrays(runs, strategy, split, "AP50-95_Dummy crack")
        paper_vals = get_metric_arrays(runs, strategy, split, "AP50-95_Paper crack")
        if len(dummy_vals) > 0 and len(paper_vals) > 0:
            gap = np.mean(dummy_vals) - np.mean(paper_vals)
            gap_means.append(gap * 100)
            gap_labels.append(STRATEGY_LABELS.get(strategy, strategy))

    bar_colors = [colors[STRATEGY_ORDER.index(s)] for s in strategies_present if any(
        r["strategy"] == s for r in runs
    )]

    bars = ax2.barh(
        range(len(gap_means)), gap_means,
        color=bar_colors[:len(gap_means)],
        edgecolor="white", linewidth=0.5,
    )

    for bar, val in zip(bars, gap_means):
        ax2.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} pp", ha="left", va="center", fontsize=9,
        )

    ax2.set_yticks(range(len(gap_labels)))
    ax2.set_yticklabels(gap_labels, fontsize=10)
    ax2.set_xlabel("Substrate Gap (pp)", fontsize=11)
    ax2.set_title("Dummy-Paper AP Gap", fontsize=12, fontweight="bold")
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure saved to: {output_path}")


# ============================================
# MARKDOWN REPORT
# ============================================

def generate_markdown_report(data, sig_results, split=SPLIT, output_path=MARKDOWN_PATH):
    runs = data["runs"]
    lines = []

    lines.append("# Mitigation Strategy Results")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("Three augmentation-based mitigation strategies were evaluated to determine")
    lines.append("whether standard training-time augmentation can close the substrate-dependent")
    lines.append("AP gap identified in baseline experiments.")
    lines.append("")
    lines.append("| Strategy | Description | Batch Size | Train ImgSz |")
    lines.append("|----------|-------------|------------|-------------|")
    lines.append("| Baseline | Standard augmentation | 8 | 640 |")
    lines.append("| CutMix | cutmix=0.5 | 8 | 640 |")
    lines.append("| Multi-Scale | multi_scale=True | 4 | 960 |")
    lines.append("| Combined | cutmix=0.5 + multi_scale=True | 4 | 960 |")
    lines.append("")
    lines.append("All evaluations at imgsz=640. 5 seeds per strategy, paired by seed.")
    lines.append(f"Bonferroni-corrected significance threshold: {BONFERRONI_ALPHA:.4f}")
    lines.append("")

    # --- Main comparison table ---
    lines.append(f"## Strategy Comparison ({split.capitalize()} Set)")
    lines.append("")
    lines.append("| Metric | Baseline | CutMix | Multi-Scale | Combined |")
    lines.append("|--------|----------|--------|-------------|----------|")

    for metric in MAIN_METRICS:
        row = [METRIC_DISPLAY[metric]]
        for strategy in STRATEGY_ORDER:
            vals = get_metric_arrays(runs, strategy, split, metric)
            if len(vals) > 0:
                row.append(f"{np.mean(vals):.4f} +/- {np.std(vals, ddof=1):.4f}")
            else:
                row.append("---")
        lines.append(f"| {' | '.join(row)} |")

    lines.append("")

    # --- Per-class AP ---
    lines.append("### Per-Class AP@50-95")
    lines.append("")
    lines.append("| Class | Baseline | CutMix | Multi-Scale | Combined |")
    lines.append("|-------|----------|--------|-------------|----------|")

    for cls in CLASS_NAMES:
        key = f"AP50-95_{cls}"
        row = [cls]
        for strategy in STRATEGY_ORDER:
            vals = get_metric_arrays(runs, strategy, split, key)
            if len(vals) > 0:
                row.append(f"{np.mean(vals):.4f} +/- {np.std(vals, ddof=1):.4f}")
            else:
                row.append("---")
        lines.append(f"| {' | '.join(row)} |")

    lines.append("")

    # --- Substrate gap ---
    lines.append("### Substrate Gap (Dummy - Paper AP@50-95)")
    lines.append("")
    lines.append("| Strategy | Gap (pp) |")
    lines.append("|----------|----------|")

    for strategy in STRATEGY_ORDER:
        dummy_vals = get_metric_arrays(runs, strategy, split, "AP50-95_Dummy crack")
        paper_vals = get_metric_arrays(runs, strategy, split, "AP50-95_Paper crack")
        if len(dummy_vals) > 0 and len(paper_vals) > 0:
            gap = np.mean(dummy_vals) - np.mean(paper_vals)
            lines.append(f"| {STRATEGY_LABELS.get(strategy, strategy)} | {gap*100:.1f} |")

    lines.append("")

    # --- Statistical significance ---
    if sig_results:
        lines.append("## Statistical Significance")
        lines.append("")
        lines.append(f"Paired t-tests (by seed), Bonferroni-corrected alpha = {BONFERRONI_ALPHA:.4f}")
        lines.append("")

        for strategy in ["cutmix", "multi_scale", "combined"]:
            if strategy not in sig_results:
                continue

            label = STRATEGY_LABELS[strategy]
            lines.append(f"### {label} vs Baseline")
            lines.append("")
            lines.append("| Metric | Diff | t-stat | p-value | Sig |")
            lines.append("|--------|------|--------|---------|-----|")

            for metric in MAIN_METRICS:
                if metric in sig_results[strategy]:
                    r = sig_results[strategy][metric]
                    lines.append(
                        f"| {METRIC_DISPLAY[metric]} | {r['diff']:+.4f} | "
                        f"{r['t_stat']:.3f} | {r['p_value']:.4f} | {r['sig']} |"
                    )

            lines.append("")

            # Substrate gap
            if "substrate_gap" in sig_results[strategy]:
                sg = sig_results[strategy]["substrate_gap"]
                lines.append(
                    f"Substrate gap: baseline={sg['baseline_gap_mean']*100:.1f} pp → "
                    f"{label}={sg['strategy_gap_mean']*100:.1f} pp "
                    f"(change={sg['gap_change']*100:+.1f} pp, p={sg['p_value']:.4f} {sg['sig']})"
                )
                lines.append("")

    # --- Delta analysis ---
    lines.append("## Delta Analysis (Change from Baseline)")
    lines.append("")

    for strategy in ["cutmix", "multi_scale", "combined"]:
        label = STRATEGY_LABELS[strategy]
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| Class | Baseline | Strategy | Delta |")
        lines.append("|-------|----------|----------|-------|")

        for cls in CLASS_NAMES:
            key = f"AP50-95_{cls}"
            b_vals, s_vals, _ = get_seed_paired_arrays(runs, strategy, split, key)
            if len(b_vals) > 0:
                lines.append(
                    f"| {cls} | {np.mean(b_vals):.4f} | {np.mean(s_vals):.4f} | "
                    f"{np.mean(s_vals) - np.mean(b_vals):+.4f} |"
                )

        lines.append("")

    # --- Generated artifacts ---
    lines.append("## Generated Artifacts")
    lines.append("")
    lines.append(f"- `{MITIGATION_FILE}` — Raw mitigation results")
    lines.append(f"- `{BASELINE_FILE}` — Raw baseline results")
    lines.append(f"- `{FIGURE_PATH}` — Comparison figure")
    lines.append(f"- `{LATEX_PATH}` — LaTeX table for paper")
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
    data = load_all_data()

    n_runs = len(data["runs"])
    n_baseline = sum(1 for r in data["runs"] if r["strategy"] == "baseline")
    n_mitigation = n_runs - n_baseline
    print(f"Loaded {n_runs} total runs ({n_baseline} baseline + {n_mitigation} mitigation)\n")

    # Console summary
    print_summary(data, split=SPLIT)

    # Statistical tests
    sig_results = run_significance_tests(data, split=SPLIT)

    # Delta analysis
    print_delta_analysis(data, split=SPLIT)

    # LaTeX table
    generate_latex_table(data, split=SPLIT)

    # Figure
    generate_comparison_figure(data, split=SPLIT)

    # Markdown report
    generate_markdown_report(data, sig_results, split=SPLIT)

    print("=" * 70)
    print("MITIGATION ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  {FIGURE_PATH}")
    print(f"  {LATEX_PATH}")
    print(f"  {MARKDOWN_PATH}")


if __name__ == "__main__":
    main()
