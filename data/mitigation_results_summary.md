# Mitigation Strategy Results

## Overview

Three augmentation-based mitigation strategies were evaluated to determine
whether standard training-time augmentation can close the substrate-dependent
AP gap identified in baseline experiments.

| Strategy | Description | Batch Size | Train ImgSz |
|----------|-------------|------------|-------------|
| Baseline | Standard augmentation | 8 | 640 |
| CutMix | cutmix=0.5 | 8 | 640 |
| Multi-Scale | multi_scale=True | 4 | 960 |
| Combined | cutmix=0.5 + multi_scale=True | 4 | 960 |

All evaluations at imgsz=640. 5 seeds per strategy, paired by seed.
Bonferroni-corrected significance threshold: 0.0167

## Strategy Comparison (Test Set)

| Metric | Baseline | CutMix | Multi-Scale | Combined |
|--------|----------|--------|-------------|----------|
| mAP@50-95 | 0.8628 +/- 0.0044 | 0.8399 +/- 0.0218 | 0.8390 +/- 0.0284 | 0.8045 +/- 0.0657 |
| mAP@50 | 0.9862 +/- 0.0019 | 0.9852 +/- 0.0025 | 0.9786 +/- 0.0081 | 0.9570 +/- 0.0506 |
| Precision | 0.9849 +/- 0.0040 | 0.9792 +/- 0.0081 | 0.9718 +/- 0.0111 | 0.9506 +/- 0.0458 |
| Recall | 0.9803 +/- 0.0047 | 0.9759 +/- 0.0048 | 0.9668 +/- 0.0120 | 0.9417 +/- 0.0606 |
| F1 Score | 0.9826 +/- 0.0040 | 0.9776 +/- 0.0060 | 0.9693 +/- 0.0108 | 0.9461 +/- 0.0532 |

### Per-Class AP@50-95

| Class | Baseline | CutMix | Multi-Scale | Combined |
|-------|----------|--------|-------------|----------|
| Dummy crack | 0.9415 +/- 0.0096 | 0.9241 +/- 0.0123 | 0.9048 +/- 0.0488 | 0.8816 +/- 0.0488 |
| Paper crack | 0.7393 +/- 0.0067 | 0.7121 +/- 0.0419 | 0.7240 +/- 0.0146 | 0.6975 +/- 0.0471 |
| PVC pipe crack | 0.9077 +/- 0.0047 | 0.8834 +/- 0.0127 | 0.8882 +/- 0.0280 | 0.8345 +/- 0.1029 |

### Substrate Gap (Dummy - Paper AP@50-95)

| Strategy | Gap (pp) |
|----------|----------|
| Baseline | 20.2 |
| CutMix | 21.2 |
| Multi-Scale | 18.1 |
| Combined | 18.4 |

## Statistical Significance

Paired t-tests (by seed), Bonferroni-corrected alpha = 0.0167

### CutMix vs Baseline

| Metric | Diff | t-stat | p-value | Sig |
|--------|------|--------|---------|-----|
| mAP@50-95 | -0.0229 | -2.276 | 0.0851 | n.s. |
| mAP@50 | -0.0009 | -0.730 | 0.5061 | n.s. |
| Precision | -0.0057 | -1.381 | 0.2394 | n.s. |
| Recall | -0.0044 | -1.571 | 0.1913 | n.s. |
| F1 Score | -0.0050 | -1.542 | 0.1980 | n.s. |

Substrate gap: baseline=20.2 pp → CutMix=21.2 pp (change=+1.0 pp, p=0.5616 n.s.)

### Multi-Scale vs Baseline

| Metric | Diff | t-stat | p-value | Sig |
|--------|------|--------|---------|-----|
| mAP@50-95 | -0.0238 | -1.797 | 0.1467 | n.s. |
| mAP@50 | -0.0075 | -1.946 | 0.1236 | n.s. |
| Precision | -0.0131 | -2.793 | 0.0491 | * |
| Recall | -0.0135 | -2.352 | 0.0783 | n.s. |
| F1 Score | -0.0133 | -2.762 | 0.0508 | n.s. |

Substrate gap: baseline=20.2 pp → Multi-Scale=18.1 pp (change=-2.1 pp, p=0.3341 n.s.)

### Combined vs Baseline

| Metric | Diff | t-stat | p-value | Sig |
|--------|------|--------|---------|-----|
| mAP@50-95 | -0.0583 | -2.009 | 0.1149 | n.s. |
| mAP@50 | -0.0291 | -1.316 | 0.2584 | n.s. |
| Precision | -0.0342 | -1.688 | 0.1667 | n.s. |
| Recall | -0.0386 | -1.493 | 0.2098 | n.s. |
| F1 Score | -0.0365 | -1.584 | 0.1883 | n.s. |

Substrate gap: baseline=20.2 pp → Combined=18.4 pp (change=-1.8 pp, p=0.1016 n.s.)

## Delta Analysis (Change from Baseline)

### CutMix

| Class | Baseline | Strategy | Delta |
|-------|----------|----------|-------|
| Dummy crack | 0.9415 | 0.9241 | -0.0174 |
| Paper crack | 0.7393 | 0.7121 | -0.0272 |
| PVC pipe crack | 0.9077 | 0.8834 | -0.0242 |

### Multi-Scale

| Class | Baseline | Strategy | Delta |
|-------|----------|----------|-------|
| Dummy crack | 0.9415 | 0.9048 | -0.0367 |
| Paper crack | 0.7393 | 0.7240 | -0.0153 |
| PVC pipe crack | 0.9077 | 0.8882 | -0.0195 |

### Combined

| Class | Baseline | Strategy | Delta |
|-------|----------|----------|-------|
| Dummy crack | 0.9415 | 0.8816 | -0.0599 |
| Paper crack | 0.7393 | 0.6975 | -0.0418 |
| PVC pipe crack | 0.9077 | 0.8345 | -0.0732 |

## Generated Artifacts

- `mitigation_results.json` — Raw mitigation results
- `multiseed_results.json` — Raw baseline results
- `fig_mitigation_comparison.png` — Comparison figure
- `table_mitigation.tex` — LaTeX table for paper
- `mitigation_results_summary.md` — This summary report
