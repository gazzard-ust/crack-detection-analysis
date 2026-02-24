# Multi-Seed Baseline Comparison Results

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Dataset | Pipe Crack Detection (3 classes: Dummy crack, PVC pipe crack, Paper crack) |
| Train / Val / Test | 2292 / 217 / 108 images |
| Seeds | [42, 0, 1, 2, 123] |
| Epochs | 100 (early stopping, patience=20) |
| Optimizer | AdamW (lr=2e-4, lrf=0.01, weight_decay=0.05) |
| Image Size | 640 |
| Augmentation | HSV, flip, mosaic, mixup, degrees, scale, shear |
| Evaluation | conf=0.25, iou=0.45 |

| Model | Architecture | Params | Batch Size |
|-------|-------------|--------|------------|
| YOLO-World XL | YOLOv8x-worldv2 (with CLIP text encoder) | ~72.9M | 8 |
| YOLOv8x | Standard YOLOv8x (detection only) | ~68.2M | 16 |

## Model Comparison (Test Set)

| Metric | YOLO-World XL | YOLOv8x | Diff |
|--------|--------------|---------|------|
| mAP@50-95 | **0.8628 +/- 0.0044** | 0.8545 +/- 0.0075 | +0.0083 |
| mAP@50 | **0.9862 +/- 0.0019** | 0.9807 +/- 0.0056 | +0.0055 |
| Precision | **0.9849 +/- 0.0040** | 0.9787 +/- 0.0072 | +0.0062 |
| Recall | **0.9803 +/- 0.0047** | 0.9686 +/- 0.0109 | +0.0117 |
| F1 Score | **0.9826 +/- 0.0040** | 0.9736 +/- 0.0079 | +0.0090 |

### Per-Class AP@50-95

| Class | YOLO-World XL | YOLOv8x | Diff |
|-------|--------------|---------|------|
| Dummy crack | **0.9415 +/- 0.0096** | 0.9396 +/- 0.0084 | +0.0019 |
| PVC pipe crack | **0.9077 +/- 0.0047** | 0.8996 +/- 0.0140 | +0.0081 |
| Paper crack | **0.7393 +/- 0.0067** | 0.7243 +/- 0.0134 | +0.0150 |

## Statistical Significance (Welch's t-test)

H0: No difference between YOLO-World XL and YOLOv8x.

| Metric | t-statistic | p-value | Significance |
|--------|------------|---------|--------------|
| mAP@50-95 | 1.944 | 0.1237 | n.s. |
| mAP@50 | 3.047 | 0.0382 | * |
| Precision | 1.739 | 0.1570 | n.s. |
| Recall | 3.520 | 0.0244 | * |
| F1 Score | 3.497 | 0.0250 | * |

Significance levels: \*\*\* p<0.001, \*\* p<0.01, \* p<0.05, n.s. not significant

## Per-Seed Results

### YOLO-World XL

| Seed | mAP@50-95 | mAP@50 | Precision | Recall | F1 |
|------|-----------|--------|-----------|--------|----|
| 0 | 0.8573 | 0.9873 | 0.9908 | 0.9837 | 0.9873 |
| 1 | 0.8604 | 0.9841 | 0.9833 | 0.9752 | 0.9792 |
| 2 | 0.8635 | 0.9877 | 0.9852 | 0.9837 | 0.9845 |
| 42 | 0.8692 | 0.9841 | 0.9799 | 0.9752 | 0.9775 |
| 123 | 0.8635 | 0.9877 | 0.9852 | 0.9837 | 0.9845 |
| **Mean+/-Std** | **0.8628+/-0.0044** | **0.9862+/-0.0019** | **0.9849+/-0.0040** | **0.9803+/-0.0047** | **0.9826+/-0.0040** |

### YOLOv8x

| Seed | mAP@50-95 | mAP@50 | Precision | Recall | F1 |
|------|-----------|--------|-----------|--------|----|
| 0 | 0.8577 | 0.9848 | 0.9791 | 0.9752 | 0.9771 |
| 1 | 0.8462 | 0.9750 | 0.9662 | 0.9585 | 0.9624 |
| 2 | 0.8569 | 0.9806 | 0.9833 | 0.9666 | 0.9749 |
| 42 | 0.8476 | 0.9753 | 0.9816 | 0.9589 | 0.9701 |
| 123 | 0.8641 | 0.9877 | 0.9833 | 0.9837 | 0.9835 |
| **Mean+/-Std** | **0.8545+/-0.0075** | **0.9807+/-0.0056** | **0.9787+/-0.0072** | **0.9686+/-0.0109** | **0.9736+/-0.0079** |

## Validation Set Results

| Metric | YOLO-World XL | YOLOv8x |
|--------|--------------|---------|
| mAP@50-95 | 0.8571 +/- 0.0025 | 0.8468 +/- 0.0019 |
| mAP@50 | 0.9793 +/- 0.0018 | 0.9790 +/- 0.0046 |
| Precision | 0.9860 +/- 0.0039 | 0.9839 +/- 0.0065 |
| Recall | 0.9665 +/- 0.0046 | 0.9665 +/- 0.0053 |
| F1 Score | 0.9761 +/- 0.0019 | 0.9752 +/- 0.0057 |

## Generated Artifacts

- `multiseed_results.json` — Raw results (per-seed metrics + aggregate statistics)
- `fig_model_comparison.png` — Grouped bar chart with error bars
- `table_model_comparison.tex` — LaTeX table for paper inclusion
- `multiseed_results_summary.md` — This summary report
