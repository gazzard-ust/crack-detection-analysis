<div align="center">

# 🔬 Substrate-Driven Localization Gaps in Pipe Crack Detection

### 🧪 How pipe surface texture limits crack localization under class-balanced training

[![Project Page](https://img.shields.io/badge/project-page-2ea44f.svg)](https://gazzard-ust.github.io/crack-detection-analysis/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ultralytics](https://img.shields.io/badge/ultralytics-8.3-purple.svg)](https://github.com/ultralytics/ultralytics)
[![License: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset](https://img.shields.io/badge/dataset-Roboflow-orange.svg)](https://universe.roboflow.com/gazxard/pipe-crack-detection/dataset/1)

</div>

---

**TL;DR.** Despite class-balanced training (400 labels per class), per-class AP@50-95 ranges from **94.2%** (smooth PVC) to **73.9%** (textured tissue), a **20.2 pp gap** consistent across both YOLO-World XL and YOLOv8x. Decomposing by IoU threshold reveals this is a *localization* problem (23.1 pp AP@50-to-AP@50-95 drop for paper crack vs. 3.9 pp for dummy crack, seed 42). Three augmentation strategies (CutMix, multi-scale, combined) fail to close the gap, indicating a fundamental substrate-driven challenge.

## ❓ Why This Matters

Pipe inspection systems encounter diverse surface conditions in the field. Balanced training data does **not** guarantee balanced per-class performance; substrate visual complexity governs detection difficulty. This has direct implications for practitioners deploying inspection systems on real-world piping infrastructure.

## 📊 Key Results

### 🧱 Substrate-Driven AP Gap

| Class | AP@50 | AP@50-95 | Loc. Gap |
|-------|:-----:|:--------:|:--------:|
| Dummy crack | 99.5% | 95.6% | 3.9 pp |
| PVC pipe crack | 98.6% | 91.2% | 7.4 pp |
| Paper crack | 97.1% | **74.0%** | **23.1 pp** |

<sub>YOLO-World XL, best run (seed 42), test split. Across 5 seeds the per-class AP@50-95 means are 94.2 / 90.8 / 73.9.</sub>

### 🏆 Model Comparison

| Metric | YOLO-World XL | YOLOv8x | *d* |
|--------|:---:|:---:|:---:|
| mAP@50-95 | **0.863** | 0.855 | 1.35 |
| mAP@50 | **0.986** | 0.981 | 1.31 |
| Recall | **0.980** | 0.969 | 1.40 |
| F1 | **0.983** | 0.974 | 1.44 |

<sub>Paired t-test, n=5 seeds. Cohen's *d* > 0.8 = large effect.</sub>

### 🛠️ Mitigation Strategies

| Strategy | mAP@50-95 | Substrate Gap | *p* (gap) |
|----------|:---------:|:------------:|:---------:|
| Baseline | **0.863** | 20.2 pp | &mdash; |
| CutMix | 0.840 | 21.2 pp | 0.562 |
| Multi-Scale | 0.839 | 18.1 pp | 0.334 |
| Combined | 0.805 | 18.4 pp | 0.102 |

<sub>None significant after Bonferroni correction (&alpha; = 0.0167).</sub>

💡 **Takeaway:** The substrate gap persists under all augmentation conditions, reinforcing its origin as a fundamental localization challenge tied to substrate visual properties.

## 🗂️ Dataset

| | Train | Val | Test | Total |
|---|:---:|:---:|:---:|:---:|
| Images | 2,292 | 217 | 108 | 2,617 |

Balanced at **400 labels per class** (1,200 annotations total). Three crack classes on a texture gradient: **Dummy crack** (smooth PVC) &middot; **PVC pipe crack** (smooth PVC, real fractures) &middot; **Paper crack** (porous tissue)

📦 Available on [Roboflow Universe](https://universe.roboflow.com/gazxard/pipe-crack-detection/dataset/1) under CC BY 4.0.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install ultralytics scipy matplotlib

# 2. Download dataset from Roboflow into data/pipe-crack-detection-1/

# 3. Train YOLO-World XL
python code/ylwd_train.py

# 4. Train YOLOv8x baseline
python code/baseline_train.py --seed 42
```

## 🔁 Reproducing the Full Experiment

### 🧪 Baseline (5 seeds x 2 models, ~8h on A100)

```bash
CUDA_VISIBLE_DEVICES=0 python code/multiseed_runner.py --device 0
python code/multiseed_analysis.py
```

### 🩹 Mitigation strategies (5 seeds x 3 strategies, ~12h on A100)

```bash
CUDA_VISIBLE_DEVICES=0 python code/mitigation_runner.py --device 0
python code/mitigation_analysis.py
```

### ⚙️ Training Configuration

All models share identical hyperparameters for fair comparison:

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 2e-4 (cosine decay to 1%) |
| Epochs | 100 (early stopping, patience=20) |
| Image size | 640 (960 for multi-scale training) |
| Augmentation | HSV, flip, mosaic, mixup, rotation, scale, shear |
| Weight decay | 0.05 |

## 📁 Repository Structure

```
.
├── 📄 paper/                             # LaTeX source
│   ├── main.tex                       # Main paper
│   ├── references.bib                 # Bibliography
│   ├── IEEEtran.cls                   # IEEE conference style
│   └── figures/                       # Paper figures
├── 📊 figures/                           # All paper figures
│   ├── fig_samples.png                # Dataset samples
│   ├── fig_qualitative.png            # Detection output examples
│   ├── fig_model_comparison.png       # YOLO-World XL vs YOLOv8x
│   ├── fig_perclass_ap.png            # Localization gap visualization
│   ├── fig_mitigation_comparison.png  # Mitigation strategy results
│   └── fig_training_curves.png        # Training convergence
├── 💻 code/                              # Training, analysis & orchestration
│   ├── ylwd_train.py                  # YOLO-World XL training
│   ├── ylwd_eval.py                   # Model evaluation
│   ├── baseline_train.py              # YOLOv8x baseline training
│   ├── multiseed_runner.py            # Multi-seed experiment runner
│   ├── multiseed_analysis.py          # Baseline analysis & figures
│   ├── mitigation_runner.py           # Mitigation experiment runner
│   ├── mitigation_analysis.py         # Mitigation analysis & figures
│   ├── mitigation_parallel.py         # Parallel multi-GPU mitigation launcher
│   ├── mitigation_worker.py           # Single-GPU mitigation worker
│   ├── mitigation_resume.py           # Resume incomplete mitigation runs
│   ├── dl_dataset.py                  # Dataset download helper
│   ├── generate_fig_*.py              # Figure generation scripts
│   ├── chain_gpu*.sh                  # GPU chaining scripts
│   └── wait_and_merge.sh              # Wait for workers & merge results
├── 📂 data/                              # Results & dataset config
│   ├── multiseed_results.json         # Raw baseline results (10 runs)
│   ├── multiseed_results_summary.md   # Baseline results summary
│   ├── mitigation_results.json        # Mitigation experiment results
│   ├── mitigation_results_summary.md  # Mitigation results summary
│   ├── evaluation_results.json        # Evaluation metrics
│   ├── inference_speed.json           # Latency benchmarks
│   └── pipe-crack-detection-1/        # Dataset configuration
│       └── data.yaml
└── 📋 README.md
```

## 📝 Citation

```bibtex
@inproceedings{biasbas2026substrate,
  title={Substrate-Driven Localization Gaps in Pipe Crack Detection},
  author={Biasbas, Mark Kenneth and Flores, Faustino Miguel and
          Gatchalian, Carl Christian and Velasco, Lorin Angela and
          Yadao, Dulce Maria and Pangaliman, Ma. Madecheen and
          Bautista, Anthony James},
  booktitle={Proc. International Conference on Robotics and Automation Sciences (ICRAS)},
  year={2026}
}
```

## 📄 License

Code: MIT &middot; Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
