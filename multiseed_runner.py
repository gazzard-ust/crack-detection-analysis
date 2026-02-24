"""
Multi-Seed Experiment Runner
============================
Orchestrates multi-seed training + evaluation for both models:
  - YOLO-World XL (5 seeds)
  - YOLOv8x baseline (5 seeds)

Total: 10 training runs, each followed by val + test evaluation.
Runs sequentially to avoid GPU memory conflicts on a single A100.

Outputs:
  - Per-run result JSONs in each run directory
  - multiseed_results.json — aggregate results with mean ± std
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import gc

import numpy as np
import torch
from ultralytics import YOLO, YOLOWorld


# ============================================
# CONFIGURATION
# ============================================

SEEDS = [42, 0, 1, 2, 123]  # 42 first to reuse existing YOLO-World XL run

DATASET_YAML = "pipe-crack-detection-1/data.yaml"
PROJECT_NAME = "crack_detection_runs"

# Existing YOLO-World XL seed=42 run (skip re-training)
EXISTING_YOLOWORLD_SEED42 = "crack_detection_runs/yoloworld_xl_20251226_171626"

# Evaluation settings (match ylwd_eval.py)
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Training batch sizes — YOLO-World XL uses more VRAM due to CLIP encoder,
# so it needs a smaller batch. Ultralytics' nbs=64 auto-scales the effective
# learning rate, so the optimisation dynamics stay matched.
BATCH_SIZE_YOLOWORLD = 8
BATCH_SIZE_YOLOV8X = 16
BATCH_SIZE_EVAL = 16


# ============================================
# GPU MEMORY CLEANUP
# ============================================

def clear_gpu():
    """Free GPU memory between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================
# EVALUATION HELPER
# ============================================

def evaluate_model(model_path, split, device, is_yoloworld=False):
    """Evaluate a trained model on the given split. Returns metrics dict."""

    print(f"\n  Evaluating on {split} set: {model_path}")

    if is_yoloworld:
        model = YOLOWorld(model_path)
    else:
        model = YOLO(model_path)

    results = model.val(
        data=DATASET_YAML,
        split=split,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE_EVAL,
        device=device,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        plots=False,
        save_json=False,
        verbose=False,
    )

    metrics = {
        "split": split,
        "mAP50-95": float(results.box.map),
        "mAP50": float(results.box.map50),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "f1_score": float(
            2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-6)
        ),
    }

    # Per-class metrics
    class_names = list(results.names.values())
    per_class_ap50 = results.box.ap50.tolist()
    per_class_ap = results.box.ap.tolist()

    for i, class_name in enumerate(class_names):
        metrics[f"AP50-95_{class_name}"] = float(per_class_ap[i])
        metrics[f"AP50_{class_name}"] = float(per_class_ap50[i])

    return metrics


# ============================================
# YOLO-WORLD XL TRAINING
# ============================================

def train_yoloworld(seed, device, epochs=100):
    """Train YOLO-World XL with the given seed. Returns the run directory name."""

    run_name = f"yoloworld_xl_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'=' * 60}")
    print(f"TRAINING YOLO-World XL — seed={seed}, device={device}")
    print(f"{'=' * 60}")

    model = YOLOWorld("yolov8x-worldv2.pt")
    model.set_classes(["Dummy crack", "Paper crack", "PVC pipe crack"])

    model.train(
        data=DATASET_YAML,
        epochs=epochs,
        patience=20,
        batch=BATCH_SIZE_YOLOWORLD,
        imgsz=IMG_SIZE,
        lr0=2e-4,
        lrf=0.01,
        warmup_epochs=3,
        warmup_momentum=0.8,
        optimizer="AdamW",
        weight_decay=0.05,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        device=device,
        workers=8,
        project=PROJECT_NAME,
        name=run_name,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
        seed=seed,
    )

    return run_name


def train_yolov8x(seed, device, epochs=100):
    """Train YOLOv8x baseline with the given seed. Returns the run directory name."""

    run_name = f"yolov8x_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'=' * 60}")
    print(f"TRAINING YOLOv8x BASELINE — seed={seed}, device={device}")
    print(f"{'=' * 60}")

    model = YOLO("yolov8x.pt")

    model.train(
        data=DATASET_YAML,
        epochs=epochs,
        patience=20,
        batch=BATCH_SIZE_YOLOV8X,
        imgsz=IMG_SIZE,
        lr0=2e-4,
        lrf=0.01,
        warmup_epochs=3,
        warmup_momentum=0.8,
        optimizer="AdamW",
        weight_decay=0.05,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        device=device,
        workers=8,
        project=PROJECT_NAME,
        name=run_name,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
        seed=seed,
    )

    return run_name


# ============================================
# AGGREGATE STATISTICS
# ============================================

def compute_aggregate(all_results):
    """Compute mean +/- std across seeds for each model and split."""

    aggregate = {}

    for model_name in ["yoloworld_xl", "yolov8x"]:
        aggregate[model_name] = {}
        model_runs = [r for r in all_results if r["model"] == model_name]

        for split in ["val", "test"]:
            split_runs = [r[split] for r in model_runs if split in r]
            if not split_runs:
                continue

            # Collect all numeric metric keys
            metric_keys = [k for k in split_runs[0] if isinstance(split_runs[0][k], (int, float))]

            stats = {}
            for key in metric_keys:
                values = [run[key] for run in split_runs]
                stats[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "values": values,
                }

            aggregate[model_name][split] = stats

    return aggregate


# ============================================
# MAIN RUNNER
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Multi-seed experiment runner")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--device", type=int, default=0, help="GPU device index (default: 0)")
    parser.add_argument("--skip-yoloworld", action="store_true", help="Skip YOLO-World XL runs")
    parser.add_argument("--skip-yolov8x", action="store_true", help="Skip YOLOv8x baseline runs")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing runs (no training)")
    args = parser.parse_args()

    device = args.device
    all_results = []
    output_file = "multiseed_results.json"

    print(f"Using GPU device: {device}")
    if torch.cuda.is_available():
        print(f"  {torch.cuda.get_device_name(device)} — "
              f"{torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB total")

    # Load any existing results to allow resuming
    if os.path.exists(output_file):
        with open(output_file) as f:
            saved = json.load(f)
            all_results = saved.get("runs", [])
        print(f"Loaded {len(all_results)} existing results from {output_file}")

    # Helper to check if a run already exists
    def run_exists(model, seed):
        return any(r["model"] == model and r["seed"] == seed for r in all_results)

    # ── YOLO-World XL runs ──────────────────────────────────────
    if not args.skip_yoloworld:
        for seed in SEEDS:
            if run_exists("yoloworld_xl", seed):
                print(f"\n[SKIP] YOLO-World XL seed={seed} already in results")
                continue

            # Reuse existing seed=42 run
            if seed == 42 and os.path.exists(EXISTING_YOLOWORLD_SEED42):
                run_dir = EXISTING_YOLOWORLD_SEED42
                print(f"\n[REUSE] YOLO-World XL seed=42 from {run_dir}")
            elif not args.eval_only:
                run_name = train_yoloworld(seed, device=device, epochs=args.epochs)
                clear_gpu()
                run_dir = f"{PROJECT_NAME}/{run_name}"
            else:
                print(f"\n[SKIP] YOLO-World XL seed={seed} — no trained model found and --eval-only set")
                continue

            # Evaluate
            best_weights = f"{run_dir}/weights/best.pt"
            if not os.path.exists(best_weights):
                print(f"  WARNING: {best_weights} not found, skipping evaluation")
                continue

            val_metrics = evaluate_model(best_weights, "val", device=device, is_yoloworld=True)
            test_metrics = evaluate_model(best_weights, "test", device=device, is_yoloworld=True)
            clear_gpu()

            result = {
                "model": "yoloworld_xl",
                "seed": seed,
                "run_dir": run_dir,
                "val": val_metrics,
                "test": test_metrics,
                "timestamp": datetime.now().isoformat(),
            }
            all_results.append(result)

            # Save incrementally
            _save_results(all_results, output_file)
            print(f"  Saved results for YOLO-World XL seed={seed}")

    # ── YOLOv8x baseline runs ──────────────────────────────────
    if not args.skip_yolov8x:
        for seed in SEEDS:
            if run_exists("yolov8x", seed):
                print(f"\n[SKIP] YOLOv8x seed={seed} already in results")
                continue

            if not args.eval_only:
                run_name = train_yolov8x(seed, device=device, epochs=args.epochs)
                clear_gpu()
                run_dir = f"{PROJECT_NAME}/{run_name}"
            else:
                print(f"\n[SKIP] YOLOv8x seed={seed} — no trained model and --eval-only set")
                continue

            # Evaluate
            best_weights = f"{run_dir}/weights/best.pt"
            if not os.path.exists(best_weights):
                print(f"  WARNING: {best_weights} not found, skipping evaluation")
                continue

            val_metrics = evaluate_model(best_weights, "val", device=device, is_yoloworld=False)
            test_metrics = evaluate_model(best_weights, "test", device=device, is_yoloworld=False)
            clear_gpu()

            result = {
                "model": "yolov8x",
                "seed": seed,
                "run_dir": run_dir,
                "val": val_metrics,
                "test": test_metrics,
                "timestamp": datetime.now().isoformat(),
            }
            all_results.append(result)

            _save_results(all_results, output_file)
            print(f"  Saved results for YOLOv8x seed={seed}")

    # ── Final aggregate ────────────────────────────────────────
    _save_results(all_results, output_file)

    print("\n" + "=" * 60)
    print("ALL RUNS COMPLETE")
    print("=" * 60)
    _print_summary(all_results)


def _save_results(all_results, output_file):
    """Save results with aggregate statistics."""
    aggregate = compute_aggregate(all_results)
    output = {
        "runs": all_results,
        "aggregate": aggregate,
        "seeds": SEEDS,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)


def _print_summary(all_results):
    """Print a summary table of all results."""
    aggregate = compute_aggregate(all_results)

    for model_name, splits in aggregate.items():
        print(f"\n{'─' * 50}")
        print(f"  {model_name}")
        print(f"{'─' * 50}")
        for split, stats in splits.items():
            print(f"\n  {split.upper()} set:")
            for metric in ["mAP50-95", "mAP50", "precision", "recall", "f1_score"]:
                if metric in stats:
                    m = stats[metric]["mean"]
                    s = stats[metric]["std"]
                    print(f"    {metric:<15} {m:.4f} +/- {s:.4f}")


if __name__ == "__main__":
    main()
