"""
Mitigation Strategy Experiment Runner
======================================
Orchestrates augmentation-based mitigation experiments for YOLO-World XL
to evaluate whether standard training-time augmentation can close the
substrate-dependent AP gap identified in baseline experiments.

Three strategies (YOLO-World XL only, 5 seeds each = 15 runs total):
  1. CutMix       — cutmix=0.5
  2. Multi-Scale   — multi_scale=True (imgsz=960 during training)
  3. Combined      — cutmix=0.5 + multi_scale=True

All other hyperparameters match the baseline (multiseed_runner.py).
Evaluation always at imgsz=640 for fair comparison.

Outputs:
  - Per-run result JSONs in each run directory
  - mitigation_results.json — aggregate results with strategy labels
"""

import argparse
import json
import os
import sys
from datetime import datetime

import gc

import numpy as np
import torch
from ultralytics import YOLOWorld


# ============================================
# CONFIGURATION
# ============================================

SEEDS = [42, 0, 1, 2, 123]

DATASET_YAML = "pipe-crack-detection-1/data.yaml"
PROJECT_NAME = "crack_detection_runs"

# Evaluation settings (match baseline)
IMG_SIZE_EVAL = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
BATCH_SIZE_EVAL = 16

# Strategy definitions
# Each strategy specifies its training overrides relative to baseline.
# Batch sizes account for VRAM: multi-scale at 960px needs smaller batch.
STRATEGIES = {
    "cutmix": {
        "description": "CutMix augmentation (cutmix=0.5)",
        "train_overrides": {
            "cutmix": 0.5,
        },
        "batch_size": 8,
        "imgsz": 640,
    },
    "multi_scale": {
        "description": "Multi-scale training (imgsz=960, multi_scale=True)",
        "train_overrides": {
            "multi_scale": True,
        },
        "batch_size": 4,  # Reduced for OOM safety at 960px
        "imgsz": 960,
    },
    "combined": {
        "description": "CutMix + Multi-scale combined",
        "train_overrides": {
            "cutmix": 0.5,
            "multi_scale": True,
        },
        "batch_size": 4,
        "imgsz": 960,
    },
}


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

def evaluate_model(model_path, split, device):
    """Evaluate a trained YOLO-World model on the given split. Returns metrics dict."""

    print(f"\n  Evaluating on {split} set: {model_path}")

    model = YOLOWorld(model_path)

    results = model.val(
        data=DATASET_YAML,
        split=split,
        imgsz=IMG_SIZE_EVAL,
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
# TRAINING WITH OOM FALLBACK
# ============================================

def train_yoloworld_mitigation(seed, device, strategy_name, strategy_config, epochs=100):
    """
    Train YOLO-World XL with the given mitigation strategy.
    Returns the run directory name.

    Includes OOM fallback: if the initial batch size causes OOM,
    retries with batch=2.
    """

    run_name = f"yoloworld_xl_{strategy_name}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_size = strategy_config["batch_size"]
    imgsz = strategy_config["imgsz"]
    overrides = strategy_config["train_overrides"]

    print(f"\n{'=' * 60}")
    print(f"TRAINING YOLO-World XL — strategy={strategy_name}, seed={seed}")
    print(f"  batch={batch_size}, imgsz={imgsz}, overrides={overrides}")
    print(f"{'=' * 60}")

    # Build training kwargs: baseline params + strategy overrides
    train_kwargs = dict(
        data=DATASET_YAML,
        epochs=epochs,
        patience=20,
        batch=batch_size,
        imgsz=imgsz,
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

    # Apply strategy-specific overrides
    train_kwargs.update(overrides)

    # Attempt training with OOM fallback
    for attempt_batch in [batch_size, 2]:
        try:
            train_kwargs["batch"] = attempt_batch
            if attempt_batch != batch_size:
                # Update run name for retry
                run_name = f"yoloworld_xl_{strategy_name}_seed{seed}_b{attempt_batch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                train_kwargs["name"] = run_name
                print(f"\n  OOM fallback: retrying with batch={attempt_batch}")

            model = YOLOWorld("yolov8x-worldv2.pt")
            model.set_classes(["Dummy crack", "Paper crack", "PVC pipe crack"])
            model.train(**train_kwargs)
            return run_name  # Success

        except RuntimeError as e:
            if "out of memory" in str(e).lower() and attempt_batch != 2:
                print(f"\n  OOM at batch={attempt_batch}, will retry with batch=2")
                clear_gpu()
                continue
            else:
                raise  # Re-raise if not OOM or already at minimum batch

    return run_name


# ============================================
# AGGREGATE STATISTICS
# ============================================

def compute_aggregate(all_results):
    """Compute mean +/- std across seeds for each strategy and split."""

    aggregate = {}

    strategy_names = sorted(set(r["strategy"] for r in all_results))

    for strategy in strategy_names:
        aggregate[strategy] = {}
        strategy_runs = [r for r in all_results if r["strategy"] == strategy]

        for split in ["val", "test"]:
            split_runs = [r[split] for r in strategy_runs if split in r]
            if not split_runs:
                continue

            metric_keys = [k for k in split_runs[0] if isinstance(split_runs[0][k], (int, float))]

            stats = {}
            for key in metric_keys:
                values = [run[key] for run in split_runs]
                stats[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "values": values,
                }

            aggregate[strategy][split] = stats

    return aggregate


# ============================================
# MAIN RUNNER
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Mitigation strategy experiment runner")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--device", type=int, default=0, help="GPU device index (default: 0)")
    parser.add_argument(
        "--strategies", nargs="+", default=list(STRATEGIES.keys()),
        choices=list(STRATEGIES.keys()),
        help="Strategies to run (default: all)"
    )
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing runs")
    args = parser.parse_args()

    device = args.device
    all_results = []
    output_file = "mitigation_results.json"

    print(f"Using GPU device: {device}")
    if torch.cuda.is_available():
        print(f"  {torch.cuda.get_device_name(device)} — "
              f"{torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB total")

    print(f"\nStrategies to run: {args.strategies}")
    print(f"Seeds: {SEEDS}")
    print(f"Total runs: {len(args.strategies) * len(SEEDS)}")

    # Load any existing results to allow resuming
    if os.path.exists(output_file):
        with open(output_file) as f:
            saved = json.load(f)
            all_results = saved.get("runs", [])
        print(f"Loaded {len(all_results)} existing results from {output_file}")

    # Helper to check if a run already exists
    def run_exists(strategy, seed):
        return any(r["strategy"] == strategy and r["seed"] == seed for r in all_results)

    # ── Run each strategy sequentially ─────────────────────────
    for strategy_name in args.strategies:
        strategy_config = STRATEGIES[strategy_name]
        print(f"\n{'#' * 60}")
        print(f"# STRATEGY: {strategy_name} — {strategy_config['description']}")
        print(f"{'#' * 60}")

        for seed in SEEDS:
            if run_exists(strategy_name, seed):
                print(f"\n[SKIP] {strategy_name} seed={seed} already in results")
                continue

            if not args.eval_only:
                run_name = train_yoloworld_mitigation(
                    seed, device=device, strategy_name=strategy_name,
                    strategy_config=strategy_config, epochs=args.epochs,
                )
                clear_gpu()
                run_dir = f"{PROJECT_NAME}/{run_name}"
            else:
                print(f"\n[SKIP] {strategy_name} seed={seed} — --eval-only set and no existing run")
                continue

            # Evaluate on both val and test at imgsz=640
            best_weights = f"{run_dir}/weights/best.pt"
            if not os.path.exists(best_weights):
                print(f"  WARNING: {best_weights} not found, skipping evaluation")
                continue

            val_metrics = evaluate_model(best_weights, "val", device=device)
            test_metrics = evaluate_model(best_weights, "test", device=device)
            clear_gpu()

            result = {
                "model": "yoloworld_xl",
                "strategy": strategy_name,
                "seed": seed,
                "run_dir": run_dir,
                "batch_size": strategy_config["batch_size"],
                "imgsz_train": strategy_config["imgsz"],
                "imgsz_eval": IMG_SIZE_EVAL,
                "overrides": strategy_config["train_overrides"],
                "val": val_metrics,
                "test": test_metrics,
                "timestamp": datetime.now().isoformat(),
            }
            all_results.append(result)

            # Save incrementally
            _save_results(all_results, output_file)
            print(f"  Saved results for {strategy_name} seed={seed}")

    # ── Final aggregate ────────────────────────────────────────
    _save_results(all_results, output_file)

    print("\n" + "=" * 60)
    print("ALL MITIGATION RUNS COMPLETE")
    print("=" * 60)
    _print_summary(all_results)


def _save_results(all_results, output_file):
    """Save results with aggregate statistics."""
    aggregate = compute_aggregate(all_results)
    output = {
        "runs": all_results,
        "aggregate": aggregate,
        "seeds": SEEDS,
        "strategies": list(STRATEGIES.keys()),
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)


def _print_summary(all_results):
    """Print a summary table of all results."""
    aggregate = compute_aggregate(all_results)

    for strategy_name, splits in aggregate.items():
        print(f"\n{'─' * 50}")
        print(f"  {strategy_name} ({STRATEGIES.get(strategy_name, {}).get('description', '')})")
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
