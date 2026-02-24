"""
YOLOv8x Baseline Training Script for Crack Detection
=====================================================
Project: TurtleBot3 Burger Crack Detection with CO2 Gas Monitoring
Platform: NVIDIA DGX (Single GPU)
Purpose: Baseline comparison (YOLOv8x without CLIP) against YOLO-World XL
Features: Weights & Biases (wandb) integration, configurable seed for multi-seed runs

Hyperparameters are IDENTICAL to ylwd_train.py to ensure a fair comparison —
only the model architecture differs (standard YOLOv8x vs YOLO-World XL).
"""

import argparse
from ultralytics import YOLO
from datetime import datetime
import torch
import os
import wandb


# ============================================
# CONFIGURATION — matches ylwd_train.py exactly
# ============================================

DATASET_YAML = "pipe-crack-detection-1/data.yaml"
MODEL = "yolov8x.pt"

EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640
LEARNING_RATE = 2e-4
PATIENCE = 20

DEVICE = 0

PROJECT_NAME = "crack_detection_runs"

WANDB_PROJECT = "yoloworld-crack-detection"
WANDB_ENTITY = None


# ============================================
# FUNCTIONS
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8x baseline training for crack detection")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Training epochs (default: {EPOCHS})")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    return parser.parse_args()


def check_system():
    """Check system configuration before training."""
    print("=" * 60)
    print("SYSTEM CHECK")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        gpu_name = torch.cuda.get_device_name(DEVICE)
        gpu_mem = torch.cuda.get_device_properties(DEVICE).total_memory / 1e9
        print(f"Using GPU {DEVICE}: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("WARNING: CUDA not available! Training will be slow on CPU.")

    if os.path.exists(DATASET_YAML):
        print(f"Dataset found: {DATASET_YAML}")
    else:
        print(f"ERROR: Dataset not found at {DATASET_YAML}")
        return False

    print("=" * 60)
    return True


def init_wandb(run_name, seed):
    """Initialize Weights & Biases for experiment tracking."""

    config = {
        "model": MODEL,
        "dataset": DATASET_YAML,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "img_size": IMG_SIZE,
        "learning_rate": LEARNING_RATE,
        "patience": PATIENCE,
        "device": DEVICE,
        "optimizer": "AdamW",
        "weight_decay": 0.05,
        "seed": seed,
    }

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config=config,
        tags=["yolov8x", "baseline", "crack-detection", f"seed-{seed}"],
    )

    print(f"wandb initialized: {wandb.run.url}")
    return config


def train(seed=42, epochs=EPOCHS, batch_size=BATCH_SIZE, use_wandb=True):
    """Main training function."""

    run_name = f"yolov8x_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\n" + "=" * 60)
    print(f"YOLOv8x BASELINE TRAINING — seed={seed}")
    print("=" * 60)

    if use_wandb:
        config = init_wandb(run_name, seed)
    else:
        config = {"model": MODEL, "seed": seed, "epochs": epochs}

    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Load YOLOv8x (standard, no CLIP)
    print(f"\nLoading model: {MODEL}")
    model = YOLO(MODEL)

    print("\n" + "=" * 60)
    print("STARTING TRAINING...")
    print("=" * 60 + "\n")

    results = model.train(
        # Dataset
        data=DATASET_YAML,

        # Training duration
        epochs=epochs,
        patience=PATIENCE,

        # Batch and image size
        batch=batch_size,
        imgsz=IMG_SIZE,

        # Learning rate schedule
        lr0=LEARNING_RATE,
        lrf=0.01,
        warmup_epochs=3,
        warmup_momentum=0.8,

        # Optimizer
        optimizer="AdamW",
        weight_decay=0.05,

        # Augmentation — identical to YOLO-World run
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

        # Single GPU
        device=DEVICE,
        workers=8,

        # Saving
        project=PROJECT_NAME,
        name=run_name,
        save=True,
        save_period=10,

        # Validation
        val=True,
        plots=True,

        # Other
        verbose=True,
        seed=seed,
    )

    # Log final metrics to wandb
    if use_wandb:
        final_metrics = {
            "final/mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
            "final/mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
            "final/precision": results.results_dict.get("metrics/precision(B)", 0),
            "final/recall": results.results_dict.get("metrics/recall(B)", 0),
        }
        wandb.log(final_metrics)

        best_model_path = f"{PROJECT_NAME}/{run_name}/weights/best.pt"
        if os.path.exists(best_model_path):
            artifact = wandb.Artifact(
                name=f"yolov8x-crack-{run_name}",
                type="model",
                description="YOLOv8x baseline for crack detection",
            )
            artifact.add_file(best_model_path)
            wandb.log_artifact(artifact)
            print(f"\nModel artifact logged to wandb")

        wandb.finish()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"\nResults saved to: {PROJECT_NAME}/{run_name}/")
    print(f"Best model: {PROJECT_NAME}/{run_name}/weights/best.pt")

    return results, run_name


def main():
    args = parse_args()

    if not check_system():
        return

    train(seed=args.seed, epochs=args.epochs, batch_size=args.batch, use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()
