"""
YOLO-World XL Fine-tuning Script for Crack Detection
=====================================================
Project: TurtleBot3 Burger Crack Detection with CO2 Gas Monitoring
Platform: NVIDIA DGX (Single GPU)
Features: Weights & Biases (wandb) integration for metrics tracking
"""

from ultralytics import YOLOWorld
from datetime import datetime
import torch
import os
import wandb

# ============================================
# CONFIGURATION - MODIFY THESE AS NEEDED
# ============================================

# Dataset path
DATASET_YAML = "pipe-crack-detection-1/data.yaml"

# Model selection
MODEL = "yolov8x-worldv2.pt"  # XL model (largest, most accurate)

# Custom classes for crack detection
CLASSES = ["Dummy crack", "Paper crack", "PVC pipe crack"]

# Training parameters
EPOCHS = 100
BATCH_SIZE = 16          # Reduce to 8 or 4 if out of memory
IMG_SIZE = 640
LEARNING_RATE = 2e-4
PATIENCE = 20            # Early stopping patience

# Single GPU Configuration
DEVICE = 0               # Use GPU 0 only

# Output directory
PROJECT_NAME = "crack_detection_runs"
RUN_NAME = f"yoloworld_xl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ============================================
# WANDB CONFIGURATION
# ============================================

WANDB_PROJECT = "yoloworld-crack-detection"
WANDB_ENTITY = None      # Your wandb username or team name (None = default)

# ============================================
# TRAINING SCRIPT
# ============================================

def check_system():
    """Check system configuration before training."""
    print("=" * 60)
    print("SYSTEM CHECK")
    print("=" * 60)
    
    # CUDA check
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
    
    # Dataset check
    if os.path.exists(DATASET_YAML):
        print(f"Dataset found: {DATASET_YAML}")
    else:
        print(f"ERROR: Dataset not found at {DATASET_YAML}")
        print("Please check the path and try again.")
        return False
    
    # wandb check
    print(f"wandb project: {WANDB_PROJECT}")
    
    print("=" * 60)
    return True


def init_wandb():
    """Initialize Weights & Biases for experiment tracking."""
    
    # Configuration to log
    config = {
        "model": MODEL,
        "dataset": DATASET_YAML,
        "classes": CLASSES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "img_size": IMG_SIZE,
        "learning_rate": LEARNING_RATE,
        "patience": PATIENCE,
        "device": DEVICE,
        "optimizer": "AdamW",
        "weight_decay": 0.05,
    }
    
    # Initialize wandb
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=RUN_NAME,
        config=config,
        tags=["yolo-world", "crack-detection", "fine-tuning"],
    )
    
    print(f"wandb initialized: {wandb.run.url}")
    return config


def train():
    """Main training function with wandb logging."""
    
    print("\n" + "=" * 60)
    print("YOLO-WORLD XL FINE-TUNING FOR CRACK DETECTION")
    print("=" * 60)
    
    # Initialize wandb
    config = init_wandb()
    
    # Print configuration
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load YOLO-World XL model
    print(f"\nLoading model: {MODEL}")
    model = YOLOWorld(MODEL)
    
    # Set custom classes
    print(f"Setting custom classes: {CLASSES}")
    model.set_classes(CLASSES)
    
    # Start training with wandb integration
    print("\n" + "=" * 60)
    print("STARTING TRAINING...")
    print("=" * 60 + "\n")
    
    results = model.train(
        # Dataset
        data=DATASET_YAML,
        
        # Training duration
        epochs=EPOCHS,
        patience=PATIENCE,
        
        # Batch and image size
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        
        # Learning rate schedule
        lr0=LEARNING_RATE,
        lrf=0.01,
        warmup_epochs=3,
        warmup_momentum=0.8,
        
        # Optimizer
        optimizer="AdamW",
        weight_decay=0.05,
        
        # Augmentation
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
        name=RUN_NAME,
        save=True,
        save_period=10,
        
        # Validation
        val=True,
        plots=True,
        
        # Other
        verbose=True,
        seed=42,
    )
    
    # Log final metrics to wandb (mAP50-95 as primary metric)
    final_metrics = {
        "final/mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
        "final/mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
        "final/precision": results.results_dict.get("metrics/precision(B)", 0),
        "final/recall": results.results_dict.get("metrics/recall(B)", 0),
    }
    wandb.log(final_metrics)
    
    # Log best model as artifact
    best_model_path = f"{PROJECT_NAME}/{RUN_NAME}/weights/best.pt"
    if os.path.exists(best_model_path):
        artifact = wandb.Artifact(
            name=f"yoloworld-crack-{RUN_NAME}",
            type="model",
            description="Fine-tuned YOLO-World XL for crack detection"
        )
        artifact.add_file(best_model_path)
        wandb.log_artifact(artifact)
        print(f"\nModel artifact logged to wandb")
    
    # Training complete
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"\nResults saved to: {PROJECT_NAME}/{RUN_NAME}/")
    print(f"Best model: {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
    print(f"wandb run: {wandb.run.url}")
    
    # Finish wandb run
    wandb.finish()
    
    return results


def main():
    """Main entry point."""
    # Check system first
    if not check_system():
        return
    
    # Confirm before starting
    print("\nReady to start training?")
    print("This may take several hours depending on your GPU.")
    response = input("Continue? [y/N]: ").strip().lower()
    
    if response == 'y':
        train()
    else:
        print("Training cancelled.")


if __name__ == "__main__":
    main()