"""
YOLO-World Model Evaluation Script
===================================
Evaluates trained model and logs metrics to wandb
"""

from ultralytics import YOLOWorld
import wandb
import os
import json
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================

# Path to your trained model (update after training)
MODEL_PATH = "crack_detection_runs/yoloworld_xl_20251226_171626/weights/best.pt"

# Dataset path
DATASET_YAML = "pipe-crack-detection-1/data.yaml"

# Evaluation settings
IMG_SIZE = 640
BATCH_SIZE = 16
DEVICE = 0
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# wandb settings
WANDB_PROJECT = "yoloworld-crack-detection"
LOG_TO_WANDB = True

# ============================================
# EVALUATION FUNCTIONS
# ============================================

def evaluate_model(model_path, data_yaml, split="test"):
    """
    Evaluate model on specified dataset split.
    """
    
    print(f"\n{'=' * 60}")
    print(f"EVALUATING MODEL ON {split.upper()} SET")
    print(f"{'=' * 60}")
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model = YOLOWorld(model_path)
    
    # Run evaluation
    print(f"Running evaluation on {split} set...")
    results = model.val(
        data=data_yaml,
        split=split,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        plots=True,
        save_json=True,
        verbose=True,
    )
    
    # Extract metrics (mAP50-95 as primary metric)
    metrics = {
        "split": split,
        "mAP50-95": float(results.box.map),
        "mAP50": float(results.box.map50),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "f1_score": float(2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-6)),
    }
    
    # Per-class metrics
    class_names = list(results.names.values())
    per_class_ap50 = results.box.ap50.tolist()
    per_class_ap = results.box.ap.tolist()
    
    for i, class_name in enumerate(class_names):
        metrics[f"AP50-95_{class_name}"] = float(per_class_ap[i])
        metrics[f"AP50_{class_name}"] = float(per_class_ap50[i])
    
    return metrics, results


def print_metrics(metrics):
    """Print metrics in a formatted table."""
    
    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 60}")
    
    # Overall metrics
    print(f"\n{'OVERALL METRICS':^40}")
    print("-" * 40)
    print(f"{'Metric':<25} {'Value':>12}")
    print("-" * 40)
    print(f"{'mAP@50-95':<25} {metrics['mAP50-95']:>12.4f}")
    print(f"{'mAP@50':<25} {metrics['mAP50']:>12.4f}")
    print(f"{'Precision':<25} {metrics['precision']:>12.4f}")
    print(f"{'Recall':<25} {metrics['recall']:>12.4f}")
    print(f"{'F1 Score':<25} {metrics['f1_score']:>12.4f}")
    
    # Per-class metrics
    print(f"\n{'PER-CLASS AP@50-95':^40}")
    print("-" * 40)
    for key, value in metrics.items():
        if key.startswith("AP50-95_"):
            class_name = key.replace("AP50-95_", "")
            print(f"{class_name:<25} {value:>12.4f}")
    
    print("=" * 60)


def log_to_wandb(metrics, model_path):
    """Log evaluation metrics to wandb."""
    
    wandb.init(
        project=WANDB_PROJECT,
        name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags=["evaluation", "yolo-world", "crack-detection"],
        config={
            "model_path": model_path,
            "conf_threshold": CONF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
        }
    )
    
    wandb.log(metrics)
    
    # Create summary table
    table = wandb.Table(columns=["Metric", "Value"])
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            table.add_data(key, value)
    wandb.log({"metrics_table": table})
    
    print(f"\nMetrics logged to wandb: {wandb.run.url}")
    wandb.finish()


def save_metrics(metrics, output_path="evaluation_results.json"):
    """Save metrics to JSON file."""
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to: {output_path}")


def main():
    """Main evaluation function."""
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("\nPlease update MODEL_PATH to your trained model, e.g.:")
        print('  MODEL_PATH = "crack_detection_runs/yoloworld_xl_20251226_143000/weights/best.pt"')
        return
    
    # Check if dataset exists
    if not os.path.exists(DATASET_YAML):
        print(f"ERROR: Dataset not found at {DATASET_YAML}")
        return
    
    # Evaluate on test set
    test_metrics, test_results = evaluate_model(MODEL_PATH, DATASET_YAML, split="test")
    print_metrics(test_metrics)
    
    # Evaluate on validation set
    val_metrics, val_results = evaluate_model(MODEL_PATH, DATASET_YAML, split="val")
    print_metrics(val_metrics)
    
    # Combine metrics
    all_metrics = {
        "test": test_metrics,
        "val": val_metrics,
        "model_path": MODEL_PATH,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save metrics locally
    save_metrics(all_metrics)
    
    # Log to wandb
    if LOG_TO_WANDB:
        wandb_metrics = {}
        for split_name, split_metrics in [("test", test_metrics), ("val", val_metrics)]:
            for key, value in split_metrics.items():
                if isinstance(value, (int, float)):
                    wandb_metrics[f"{split_name}/{key}"] = value
        
        log_to_wandb(wandb_metrics, MODEL_PATH)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()