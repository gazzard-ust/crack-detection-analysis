"""Single-GPU mitigation worker. Runs specific strategy+seed pairs."""
import argparse
import json
import os
import sys
from datetime import datetime

import gc
import torch
import numpy as np
from ultralytics import YOLOWorld

from mitigation_runner import (
    STRATEGIES, SEEDS, DATASET_YAML, PROJECT_NAME,
    IMG_SIZE_EVAL, CONF_THRESHOLD, IOU_THRESHOLD, BATCH_SIZE_EVAL,
    evaluate_model, train_yoloworld_mitigation, clear_gpu, compute_aggregate,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = 0  # CUDA_VISIBLE_DEVICES remaps physical GPU to 0
    all_results = []

    # Load existing results from this worker's output file if resuming
    if os.path.exists(args.output):
        with open(args.output) as f:
            data = json.load(f)
        all_results = data.get("runs", [])
        print(f"Loaded {len(all_results)} existing results from {args.output}")

    done_seeds = {r["seed"] for r in all_results if r["strategy"] == args.strategy}

    strategy_config = STRATEGIES[args.strategy]
    print(f"Worker: strategy={args.strategy}, seeds={args.seeds}, device={device}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Already done seeds: {done_seeds}")

    for seed in args.seeds:
        if seed in done_seeds:
            print(f"\n[SKIP] {args.strategy} seed={seed} already done")
            continue

        print(f"\n{'#' * 60}")
        print(f"# {args.strategy} seed={seed}")
        print(f"{'#' * 60}")

        run_name = train_yoloworld_mitigation(
            seed, device=device, strategy_name=args.strategy,
            strategy_config=strategy_config, epochs=100,
        )
        clear_gpu()

        run_dir = f"{PROJECT_NAME}/{run_name}"
        best_weights = f"{run_dir}/weights/best.pt"

        if not os.path.exists(best_weights):
            print(f"  WARNING: {best_weights} not found, skipping evaluation")
            continue

        val_metrics = evaluate_model(best_weights, "val", device=device)
        test_metrics = evaluate_model(best_weights, "test", device=device)
        clear_gpu()

        result = {
            "model": "yoloworld_xl",
            "strategy": args.strategy,
            "seed": seed,
            "run_dir": run_dir,
            "batch_size": strategy_config["batch_size"],
            "imgsz_train": strategy_config["imgsz"],
            "imgsz_eval": IMG_SIZE_EVAL,
            "overrides": dict(strategy_config["train_overrides"]),
            "val": val_metrics,
            "test": test_metrics,
            "timestamp": datetime.now().isoformat(),
        }
        all_results.append(result)

        # Save incrementally
        output = {
            "runs": all_results,
            "aggregate": compute_aggregate(all_results),
            "timestamp": datetime.now().isoformat(),
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved to {args.output}")

    print(f"\nWorker complete. {len(all_results)} total runs.")


if __name__ == "__main__":
    main()
