"""
Parallel Mitigation Runner
===========================
Splits remaining mitigation runs across multiple GPUs.
Each GPU worker saves to its own results file to avoid race conditions.
A final merge step combines all results into mitigation_results.json.
"""

import argparse
import json
import os
import sys
import subprocess
import time
from datetime import datetime


def get_completed_runs():
    """Check which runs are already done."""
    results_file = "mitigation_results.json"
    if os.path.exists(results_file):
        with open(results_file) as f:
            data = json.load(f)
        return [(r["strategy"], r["seed"]) for r in data.get("runs", [])]
    return []


def main():
    parser = argparse.ArgumentParser(description="Parallel mitigation runner")
    parser.add_argument("--merge-only", action="store_true",
                        help="Only merge existing worker result files")
    args = parser.parse_args()

    if args.merge_only:
        merge_results()
        return

    # All runs needed
    all_strategies = ["cutmix", "multi_scale", "combined"]
    seeds = [42, 0, 1, 2, 123]
    completed = get_completed_runs()

    remaining = []
    for strat in all_strategies:
        for seed in seeds:
            if (strat, seed) not in completed:
                remaining.append((strat, seed))

    print(f"Completed: {len(completed)}/15")
    print(f"Remaining: {len(remaining)}")
    for s, sd in remaining:
        print(f"  - {s}, seed={sd}")

    # GPU assignments: distribute remaining runs across GPUs 3, 4, 6, 7
    # (GPU 1 is still running the original process for cutmix)
    gpus = [3, 4, 6, 7]

    # Filter out cutmix runs (the original process on GPU 1 handles those)
    parallel_runs = [(s, sd) for s, sd in remaining if s != "cutmix"]

    print(f"\nRuns to parallelize (non-cutmix): {len(parallel_runs)}")

    # Distribute round-robin across GPUs
    gpu_assignments = {gpu: [] for gpu in gpus}
    for i, (strat, seed) in enumerate(parallel_runs):
        gpu = gpus[i % len(gpus)]
        gpu_assignments[gpu].append((strat, seed))

    # Print assignments
    for gpu, runs in gpu_assignments.items():
        print(f"\n  GPU {gpu}: {len(runs)} runs")
        for s, sd in runs:
            print(f"    - {s}, seed={sd}")

    # Launch worker processes
    workers = []
    for gpu, runs in gpu_assignments.items():
        if not runs:
            continue
        output_file = f"mitigation_results_gpu{gpu}.json"
        # Build run spec as JSON string
        run_spec = json.dumps([(s, sd) for s, sd in runs])
        cmd = [
            sys.executable, "-c", WORKER_CODE,
            "--gpu", str(gpu),
            "--runs", run_spec,
            "--output", output_file,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print(f"\nLaunching worker on GPU {gpu} (PID will follow)...")
        proc = subprocess.Popen(cmd, env=env)
        workers.append((gpu, proc, output_file))
        print(f"  PID: {proc.pid}")
        time.sleep(2)  # Stagger launches

    print(f"\nAll {len(workers)} workers launched. Waiting for completion...")

    # Wait for all workers
    for gpu, proc, output_file in workers:
        proc.wait()
        status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        print(f"  GPU {gpu}: {status}")

    # Merge all results
    merge_results()


def merge_results():
    """Merge all worker result files + original results into mitigation_results.json."""
    import glob
    import numpy as np

    all_runs = []
    seen = set()

    # Load original results first
    if os.path.exists("mitigation_results.json"):
        with open("mitigation_results.json") as f:
            data = json.load(f)
        for r in data.get("runs", []):
            key = (r["strategy"], r["seed"])
            if key not in seen:
                all_runs.append(r)
                seen.add(key)

    # Load worker results
    for path in sorted(glob.glob("mitigation_results_gpu*.json")):
        print(f"  Merging {path}")
        with open(path) as f:
            data = json.load(f)
        for r in data.get("runs", []):
            key = (r["strategy"], r["seed"])
            if key not in seen:
                all_runs.append(r)
                seen.add(key)

    print(f"\nTotal merged runs: {len(all_runs)}/15")

    # Recompute aggregate
    from mitigation_runner import compute_aggregate, STRATEGIES, SEEDS, _print_summary

    aggregate = compute_aggregate(all_runs)
    output = {
        "runs": all_runs,
        "aggregate": aggregate,
        "seeds": SEEDS,
        "strategies": list(STRATEGIES.keys()),
        "timestamp": datetime.now().isoformat(),
    }
    with open("mitigation_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Saved merged results to mitigation_results.json")
    _print_summary(all_runs)


# Worker code that runs inside each subprocess
WORKER_CODE = r'''
import argparse
import json
import os
import gc
import sys
from datetime import datetime

import torch
import numpy as np
from ultralytics import YOLOWorld

sys.path.insert(0, os.getcwd())
from mitigation_runner import (
    STRATEGIES, SEEDS, DATASET_YAML, PROJECT_NAME,
    IMG_SIZE_EVAL, CONF_THRESHOLD, IOU_THRESHOLD, BATCH_SIZE_EVAL,
    evaluate_model, train_yoloworld_mitigation, clear_gpu, compute_aggregate,
)

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--runs", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

device = 0  # CUDA_VISIBLE_DEVICES remaps to 0
runs_to_do = json.loads(args.runs)
all_results = []

print(f"Worker starting on GPU {args.gpu} (mapped to device {device})")
print(f"Runs: {runs_to_do}")

for strategy_name, seed in runs_to_do:
    strategy_config = STRATEGIES[strategy_name]
    print(f"\n{'#' * 60}")
    print(f"# {strategy_name} seed={seed} on GPU {args.gpu}")
    print(f"{'#' * 60}")

    run_name = train_yoloworld_mitigation(
        seed, device=device, strategy_name=strategy_name,
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
        "strategy": strategy_name,
        "seed": seed,
        "run_dir": run_dir,
        "batch_size": strategy_config["batch_size"],
        "imgsz_train": strategy_config["imgsz"],
        "imgsz_eval": IMG_SIZE_EVAL,
        "overrides": {k: v if not isinstance(v, bool) else v for k, v in strategy_config["train_overrides"].items()},
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

print(f"\nWorker on GPU {args.gpu} complete. {len(all_results)} runs done.")
'''


if __name__ == "__main__":
    main()
