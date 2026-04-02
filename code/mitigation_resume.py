"""
Resume missing mitigation runs.

Phase 1 (eval-only): 5 runs have best.pt on disk but no JSON results.
  - cutmix seed=123
  - multi_scale seed=2
  - combined seed=42, seed=1, seed=123

Phase 2 (train + eval): 2 runs have no best.pt.
  - multi_scale seed=123   (GPU 0)
  - combined seed=0        (GPU 1)

Phase 3: Merge all results into mitigation_results.json.
"""

import json
import os
import sys
import subprocess
import time
from datetime import datetime

# Runs that have best.pt but are missing from result JSONs
EVAL_ONLY_RUNS = [
    ("cutmix",      123, "crack_detection_runs/yoloworld_xl_cutmix_seed123_20260224_064922"),
    ("multi_scale",   2, "crack_detection_runs/yoloworld_xl_multi_scale_seed2_20260224_063546"),
    ("combined",     42, "crack_detection_runs/yoloworld_xl_combined_seed42_20260224_023140"),
    ("combined",      1, "crack_detection_runs/yoloworld_xl_combined_seed1_20260224_065321"),
    ("combined",    123, "crack_detection_runs/yoloworld_xl_combined_seed123_20260224_054540"),
]

# Runs that need training from scratch
TRAIN_RUNS = [
    ("multi_scale", 123),
    ("combined",      0),
]


def phase1_eval(device=0):
    """Evaluate runs that have best.pt but no JSON results."""
    from mitigation_runner import (
        STRATEGIES, evaluate_model, clear_gpu, IMG_SIZE_EVAL,
    )

    results = []
    for strategy_name, seed, run_dir in EVAL_ONLY_RUNS:
        best_weights = f"{run_dir}/weights/best.pt"
        if not os.path.exists(best_weights):
            print(f"  SKIP {strategy_name} seed={seed}: {best_weights} not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"  EVAL: {strategy_name} seed={seed}")
        print(f"  Weights: {best_weights}")
        print(f"{'=' * 60}")

        val_metrics = evaluate_model(best_weights, "val", device=device)
        test_metrics = evaluate_model(best_weights, "test", device=device)
        clear_gpu()

        strategy_config = STRATEGIES[strategy_name]
        result = {
            "model": "yoloworld_xl",
            "strategy": strategy_name,
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
        results.append(result)
        print(f"  mAP50-95(test) = {test_metrics['mAP50-95']:.4f}")

    # Save to a temporary file
    output = {"runs": results, "timestamp": datetime.now().isoformat()}
    with open("mitigation_results_eval_only.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nPhase 1 complete: {len(results)} runs evaluated -> mitigation_results_eval_only.json")


def phase2_train():
    """Launch training for the 2 missing runs on separate GPUs."""
    train_gpus = [4, 5]  # GPUs allocated for training
    procs = []
    for i, (strategy, seed) in enumerate(TRAIN_RUNS):
        gpu = train_gpus[i]
        output_file = f"mitigation_results_resume_gpu{gpu}.json"
        cmd = [
            sys.executable, "mitigation_worker.py",
            "--strategy", strategy,
            "--seeds", str(seed),
            "--output", output_file,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print(f"Launching {strategy} seed={seed} on GPU {gpu} -> {output_file}")
        proc = subprocess.Popen(cmd, env=env)
        procs.append((gpu, strategy, seed, proc, output_file))
        time.sleep(2)

    print(f"\n{len(procs)} training jobs launched. Waiting...")
    for gpu, strategy, seed, proc, output_file in procs:
        proc.wait()
        status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        print(f"  GPU {gpu} ({strategy} s{seed}): {status}")

    print("Phase 2 complete.")


def phase3_merge():
    """Merge all result files into mitigation_results.json."""
    import glob
    import numpy as np
    from mitigation_runner import compute_aggregate, STRATEGIES, SEEDS, _print_summary

    all_runs = []
    seen = set()

    # Load from all result files
    files = (
        ["mitigation_results.json"]
        + sorted(glob.glob("mitigation_results_gpu*.json"))
        + ["mitigation_results_eval_only.json"]
        + sorted(glob.glob("mitigation_results_resume_gpu*.json"))
    )

    for path in files:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for r in data.get("runs", []):
            key = (r["strategy"], r["seed"])
            if key not in seen:
                all_runs.append(r)
                seen.add(key)
                print(f"  {r['strategy']:12s} seed={r['seed']:3d}  from {path}")

    print(f"\nTotal merged: {len(all_runs)}/15")

    # Check completeness
    all_strategies = ["cutmix", "multi_scale", "combined"]
    all_seeds = [42, 0, 1, 2, 123]
    missing = [(s, sd) for s in all_strategies for sd in all_seeds if (s, sd) not in seen]
    if missing:
        print(f"\nWARNING: Still missing {len(missing)} runs:")
        for s, sd in missing:
            print(f"  {s} seed={sd}")
    else:
        print("\nAll 15 runs present!")

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

    print("\nSaved merged results to mitigation_results.json")
    _print_summary(all_runs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=0,
                        help="Run specific phase (0=all)")
    parser.add_argument("--device", type=int, default=0, help="GPU for phase 1 eval")
    args = parser.parse_args()

    if args.phase == 0 or args.phase == 1:
        print("\n" + "=" * 60)
        print("PHASE 1: Evaluate runs with best.pt but no JSON results")
        print("=" * 60)
        phase1_eval(device=args.device)

    if args.phase == 0 or args.phase == 2:
        print("\n" + "=" * 60)
        print("PHASE 2: Train missing runs (multi_scale s123, combined s0)")
        print("=" * 60)
        phase2_train()

    if args.phase == 0 or args.phase == 3:
        print("\n" + "=" * 60)
        print("PHASE 3: Merge all results")
        print("=" * 60)
        phase3_merge()
