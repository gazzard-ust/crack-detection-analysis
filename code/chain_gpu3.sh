#!/bin/bash
# GPU 3: wait for multi_scale to finish, then run combined seed=42
tail --pid=220499 -f /dev/null 2>/dev/null
echo "$(date): GPU 3 multi_scale done. Starting combined seed=42..."
CUDA_VISIBLE_DEVICES=3 python ./mitigation_worker.py \
  --strategy combined --seeds 42 \
  --output ./mitigation_results_gpu3_combined.json
echo "$(date): GPU 3 combined done."
