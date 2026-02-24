#!/bin/bash
# GPU 4: wait for multi_scale to finish, then run combined seed=0
tail --pid=220662 -f /dev/null 2>/dev/null
echo "$(date): GPU 4 multi_scale done. Starting combined seed=0..."
CUDA_VISIBLE_DEVICES=4 python ./mitigation_worker.py \
  --strategy combined --seeds 0 \
  --output ./mitigation_results_gpu4_combined.json
echo "$(date): GPU 4 combined done."
