#!/bin/bash
# GPU 6: wait for multi_scale to finish, then run combined seed=1
tail --pid=220888 -f /dev/null 2>/dev/null
echo "$(date): GPU 6 multi_scale done. Starting combined seed=1..."
CUDA_VISIBLE_DEVICES=6 python ./mitigation_worker.py \
  --strategy combined --seeds 1 \
  --output ./mitigation_results_gpu6_combined.json
echo "$(date): GPU 6 combined done."
