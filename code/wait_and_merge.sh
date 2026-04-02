#!/bin/bash
# Wait for ALL mitigation workers (original + parallel + chained), then merge.

echo "$(date): Waiting for all mitigation workers..."
echo ""
echo "Workers:"
echo "  GPU 1 (original cutmix):  mitigation_runner.py"
echo "  GPU 3 (multi_scale):      PID 220499"
echo "  GPU 4 (multi_scale):      PID 220662"
echo "  GPU 6 (multi_scale):      PID 220888"
echo "  GPU 7 (combined 2,123):   PID 569977"
echo "  GPU 3 chain (combined 42): PID 570232"
echo "  GPU 4 chain (combined 0):  PID 570406"
echo "  GPU 6 chain (combined 1):  PID 570499"
echo ""

# Wait for all known PIDs
for pid in 220499 220662 220888 569977 570232 570406 570499; do
  tail --pid=$pid -f /dev/null 2>/dev/null
  echo "$(date): PID $pid finished"
done

# Wait for original mitigation_runner.py
ORIG_PID=$(ps aux | grep '[m]itigation_runner.py' | awk '{print $2}' | head -1)
if [ -n "$ORIG_PID" ]; then
  echo "$(date): Waiting for original process PID $ORIG_PID..."
  tail --pid=$ORIG_PID -f /dev/null 2>/dev/null
  echo "$(date): Original process finished"
else
  echo "$(date): Original process already finished"
fi

echo ""
echo "$(date): All workers complete! Running merge..."
cd .
python mitigation_parallel.py --merge-only
echo "$(date): Merge complete! Results saved to mitigation_results.json"
