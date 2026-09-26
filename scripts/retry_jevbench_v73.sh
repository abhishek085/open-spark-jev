#!/usr/bin/env bash
# Retry v7.3's JevBench run through the now-fixed scripts/run_jevbench.sh (unbuffered gateway
# output, label-hashed port so a still-shutting-down previous gateway can't collide, bounded health
# probe, and a real non-zero exit when n_valid comes back 0). Waits for part2's GPU work to finish.
set -uo pipefail
cd "$(dirname "$0")/.."
while pgrep -f "scripts/run_v7x_part2.sh" > /dev/null; do sleep 60; done
echo "$(date '+%F %T') retrying jevbench for v7.3-4b" | tee -a runs/v7/v7x.log
if scripts/run_jevbench.sh v7.3-4b v7.3-4b-retry > runs/v7/jevbench_v7.3_retry.log 2>&1; then
  echo "$(date '+%F %T') jevbench retry v7.3 OK" | tee -a runs/v7/v7x.log
else
  echo "$(date '+%F %T') jevbench retry v7.3 FAILED (see runs/v7/jevbench_v7.3_retry.log)" | tee -a runs/v7/v7x.log
fi
