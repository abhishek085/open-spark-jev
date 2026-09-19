#!/usr/bin/env bash
# Cron-managed watchdog for thermal_guard.sh. Run via crontab (installed by
# scripts/ops/install_thermal_cron.sh), never launched by hand or watched interactively --
# it ensures the guard daemon is alive AND actually running (not merely present), restarting
# it after a reboot or if it ever dies or gets stuck, with zero ongoing attention required.
set -uo pipefail

# Resolve paths from this script's own location so a clone works anywhere.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GUARD="$REPO/scripts/ops/thermal_guard.sh"
LOG="$REPO/.thermal_guard.log"
PATTERN_MARKER="thermal_guard.sh"

# A stopped (T-state) guard process still matches a plain existence check but is not doing
# anything -- exactly the failure mode a first real version of this guard hit (it briefly
# SIGSTOPped itself along with the job it was pausing; see thermal_guard.sh's matching_pids()
# comment for the fix on the guard side -- this check is the second, independent safety net).
# Only R (running) or S (interruptible sleep, e.g. its `sleep $POLL_SECONDS`) count as alive.
alive_pid=""
for pid in $(pgrep -f "$PATTERN_MARKER" 2>/dev/null); do
  state=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ')
  case "$state" in
    R*|S*) alive_pid="$pid" ;;
    T*) echo "$(date '+%Y-%m-%d %H:%M:%S') [watchdog] guard pid $pid is STOPPED (state=$state) -- killing it, it cannot self-resume" >> "$LOG"
        kill -KILL "$pid" 2>/dev/null || true ;;
  esac
done

if [ -z "$alive_pid" ]; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') [watchdog] guard not alive, starting it" >> "$LOG"
  nohup bash "$GUARD" "python -m open_spark_jev" >> "$LOG" 2>&1 &
  disown
fi
exit 0
