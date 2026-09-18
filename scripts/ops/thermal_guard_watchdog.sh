#!/usr/bin/env bash
# Cron-managed watchdog for thermal_guard.sh. Run via crontab (installed by
# scripts/ops/install_thermal_cron.sh), never launched by hand or watched interactively --
# it ensures the guard daemon is always running, restarting it after a reboot or if it ever
# dies, with zero ongoing attention required.
set -uo pipefail

GUARD="/home/admin/llm-workspace/Open-Spark-Jev/scripts/ops/thermal_guard.sh"
LOG="/home/admin/llm-workspace/Open-Spark-Jev/.thermal_guard.log"
PATTERN_MARKER="thermal_guard.sh"  # how we recognize a live guard process via pgrep

if ! pgrep -f "$PATTERN_MARKER" >/dev/null 2>&1; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') [watchdog] guard not running, starting it" >> "$LOG"
  nohup bash "$GUARD" "python -m open_spark_jev.train" >> "$LOG" 2>&1 &
  disown
fi
exit 0
