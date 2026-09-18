#!/usr/bin/env bash
# One-time install: makes the thermal guard a real cron-managed background service instead of
# something that has to be launched by hand and watched every session. Run once per box.
#   - @reboot: start the guard immediately after any reboot (including a thermal one).
#   - */2 * * * *: every 2 minutes, restart it if it's not running for any other reason.
# After this, nobody -- human or agent -- needs to check on it again; cron does.
set -euo pipefail
WD="/home/admin/llm-workspace/Open-Spark-Jev/scripts/ops/thermal_guard_watchdog.sh"
( crontab -l 2>/dev/null | grep -v "thermal_guard_watchdog.sh" ; \
  echo "@reboot $WD" ; \
  echo "*/2 * * * * $WD" \
) | crontab -
echo "installed. current crontab:"
crontab -l
