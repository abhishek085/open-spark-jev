#!/usr/bin/env bash
# Thermal guard for long-running GPU jobs on the DGX Spark (GB10).
#
# Why this exists: the Spark's small form factor throttles/shuts down protectively around
# 92-95C package/SoC or GPU temperature. Sustained near-100% GPU utilization for hours
# (exactly the load pattern of back-to-back SFT/RLCD/GRPO training runs) can drive it there.
# This polls GPU temp (nvidia-smi) and the max of the ACPI thermal zones (SoC/package) every
# POLL_SECONDS, and when either crosses PAUSE_C, SIGSTOPs every process matching PATTERN
# (paused processes hold their GPU memory/context but do no compute, so training resumes
# exactly where it left off) until both drop below RESUME_C, then SIGCONTs them. Logs every
# reading so a post-hoc review is possible, and prints clear PAUSE:/RESUME: lines to watch for.
#
# Usage:
#   scripts/ops/thermal_guard.sh [pattern] &
#   pattern defaults to 'python -m open_spark_jev' (matches any of this repo's training
#   jobs by process command line, so a single guard protects the whole pipeline without needing
#   the PID passed in and re-armed per stage).
#
# Docker containers are a second, separate load source (scripts/teachers/*.sh) this same guard
# must cover: a teacher container's vLLM process runs inside the container's own PID namespace
# (launched with --network host but not --pid=host), so host-side `pgrep`/SIGSTOP can never see
# or pause it -- that gap existed silently until a real gemma26b data-generation job was running
# unprotected under it (2026-09-19). `docker pause`/`unpause` freezes all of a container's
# processes at the cgroup level and needs no PID visibility, so it is used instead. Any running
# container matching DOCKER_NAME_GLOB (default 'osj-teacher-*', i.e. any of this repo's teacher
# containers, not just one hardcoded name) is paused/resumed alongside the pattern-matched host
# processes, on the same hot/cool cycle.
set -uo pipefail

PATTERN="${1:-python -m open_spark_jev}"
DOCKER_NAME_GLOB="${2:-${THERMAL_DOCKER_GLOB:-osj-*}}"
POLL_SECONDS="${THERMAL_POLL_SECONDS:-5}"
PAUSE_C="${THERMAL_PAUSE_C:-91}"    # pause when either sensor crosses this
RESUME_C="${THERMAL_RESUME_C:-85}"  # resume only once BOTH sensors are back under this
                                     # (hysteresis -- avoids rapid pause/resume flapping)

paused=0
echo "[thermal_guard] watching pattern='$PATTERN' docker_glob='$DOCKER_NAME_GLOB' poll=${POLL_SECONDS}s pause>=${PAUSE_C}C resume<${RESUME_C}C (self pid $$, excluded from matches)"

matching_containers() {
  command -v docker >/dev/null 2>&1 || return 0
  docker ps --filter "status=running" --format '{{.Names}}' 2>/dev/null | while read -r name; do
    case "$name" in
      $DOCKER_NAME_GLOB) echo "$name" ;;
    esac
  done
}

matching_pids() {
  # Exclude our own PID and anything that looks like another thermal_guard.sh instance -- the
  # guard's own command line literally contains $PATTERN as an argv token (it was invoked as
  # `bash thermal_guard.sh "$PATTERN"`), so a naive `pgrep -f "$PATTERN"` matches the guard
  # itself. A first real run learned this the hard way: it SIGSTOPped its own process right
  # after pausing the training job, freezing both indefinitely with no way to self-resume
  # (harmless here only because the paused training job also stopped consuming GPU/thermal
  # headroom -- do not rely on that luck holding in general).
  pgrep -f "$PATTERN" 2>/dev/null | grep -vxF "$$" | while read -r pid; do
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    case "$cmd" in
      *thermal_guard.sh*) ;;  # skip -- this is a guard process, never a signal target
      *) echo "$pid" ;;
    esac
  done
}

read_gpu_temp() {
  nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1
}

read_soc_temp() {
  # max across all ACPI thermal zones, in whole C (files are millidegrees)
  local max=0 v
  for f in /sys/class/thermal/thermal_zone*/temp; do
    [ -r "$f" ] || continue
    v=$(($(cat "$f" 2>/dev/null || echo 0) / 1000))
    [ "$v" -gt "$max" ] && max=$v
  done
  echo "$max"
}

while true; do
  gpu=$(read_gpu_temp)
  soc=$(read_soc_temp)
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[thermal_guard] $ts gpu=${gpu:-NA}C soc=${soc:-NA}C paused=$paused"

  hot=0
  { [ -n "$gpu" ] && [ "$gpu" -ge "$PAUSE_C" ]; } && hot=1
  { [ -n "$soc" ] && [ "$soc" -ge "$PAUSE_C" ]; } && hot=1

  cool=1
  { [ -n "$gpu" ] && [ "$gpu" -ge "$RESUME_C" ]; } && cool=0
  { [ -n "$soc" ] && [ "$soc" -ge "$RESUME_C" ]; } && cool=0

  if [ "$hot" = "1" ] && [ "$paused" = "0" ]; then
    pids=$(matching_pids)
    containers=$(matching_containers)
    if [ -n "$pids" ] || [ -n "$containers" ]; then
      [ -n "$pids" ] && { echo "[thermal_guard] PAUSE: gpu=${gpu}C soc=${soc}C >= ${PAUSE_C}C -- SIGSTOP on pids: $pids"; kill -STOP $pids 2>/dev/null || true; }
      for c in $containers; do
        echo "[thermal_guard] PAUSE: gpu=${gpu}C soc=${soc}C >= ${PAUSE_C}C -- docker pause $c"
        docker pause "$c" 2>/dev/null || true
      done
      paused=1
    fi
  elif [ "$paused" = "1" ] && [ "$cool" = "1" ]; then
    pids=$(matching_pids)
    if [ -n "$pids" ]; then
      echo "[thermal_guard] RESUME: gpu=${gpu}C soc=${soc}C < ${RESUME_C}C -- SIGCONT on pids: $pids"
      kill -CONT $pids 2>/dev/null || true
    fi
    # docker ps only lists running containers, so a paused one won't show up in
    # matching_containers() -- ask docker directly which of our containers are paused instead.
    for c in $(docker ps --filter "status=paused" --format '{{.Names}}' 2>/dev/null); do
      case "$c" in
        $DOCKER_NAME_GLOB)
          echo "[thermal_guard] RESUME: gpu=${gpu}C soc=${soc}C < ${RESUME_C}C -- docker unpause $c"
          docker unpause "$c" 2>/dev/null || true ;;
      esac
    done
    paused=0
  fi

  sleep "$POLL_SECONDS"
done
