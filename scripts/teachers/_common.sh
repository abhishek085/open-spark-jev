#!/usr/bin/env bash
# Shared helpers for teacher launch scripts. Source, don't execute.

mem_preflight() {
  # $1 = required GiB. Uses MemAvailable (reclaimable-aware), matching the dgx_spark_benchy
  # convention on this box (see docs/DGX_SPARK.md) rather than raw MemFree, which
  # under-reports on this unified-memory architecture.
  local need_gb="$1"
  local avail_gb
  avail_gb=$(awk '/MemAvailable/{printf "%d", $2/1024/1024}' /proc/meminfo)
  if [ "$avail_gb" -lt "$need_gb" ]; then
    echo "REFUSING TO LAUNCH: need ~${need_gb}GB available, only ${avail_gb}GB free." >&2
    echo "This box is unified memory (GPU+CPU share the same pool). Free some up first:" >&2
    echo "  docker ps   # see what else is running" >&2
    echo "  - the sibling nokast-foundry project's teacher/student containers are NOT" >&2
    echo "    stopped automatically by this repo; stop them yourself if you need the memory:" >&2
    echo "    docker stop nokast-teacher-vllm nokast-student-vllm" >&2
    echo "  - or stop another osj-teacher-* container this repo may have left running:" >&2
    echo "    docker ps --filter name=osj-teacher" >&2
    exit 1
  fi
  echo "memory preflight ok: ${avail_gb}GB available (need ${need_gb}GB)"
}

wait_healthy() {
  local port="$1" name="$2" timeout_s="${3:-1800}"
  local waited=0
  until curl -s -m 2 "http://localhost:$port/health" >/dev/null 2>&1; do
    if ! docker ps --format '{{.Names}}' | grep -qx "$name"; then
      echo "$name exited before becoming healthy; see: docker logs $name" >&2
      exit 1
    fi
    if [ "$waited" -ge "$timeout_s" ]; then
      echo "$name did not become healthy within ${timeout_s}s; see: docker logs $name" >&2
      exit 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "$name healthy on port $port after ${waited}s"
}
