#!/usr/bin/env bash
# Stop one of THIS repo's ephemeral teacher containers (never touches nokast-* containers).
#   scripts/teachers/stop_teacher.sh nemotron120b|gptoss120b|all
set -euo pipefail
case "${1:-all}" in
  nemotron120b) docker stop osj-teacher-nemotron 2>/dev/null || true ;;
  gptoss120b)   docker stop osj-teacher-gptoss 2>/dev/null || true ;;
  all)          docker stop osj-teacher-nemotron osj-teacher-gptoss 2>/dev/null || true ;;
  *) echo "usage: $0 nemotron120b|gptoss120b|all" >&2; exit 1 ;;
esac
