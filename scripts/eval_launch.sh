#!/bin/sh
# launchd entrypoint for the weekly Synapse eval run.
# Computes the date-stamped report path and runs the harness.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REPORT_DIR="${HOME}/.synapse/eval/reports"
mkdir -p "${REPORT_DIR}" "${HOME}/.synapse/.logs"
exec "${REPO}/.venv/bin/python" -m synapse eval \
  --golden "${HOME}/.synapse/eval/golden-v1.json" \
  --history "${HOME}/.synapse/eval/history.jsonl" \
  --report "${REPORT_DIR}/$(date +%F).json"
