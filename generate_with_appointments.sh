#!/bin/bash
set -e
cd "$(dirname "$0")"

# Priority:
# 1. CLI args
# 2. Environment variables
# 3. Defaults

PATIENTS="${1:-${PATIENTS:-10}}"
MAX_BUNDLE_MB="${2:-${MAX_BUNDLE_MB:-2}}"

echo "▶ Generating $PATIENTS patients"
echo "▶ Max bundle size: ${MAX_BUNDLE_MB}MB"

python3 add_appointments.py \
  -p "$PATIENTS" \
  --max-bundle-size "$MAX_BUNDLE_MB"
