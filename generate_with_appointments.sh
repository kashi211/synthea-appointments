#!/bin/bash
# Run Synthea then add Appointment resources to each patient bundle.
# Usage:
#   ./generate_with_appointments.sh [patient_count] [max_bundle_size_mb]
#
# Defaults:
#   patient_count = 10
#   max_bundle_size_mb = 2

set -e
cd "$(dirname "$0")"

PATIENTS="${1:-10}"
MAX_BUNDLE_MB="${2:-2}"

echo "▶ Generating $PATIENTS patients"
echo "▶ Max bundle size: ${MAX_BUNDLE_MB}MB"

python3 add_appointments.py \
  -p "$PATIENTS" \
  --max-bundle-size "$MAX_BUNDLE_MB"