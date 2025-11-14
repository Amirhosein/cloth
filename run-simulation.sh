#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run-simulation.sh <seed> <outdir>
# Example:
#   ./run-simulation.sh 1992 ./out

SEED="${1:-1992}"
OUTDIR="${2:-./out}"

# macOS: LeakSanitizer unsupported. Keep ASan/UBSan helpful but don't abort the shell.
export ASAN_OPTIONS="halt_on_error=1,alloc_dealloc_mismatch=1,symbolize=1,fast_unwind_on_malloc=0"
export UBSAN_OPTIONS="halt_on_error=1,print_stacktrace=1"

# GSL seed for reproducibility
export GSL_RNG_SEED="${SEED}"

mkdir -p "${OUTDIR}"
echo "Running cloth with GSL_RNG_SEED=${GSL_RNG_SEED}, OUTDIR=${OUTDIR}"

# Don't 'exec' so the shell stays alive even if the program crashes
./cloth "${OUTDIR}" || {
  status=$?
  echo
  echo "cloth exited with status: ${status}"
  echo "If AddressSanitizer printed an error above, use it to locate the bug."
  # Pause if launched from Finder / a terminal profile that closes on exit
  read -r -p "Press Enter to close..."
  exit "${status}"
}

python batch-means.py "${OUTDIR}"
