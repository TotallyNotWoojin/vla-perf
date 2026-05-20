#!/usr/bin/env bash
# Run all VLA performance modeling scripts sequentially.
# Run from the vla-perf/vla-perf/ directory:
#   bash run_all_vla_perf.sh
# Optional: redirect all output to a file:
#   bash run_all_vla_perf.sh 2>&1 | tee perf_results/run_all.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p perf_results

SCRIPTS=(
    openvla_perf.py
    pi0_perf.py
    smolvla_perf.py
    qwen2vla_perf.py
    xvla_perf.py
)

PASS=()
FAIL=()
TIMES=()

total_start=$(date +%s)

for script in "${SCRIPTS[@]}"; do
    echo ""
    echo "=================================================================="
    echo " START: $script  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "=================================================================="

    t_start=$(date +%s)
    if python "$script"; then
        t_end=$(date +%s)
        elapsed=$(( t_end - t_start ))
        PASS+=("$script")
        TIMES+=("${elapsed}s")
        echo "------------------------------------------------------------------"
        echo " DONE:  $script  [${elapsed}s]"
        echo "------------------------------------------------------------------"
    else
        t_end=$(date +%s)
        elapsed=$(( t_end - t_start ))
        FAIL+=("$script")
        TIMES+=("${elapsed}s  FAILED")
        echo "------------------------------------------------------------------"
        echo " FAIL:  $script  [${elapsed}s]"
        echo "------------------------------------------------------------------"
        # Continue with remaining scripts even if one fails
    fi
done

total_end=$(date +%s)
total_elapsed=$(( total_end - total_start ))

echo ""
echo "=================================================================="
echo " SUMMARY  (total: ${total_elapsed}s)"
echo "=================================================================="
echo " Passed (${#PASS[@]}):"
for s in "${PASS[@]}"; do echo "   OK  $s"; done
if [ ${#FAIL[@]} -gt 0 ]; then
    echo " Failed (${#FAIL[@]}):"
    for s in "${FAIL[@]}"; do echo "   FAIL $s"; done
    echo ""
    echo "Results written to perf_results/"
    exit 1
fi
echo ""
echo "Results written to perf_results/"
