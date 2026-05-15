#!/usr/bin/env bash
# multi-run.sh — run benchmarks/run.sh N times against the same submission
# directory and emit a summary of medians + noise.
#
# Usage:
#   benchmarks/multi-run.sh <submission-dir> <name-prefix> [N=3]
#
# Output: one line per run + a summary block at the end.
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <submission-dir> <name-prefix> [N]" >&2
    exit 2
fi

SUB="$1"
PREFIX="$2"
N="${3:-3}"

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_DIR"

P99S=()
SCORES=()

for i in $(seq 1 "$N"); do
    NAME="${PREFIX}-r${i}"
    echo ""
    echo "=== Run $i / $N ($NAME) ==="
    bash benchmarks/run.sh "$SUB" "$NAME" 2>&1 | tail -10 | tee /tmp/multi-run-$i.log
    # Find latest run dir for this name
    RUN_DIR=$(ls -td benchmarks/${NAME}-* 2>/dev/null | head -1)
    if [ -z "$RUN_DIR" ] || [ ! -f "$RUN_DIR/results.json" ]; then
        echo "WARN: no results.json for run $i"
        continue
    fi
    P99=$(jq -r '.p99' "$RUN_DIR/results.json" | sed 's/ms//')
    SCORE=$(jq -r '.scoring.final_score' "$RUN_DIR/results.json")
    P99S+=("$P99")
    SCORES+=("$SCORE")
done

echo ""
echo "=== SUMMARY for $PREFIX ($N runs) ==="
printf "p99 (ms):     "
printf "%s  " "${P99S[@]}"
echo ""
printf "score:        "
printf "%s  " "${SCORES[@]}"
echo ""

# Compute medians via sort+pick
P99_SORTED=$(printf "%s\n" "${P99S[@]}" | sort -n)
SCORE_SORTED=$(printf "%s\n" "${SCORES[@]}" | sort -n)
MID=$(( (N + 1) / 2 ))
P99_MED=$(echo "$P99_SORTED" | sed -n "${MID}p")
SCORE_MED=$(echo "$SCORE_SORTED" | sed -n "${MID}p")
echo "p99 median:   ${P99_MED} ms"
echo "score median: ${SCORE_MED}"
