#!/usr/bin/env bash
#
# Run the official Rinha 2026 benchmark against a submission.
#
# Usage:
#   ./benchmarks/run.sh <submission-dir> <run-name> [extra-compose-override.yml...]
#
# Example:
#   ./benchmarks/run.sh references/qrust-luanmonteiro qrust-baseline \
#     benchmarks/_harness/qrust-resources.override.yml
#   ./benchmarks/run.sh submission go-scaffold-v1
#
# What it does, in order:
#   1. docker compose up -d --build, with all provided overrides
#   2. waits for GET http://localhost:9999/ready to return 200 (max 90s)
#   3. smoke test: k6 run test/smoke.js                — must pass
#   4. start docker stats collector in background
#   5. k6 run test/test.js, summary into the run dir
#   6. capture compose logs
#   7. tear down, write manifest.json
#
# Outputs to benchmarks/<run-name>-<UTC-timestamp>-<sha-or-untracked>/
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_DIR"

if [ $# -lt 2 ]; then
    echo "usage: $0 <submission-dir> <run-name> [extra-compose-override.yml...]" >&2
    exit 2
fi

SUBMISSION_DIR="$1"
RUN_NAME="$2"
shift 2
EXTRA_OVERRIDES=("$@")

if [ ! -f "$SUBMISSION_DIR/docker-compose.yml" ]; then
    echo "error: $SUBMISSION_DIR/docker-compose.yml not found" >&2
    exit 1
fi

# === preflight checks ===
command -v docker >/dev/null || { echo "error: docker not installed"; exit 1; }
command -v k6 >/dev/null     || { echo "error: k6 not installed"; exit 1; }
command -v jq >/dev/null     || { echo "error: jq not installed"; exit 1; }

K6_RINHA_DIR="$WORKSPACE_DIR/references/rinha-official"
if [ ! -f "$K6_RINHA_DIR/test/test.js" ]; then
    echo "error: $K6_RINHA_DIR/test/test.js not found"
    echo "       did you clone references/rinha-official?"
    exit 1
fi

# === run identifier ===
TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"
SUB_SHA="$(git -C "$SUBMISSION_DIR" rev-parse --short HEAD 2>/dev/null || echo "untracked")"
RUN_ID="${RUN_NAME}-${TIMESTAMP}-${SUB_SHA}"
RUN_DIR="$WORKSPACE_DIR/benchmarks/$RUN_ID"
mkdir -p "$RUN_DIR"

echo ">>> run: $RUN_ID"
echo ">>> output: $RUN_DIR"

# === build compose command ===
COMPOSE=(docker compose -f "$SUBMISSION_DIR/docker-compose.yml")
for override in "${EXTRA_OVERRIDES[@]}"; do
    if [ ! -f "$override" ]; then
        echo "error: override file $override not found" >&2
        exit 1
    fi
    COMPOSE+=(-f "$override")
done

# === ensure clean state ===
echo ">>> tearing down any existing stack..."
"${COMPOSE[@]}" down --remove-orphans >/dev/null 2>&1 || true

# === bring up ===
echo ">>> compose up --build (this can take a while on first run)..."
"${COMPOSE[@]}" up -d --build > "$RUN_DIR/compose-up.log" 2>&1

cleanup() {
    set +e
    echo ">>> tearing down..."
    "${COMPOSE[@]}" logs > "$RUN_DIR/compose-logs.txt" 2>&1
    "${COMPOSE[@]}" down --remove-orphans >/dev/null 2>&1
    [ -n "${STATS_PID:-}" ] && kill "$STATS_PID" 2>/dev/null
}
trap cleanup EXIT

# === wait for /ready ===
echo ">>> waiting for /ready..."
READY_TIMEOUT_S=${READY_TIMEOUT_S:-180}
READY_START=$(date +%s)
while true; do
    if curl -fsS -o /dev/null --max-time 2 http://localhost:9999/ready; then
        echo ">>> /ready returned 200 after $(( $(date +%s) - READY_START ))s"
        break
    fi
    if [ $(( $(date +%s) - READY_START )) -gt $READY_TIMEOUT_S ]; then
        echo "error: /ready did not return 200 in $READY_TIMEOUT_S s" >&2
        exit 1
    fi
    sleep 1
done

# === smoke test ===
echo ">>> smoke test..."
( cd "$K6_RINHA_DIR" && k6 run test/smoke.js > "$RUN_DIR/smoke.log" 2>&1 ) \
    && echo ">>> smoke passed" \
    || { echo "error: smoke test failed; see $RUN_DIR/smoke.log" >&2; exit 1; }

# === docker stats collector ===
echo ">>> starting docker stats collector..."
docker stats --no-stream --format '{{json .}}' > /dev/null 2>&1 || true  # warmup
(
    while true; do
        ts="$(date -u +%FT%TZ)"
        docker stats --no-stream --format "{\"ts\":\"$ts\",\"name\":\"{{ .Name }}\",\"cpu\":\"{{ .CPUPerc }}\",\"mem\":\"{{ .MemUsage }}\",\"netio\":\"{{ .NetIO }}\",\"pids\":\"{{ .PIDs }}\"}" 2>/dev/null
        sleep 1
    done
) > "$RUN_DIR/stats.ndjson" &
STATS_PID=$!

# === main load test ===
echo ">>> running k6 ramping-arrival-rate test (≈ 2 min)..."
START_TS=$(date -u +%FT%TZ)
( cd "$K6_RINHA_DIR" && k6 run \
    --summary-export="$RUN_DIR/results.json" \
    test/test.js > "$RUN_DIR/k6.log" 2>&1 ) || true
END_TS=$(date -u +%FT%TZ)

# Move any results.json the script wrote into the run dir too
[ -f "$K6_RINHA_DIR/test/results.json" ] && \
    mv "$K6_RINHA_DIR/test/results.json" "$RUN_DIR/results.json"

# === manifest ===
echo ">>> writing manifest..."
cat > "$RUN_DIR/manifest.json" <<EOF
{
  "run_id":         "$RUN_ID",
  "run_name":       "$RUN_NAME",
  "submission":     "$SUBMISSION_DIR",
  "submission_sha": "$SUB_SHA",
  "overrides":      $(printf '%s\n' "${EXTRA_OVERRIDES[@]}" | jq -R . | jq -s .),
  "started_utc":    "$START_TS",
  "ended_utc":      "$END_TS",
  "host": {
    "os":   "$(uname -s)",
    "rel":  "$(uname -r)",
    "arch": "$(uname -m)"
  },
  "k6_version": "$(k6 version 2>&1 | head -1)"
}
EOF

# === summary ===
if [ -f "$RUN_DIR/results.json" ]; then
    echo ""
    echo ">>> SUMMARY for $RUN_ID:"
    jq -r '
      "  p99:          " + .p99,
      "  final_score:  " + (.scoring.final_score | tostring),
      "  p99_score:    " + (.scoring.p99_score.value | tostring) + " (cut: " + (.scoring.p99_score.cut_triggered | tostring) + ")",
      "  det_score:    " + (.scoring.detection_score.value | tostring) + " (cut: " + (.scoring.detection_score.cut_triggered | tostring) + ")",
      "  TP/TN/FP/FN/Err: " + (.scoring.breakdown.true_positive_detections | tostring)
                            + "/" + (.scoring.breakdown.true_negative_detections | tostring)
                            + "/" + (.scoring.breakdown.false_positive_detections | tostring)
                            + "/" + (.scoring.breakdown.false_negative_detections | tostring)
                            + "/" + (.scoring.breakdown.http_errors | tostring),
      "  failure_rate: " + .scoring.failure_rate
    ' "$RUN_DIR/results.json"
else
    echo ">>> no results.json produced (k6 likely failed) — see $RUN_DIR/k6.log"
fi
