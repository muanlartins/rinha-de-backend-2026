#!/bin/sh
# Pre-flight warmup: hit /fraud-score with a representative slice of payloads
# before k6's real load fires. Warms the branch predictor, L1/L2 caches, the
# Go scheduler, and HAProxy keepalive sockets so the first 100 real requests
# don't pay cold-start tail-latency cost.
set -eu

BASE_URL="${BASE_URL:-http://lb:9999}"
READY_URL="${BASE_URL}/ready"
SCORE_URL="${BASE_URL}/fraud-score"
DEBUG_URL="${BASE_URL}/debug/info"
READY_RETRIES="${READY_RETRIES:-240}"
READY_SLEEP="${READY_SLEEP:-0.25}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-120}"

p01='{"id":"w-01","transaction":{"amount":42.10,"installments":1,"requested_at":"2026-03-11T00:15:23Z"},"customer":{"avg_amount":85.50,"tx_count_24h":2,"known_merchants":["MERC-001","MERC-002"]},"merchant":{"id":"MERC-001","mcc":"5411","avg_amount":52.30},"terminal":{"is_online":false,"card_present":true,"km_from_home":2.1},"last_transaction":null}'
p02='{"id":"w-02","transaction":{"amount":384.88,"installments":3,"requested_at":"2026-03-11T06:23:35Z"},"customer":{"avg_amount":769.76,"tx_count_24h":3,"known_merchants":["MERC-009","MERC-001"]},"merchant":{"id":"MERC-009","mcc":"5912","avg_amount":298.95},"terminal":{"is_online":false,"card_present":true,"km_from_home":13.7},"last_transaction":{"timestamp":"2026-03-11T05:58:35Z","km_from_current":18.86}}'
p03='{"id":"w-03","transaction":{"amount":2911.41,"installments":12,"requested_at":"2026-03-11T12:17:11Z"},"customer":{"avg_amount":411.03,"tx_count_24h":8,"known_merchants":["MERC-221","MERC-010"]},"merchant":{"id":"MERC-551","mcc":"7995","avg_amount":712.22},"terminal":{"is_online":true,"card_present":false,"km_from_home":2.18},"last_transaction":{"timestamp":"2026-03-11T11:51:05Z","km_from_current":1.34}}'
p04='{"id":"w-04","transaction":{"amount":567.81,"installments":4,"requested_at":"2026-03-23T16:25:31Z"},"customer":{"avg_amount":146.45,"tx_count_24h":7,"known_merchants":["MERC-010","MERC-011"]},"merchant":{"id":"MERC-043","mcc":"7801","avg_amount":278.43},"terminal":{"is_online":true,"card_present":false,"km_from_home":181.28},"last_transaction":{"timestamp":"2026-03-23T14:52:31Z","km_from_current":251.60}}'
p05='{"id":"w-05","transaction":{"amount":85.40,"installments":1,"requested_at":"2026-03-13T03:30:00Z"},"customer":{"avg_amount":92.10,"tx_count_24h":1,"known_merchants":[]},"merchant":{"id":"MERC-512","mcc":"4511","avg_amount":340.00},"terminal":{"is_online":true,"card_present":false,"km_from_home":1250.5},"last_transaction":{"timestamp":"2026-03-12T15:20:00Z","km_from_current":900.2}}'
p06='{"id":"w-06","transaction":{"amount":5500.00,"installments":10,"requested_at":"2026-03-14T22:05:18Z"},"customer":{"avg_amount":80.25,"tx_count_24h":15,"known_merchants":["MERC-001","MERC-002","MERC-003"]},"merchant":{"id":"MERC-999","mcc":"7801","avg_amount":4200.00},"terminal":{"is_online":true,"card_present":false,"km_from_home":700.0},"last_transaction":{"timestamp":"2026-03-14T21:30:00Z","km_from_current":500.0}}'
p07='{"id":"w-07","transaction":{"amount":25.50,"installments":1,"requested_at":"2026-03-15T09:11:00Z"},"customer":{"avg_amount":28.00,"tx_count_24h":4,"known_merchants":["MERC-101","MERC-102","MERC-103","MERC-104"]},"merchant":{"id":"MERC-104","mcc":"5812","avg_amount":31.25},"terminal":{"is_online":false,"card_present":true,"km_from_home":3.5},"last_transaction":{"timestamp":"2026-03-15T08:45:00Z","km_from_current":4.0}}'
p08='{"id":"w-08","transaction":{"amount":1763.51,"installments":3,"requested_at":"2026-03-14T07:30:17Z"},"customer":{"avg_amount":264.5,"tx_count_24h":6,"known_merchants":["MERC-016","MERC-006","MERC-009","MERC-014"]},"merchant":{"id":"MERC-009","mcc":"4511","avg_amount":105.57},"terminal":{"is_online":false,"card_present":false,"km_from_home":340.87},"last_transaction":{"timestamp":"2026-03-14T05:50:17Z","km_from_current":216.85}}'
p09='{"id":"w-09","transaction":{"amount":2690.86,"installments":7,"requested_at":"2026-03-13T13:13:03Z"},"customer":{"avg_amount":321.52,"tx_count_24h":10,"known_merchants":["MERC-001","MERC-013","MERC-010"]},"merchant":{"id":"MERC-041","mcc":"5999","avg_amount":54.8},"terminal":{"is_online":true,"card_present":false,"km_from_home":131.20},"last_transaction":{"timestamp":"2026-03-13T11:15:03Z","km_from_current":65.44}}'
p10='{"id":"w-10","transaction":{"amount":996.43,"installments":6,"requested_at":"2026-03-26T07:37:47Z"},"customer":{"avg_amount":391.41,"tx_count_24h":6,"known_merchants":["MERC-004","MERC-014","MERC-009"]},"merchant":{"id":"MERC-040","mcc":"5999","avg_amount":161.14},"terminal":{"is_online":true,"card_present":false,"km_from_home":321.62},"last_transaction":null}'
p11='{"id":"w-11","transaction":{"amount":878.11,"installments":4,"requested_at":"2026-03-26T09:57:57Z"},"customer":{"avg_amount":334.71,"tx_count_24h":5,"known_merchants":["MERC-010","MERC-016","MERC-004"]},"merchant":{"id":"MERC-032","mcc":"4511","avg_amount":238.8},"terminal":{"is_online":false,"card_present":true,"km_from_home":84.24},"last_transaction":{"timestamp":"2026-03-26T09:35:57Z","km_from_current":186.81}}'
p12='{"id":"w-12","transaction":{"amount":603.17,"installments":3,"requested_at":"2026-03-24T20:42:30Z"},"customer":{"avg_amount":363.36,"tx_count_24h":8,"known_merchants":["MERC-011","MERC-015","MERC-002"]},"merchant":{"id":"MERC-002","mcc":"5812","avg_amount":165.45},"terminal":{"is_online":false,"card_present":true,"km_from_home":290.22},"last_transaction":{"timestamp":"2026-03-24T19:59:30Z","km_from_current":35.94}}'

echo "warmup: waiting on ${READY_URL}"
i=1
while [ "${i}" -le "${READY_RETRIES}" ]; do
  if curl -fsS --max-time 1 "${READY_URL}" >/dev/null; then
    break
  fi
  if [ "${i}" -eq "${READY_RETRIES}" ]; then
    echo "warmup: ready check failed after ${READY_RETRIES} attempts" >&2
    exit 1
  fi
  sleep "${READY_SLEEP}"
  i=$((i + 1))
done

# Print one /debug/info snapshot so the rinha bot's logs capture our state.
echo "warmup: $(curl -fsS --max-time 2 "${DEBUG_URL}" 2>/dev/null || echo 'no /debug/info')"

i=1
while [ "${i}" -le "${WARMUP_ROUNDS}" ]; do
  case $((i % 12)) in
    0)  body="${p01}" ;;
    1)  body="${p02}" ;;
    2)  body="${p03}" ;;
    3)  body="${p04}" ;;
    4)  body="${p05}" ;;
    5)  body="${p06}" ;;
    6)  body="${p07}" ;;
    7)  body="${p08}" ;;
    8)  body="${p09}" ;;
    9)  body="${p10}" ;;
    10) body="${p11}" ;;
    11) body="${p12}" ;;
  esac
  curl -fsS --max-time 2 -H 'content-type: application/json' -d "${body}" "${SCORE_URL}" >/dev/null
  i=$((i + 1))
done

echo "warmup: ${WARMUP_ROUNDS} requests done"
