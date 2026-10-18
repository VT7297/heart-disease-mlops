#!/usr/bin/env bash
# Build the serving image, run it, and verify the API end to end.
# Usage: scripts/container_smoke_test.sh [image-tag]
# Uses podman if available, else docker (override with CONTAINER_ENGINE=docker).
set -euo pipefail

IMAGE="${1:-heart-api:local}"
PORT="${PORT:-8000}"
NAME="heart-api-smoke"
ENGINE="${CONTAINER_ENGINE:-$(command -v podman >/dev/null && echo podman || echo docker)}"
BASE="http://127.0.0.1:${PORT}"
cd "$(dirname "$0")/.."

FORMAT=""
[ "$ENGINE" = "podman" ] && FORMAT="--format docker"   # keeps HEALTHCHECK in the image

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\033[31mFAIL: %s\033[0m\n' "$*"; $ENGINE logs "$NAME" 2>&1 | tail -20; exit 1; }
cleanup() { $ENGINE rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

step "Build ($ENGINE) -> $IMAGE"
# shellcheck disable=SC2086
$ENGINE build $FORMAT -f docker/Dockerfile -t "$IMAGE" .
$ENGINE images "$IMAGE"

step "Run container"
cleanup
$ENGINE run -d --name "$NAME" -p "${PORT}:8000" --read-only --tmpfs /tmp "$IMAGE" >/dev/null
echo "user inside container: $($ENGINE exec "$NAME" id)"

step "Wait for /ready"
for _ in $(seq 1 30); do
  curl -sf "$BASE/ready" >/dev/null && break
  sleep 1
done
curl -sf "$BASE/ready" || fail "service did not become ready"
echo

step "GET /health"
curl -sf "$BASE/health"; echo

step "POST /predict  (high-risk sample)"
curl -sf -X POST "$BASE/predict" -H 'Content-Type: application/json' -d @samples/patient_high_risk.json \
  | tee /tmp/pred_high.json; echo
grep -q '"confidence"' /tmp/pred_high.json || fail "no confidence in response"

step "POST /predict  (low-risk sample)"
curl -sf -X POST "$BASE/predict" -H 'Content-Type: application/json' -d @samples/patient_low_risk.json; echo

step "POST /predict  (partial input, missing fields imputed)"
curl -sf -X POST "$BASE/predict" -H 'Content-Type: application/json' -d @samples/patient_partial.json; echo

step "POST /predict  (invalid input -> expect 422)"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/predict" \
  -H 'Content-Type: application/json' -d '{"age": 63, "sex": 1, "cp": 9}')
echo "HTTP $code"
[ "$code" = "422" ] || fail "expected 422, got $code"

step "GET /metrics (excerpt)"
curl -sf "$BASE/metrics" | grep -E '^(predictions_total|http_requests_total|model_info)'

step "Container logs (structured JSON)"
$ENGINE logs "$NAME" 2>&1 | tail -8

printf '\n\033[32mSMOKE TEST PASSED\033[0m  image=%s\n' "$IMAGE"
