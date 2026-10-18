#!/usr/bin/env bash
# Verify the deployed API through the Minikube Ingress.
# Works without /etc/hosts changes by sending the Host header directly.
set -euo pipefail

NS="heart-mlops"
HOST="heart.local"
BASE="http://$(minikube ip)"
cd "$(dirname "$0")/.."

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
api() { curl -sf -H "Host: $HOST" "$@"; }

step "Pods"
kubectl -n "$NS" get pods -o wide

step "Ingress"
kubectl -n "$NS" get ingress heart-api

step "GET /health   via Ingress ($BASE, Host: $HOST)"
api "$BASE/health"; echo

step "GET /model-info"
api "$BASE/model-info" | python3 -m json.tool

for f in high_risk low_risk partial; do
  step "POST /predict  samples/patient_$f.json"
  api -X POST "$BASE/predict" -H 'Content-Type: application/json' -d @"samples/patient_$f.json"; echo
done

step "POST /predict  invalid input (expect 422)"
code=$(curl -s -o /dev/null -w '%{http_code}' -H "Host: $HOST" -X POST "$BASE/predict" \
  -H 'Content-Type: application/json' -d '{"age": 63, "sex": 1, "cp": 9}')
echo "HTTP $code"; [ "$code" = "422" ]

step "Load balancing: 20 requests -> which pod answered (X-Served-By)"
for _ in $(seq 1 20); do
  curl -s -o /dev/null -D - -H "Host: $HOST" -X POST "$BASE/predict" \
    -H 'Content-Type: application/json' -d @samples/patient_high_risk.json \
    | awk 'tolower($1)=="x-served-by:" {print $2}' | tr -d '\r'
done | sort | uniq -c

step "Recent pod logs (structured JSON)"
kubectl -n "$NS" logs -l app.kubernetes.io/name=heart-api --tail=3 --prefix

printf '\n\033[32mDEPLOYMENT VERIFIED\033[0m\n'
