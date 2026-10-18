#!/usr/bin/env bash
# Deploy Prometheus + Grafana into Minikube (namespace: monitoring).
# Requires the API to be deployed first (scripts/deploy_minikube.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
NS=monitoring

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

step "Images cached in Minikube (no internet needed if both are listed)"
minikube image ls | grep -E 'prom/prometheus|grafana/grafana' || \
  echo "WARNING: images not cached - Minikube will try to pull them (needs internet)"

step "Apply namespace and RBAC"
kubectl apply -f k8s/monitoring/namespace.yaml

step "Grafana admin password (Kubernetes Secret, never committed to git)"
if ! kubectl -n "$NS" get secret grafana-admin >/dev/null 2>&1; then
  kubectl -n "$NS" create secret generic grafana-admin \
    --from-literal=password="$(openssl rand -base64 18 | tr -d '/+=' | cut -c1-20)"
fi

step "Apply Prometheus, Grafana, dashboards, alerts, Ingress"
kubectl apply -k k8s/monitoring/
kubectl -n "$NS" rollout status deployment/prometheus --timeout=180s
kubectl -n "$NS" rollout status deployment/grafana --timeout=180s
kubectl -n "$NS" get pods,svc,ingress

IP="$(minikube ip)"
PASS="$(kubectl -n "$NS" get secret grafana-admin -o jsonpath='{.data.password}' | base64 --decode)"
cat <<MSG

Monitoring is up.
  1. Add hostnames once:   echo "$IP grafana.local prometheus.local" | sudo tee -a /etc/hosts
  2. Grafana:              http://grafana.local      user: admin   password: $PASS
  3. Prometheus targets:   http://prometheus.local/targets
  4. Generate traffic:     python scripts/load_test.py --url http://$IP --host heart.local --duration 300
MSG
