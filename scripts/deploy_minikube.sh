#!/usr/bin/env bash
# Build the API image with Podman, load it into Minikube and deploy with kubectl.
# Usage: scripts/deploy_minikube.sh            (build + load + deploy)
#        SKIP_BUILD=1 scripts/deploy_minikube.sh
set -euo pipefail

IMAGE="${IMAGE:-localhost/heart-api:1.0}"
NS="heart-mlops"
cd "$(dirname "$0")/.."

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

step "Cluster"
minikube status
kubectl config use-context minikube >/dev/null
kubectl get nodes -o wide

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  step "Build image $IMAGE"
  podman build --format docker -f docker/Dockerfile -t "$IMAGE" .
fi

step "Load image into Minikube (no registry needed)"
TAR="$(mktemp -t heart-api).tar"
podman save -o "$TAR" "$IMAGE"
minikube image load "$TAR"
rm -f "$TAR"
minikube image ls | grep heart-api

step "Apply manifests (k8s/)"
kubectl apply -k k8s/
# Same tag may now point to a new image: restart so pods pick it up.
kubectl -n "$NS" rollout restart deployment/heart-api
kubectl -n "$NS" rollout status deployment/heart-api --timeout=180s

step "Resources"
kubectl -n "$NS" get deploy,rs,pods,svc,ingress,hpa,pdb -o wide

IP="$(minikube ip)"
cat <<MSG

Deployed. Ingress host: heart.local -> $IP
  Without editing /etc/hosts:  curl -H 'Host: heart.local' http://$IP/health
  Or add once:                 echo "$IP heart.local" | sudo tee -a /etc/hosts
  Then verify:                 scripts/verify_deployment.sh
MSG
