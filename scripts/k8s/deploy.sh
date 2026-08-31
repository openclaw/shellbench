#!/usr/bin/env bash
# Deploy ClawBench evals on Kubernetes (works on OpenShift too).
#
# 0-to-hero pipeline:
#   Step 0: Create a cluster (see --help for Kind instructions)
#   Step 1: Deploy OpenClaw gateway         (optional — bring your own)
#   Step 2: Deploy MLflow tracking server   (optional — bring your own)
#   Step 3: Run evals via sidecar           (add / remove)
#
# Usage:
#   ./scripts/k8s/deploy.sh                        # Full deploy: OpenClaw + MLflow + eval
#   ./scripts/k8s/deploy.sh --openclaw-only         # Step 1: deploy OpenClaw gateway
#   ./scripts/k8s/deploy.sh --mlflow-only           # Step 2: deploy MLflow
#   ./scripts/k8s/deploy.sh --add-sidecar           # Step 3: add eval sidecar (starts eval)
#   ./scripts/k8s/deploy.sh --remove-sidecar        # Step 3: remove eval sidecar
#   ./scripts/k8s/deploy.sh --logs                  # Tail clawbench sidecar logs
#   ./scripts/k8s/deploy.sh --teardown              # Delete eval namespace (keeps MLflow)
#
# Environment (required):
#   CLAWBENCH_NAMESPACE            Namespace for OpenClaw + eval
#   OPENAI_API_KEY                 Model provider API key (or another provider key)
#
# Environment (optional):
#   CLAWBENCH_IMAGE                Clawbench image (default: quay.io/sallyom/clawbench:latest)
#   OPENCLAW_IMAGE                 OpenClaw image (default: ghcr.io/openclaw/openclaw:latest)
#   OPENCLAW_GATEWAY_TOKEN         Existing gateway token (generated if unset)
#   CLAWBENCH_MODEL                Model to eval (default: openai/gpt-5.5)
#   MLFLOW_NAMESPACE               MLflow namespace (default: mlflow)
#   MLFLOW_TRACKING_URI            External MLflow URI (skips MLflow deploy if set)
#   MLFLOW_EXPERIMENT_ID           MLflow experiment ID
#   MLFLOW_EXPERIMENT_NAME         MLflow experiment name
#   MLFLOW_IMAGE                   MLflow image (default: ghcr.io/mlflow/mlflow:v3.15.2)
#   ANTHROPIC_API_KEY              Anthropic key (added to secret if set)
#   OPENROUTER_API_KEY             OpenRouter key (added to secret if set)
#   GEMINI_API_KEY                 Gemini key (added to secret if set)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS="${CLAWBENCH_NAMESPACE:-}"
MLFLOW_NS="${MLFLOW_NAMESPACE:-mlflow}"
CLAWBENCH_IMG="${CLAWBENCH_IMAGE:-quay.io/sallyom/clawbench:latest}"
OPENCLAW_IMG="${OPENCLAW_IMAGE:-ghcr.io/openclaw/openclaw:latest}"
MLFLOW_IMG="${MLFLOW_IMAGE:-ghcr.io/mlflow/mlflow:v3.15.2}"

# ---------------------------------------------------------------------------
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'HELP'
ClawBench Kubernetes Deployment
===============================

0-to-hero pipeline for running ClawBench evals on Kubernetes.

  Step 0: Create a cluster
          For local testing with Kind, see:
          https://github.com/openclaw/openclaw/blob/main/docs/install/kubernetes.md#local-testing-with-kind

  Step 1: Deploy OpenClaw gateway (optional — skip if you have one)
  Step 2: Deploy MLflow tracking server (optional — skip if you have one)
  Step 3: Run evals via sidecar (add/remove to OpenClaw deployment)

Usage:
  ./scripts/k8s/deploy.sh                    Full deploy (steps 1+2+3)
  ./scripts/k8s/deploy.sh --openclaw-only     Step 1: OpenClaw only
  ./scripts/k8s/deploy.sh --mlflow-only       Step 2: MLflow only
  ./scripts/k8s/deploy.sh --add-sidecar       Step 3: add eval sidecar (starts eval)
  ./scripts/k8s/deploy.sh --remove-sidecar    Step 3: remove eval sidecar
  ./scripts/k8s/deploy.sh --logs              Tail clawbench sidecar logs
  ./scripts/k8s/deploy.sh --teardown          Delete eval namespace (keeps MLflow)

Required environment:
  CLAWBENCH_NAMESPACE          Namespace for OpenClaw + eval
  OPENAI_API_KEY               Model provider API key (or ANTHROPIC_API_KEY, etc.)

Optional environment:
  CLAWBENCH_IMAGE              Clawbench image (default: quay.io/sallyom/clawbench:latest)
  OPENCLAW_IMAGE               OpenClaw image (default: ghcr.io/openclaw/openclaw:latest)
  OPENCLAW_GATEWAY_TOKEN       Existing gateway token (generated if unset)
  CLAWBENCH_MODEL              Model to eval (default: openai/gpt-5.5)
  MLFLOW_NAMESPACE             MLflow namespace (default: mlflow)
  MLFLOW_TRACKING_URI          External MLflow URI (skips MLflow deploy)
  MLFLOW_EXPERIMENT_ID         MLflow experiment ID
  MLFLOW_EXPERIMENT_NAME       MLflow experiment name
  MLFLOW_IMAGE                 MLflow image (default: ghcr.io/mlflow/mlflow:v3.15.2)
  ANTHROPIC_API_KEY            Anthropic key (added to secret if set)
  OPENROUTER_API_KEY           OpenRouter key (added to secret if set)
  GEMINI_API_KEY               Gemini key (added to secret if set)

Works on Kubernetes and OpenShift.
HELP
  exit 0
fi

command -v kubectl &>/dev/null || { echo "Missing: kubectl" >&2; exit 1; }

if [[ -z "$NS" ]]; then
  echo "CLAWBENCH_NAMESPACE is required." >&2
  echo "  export CLAWBENCH_NAMESPACE=clawbench-eval" >&2
  exit 1
fi

MODE="full"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --openclaw-only)   MODE="openclaw-only" ;;
    --mlflow-only)     MODE="mlflow-only" ;;
    --add-sidecar)     MODE="add-sidecar" ;;
    --remove-sidecar)  MODE="remove-sidecar" ;;
    --logs)            MODE="logs" ;;
    --teardown)        MODE="teardown" ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

kubectl cluster-info &>/dev/null || { echo "Cannot connect to cluster. Check kubeconfig." >&2; exit 1; }

# ---------------------------------------------------------------------------
# --logs
# ---------------------------------------------------------------------------
if [[ "$MODE" == "logs" ]]; then
  kubectl logs deploy/openclaw -c clawbench -n "$NS" -f
  exit 0
fi

# ---------------------------------------------------------------------------
# --teardown
# ---------------------------------------------------------------------------
if [[ "$MODE" == "teardown" ]]; then
  echo "Deleting namespace '$NS'..."
  kubectl delete namespace "$NS" --ignore-not-found
  echo "Done. MLflow namespace '$MLFLOW_NS' was not deleted."
  exit 0
fi

# ---------------------------------------------------------------------------
# --remove-sidecar
# ---------------------------------------------------------------------------
if [[ "$MODE" == "remove-sidecar" ]]; then
  echo "Removing clawbench sidecar from openclaw in namespace '$NS'..."
  INDEX=$(kubectl get deploy/openclaw -n "$NS" -o json \
    | python3 -c "import json,sys; cs=json.load(sys.stdin)['spec']['template']['spec']['containers']; print(next((i for i,c in enumerate(cs) if c['name']=='clawbench'),-1))")
  if [[ "$INDEX" == "-1" ]]; then
    echo "No clawbench sidecar found."
  else
    kubectl patch deploy/openclaw -n "$NS" --type=json \
      -p "[{\"op\":\"remove\",\"path\":\"/spec/template/spec/containers/$INDEX\"}]"
    echo "Sidecar removed."
  fi
  exit 0
fi

# ---------------------------------------------------------------------------
# Create namespace + secret
# ---------------------------------------------------------------------------
ensure_namespace_and_secret() {
  if ! kubectl get namespace "$NS" &>/dev/null; then
    echo "Creating namespace '$NS'..."
    kubectl create namespace "$NS"
  fi

  if ! kubectl get secret clawbench-secrets -n "$NS" &>/dev/null; then
    echo "Creating clawbench-secrets..."
    if [[ -n "${OPENCLAW_GATEWAY_TOKEN:-}" ]]; then
      GATEWAY_TOKEN="$OPENCLAW_GATEWAY_TOKEN"
      GATEWAY_TOKEN_SOURCE="from OPENCLAW_GATEWAY_TOKEN"
    else
      GATEWAY_TOKEN=$(python3 -c "import secrets,base64; print(base64.b64encode(secrets.token_bytes(32)).decode())")
      GATEWAY_TOKEN_SOURCE="generated"
    fi

    SECRET_ARGS=(
      --from-literal=OPENCLAW_GATEWAY_TOKEN="$GATEWAY_TOKEN"
    )
    [[ -n "${OPENAI_API_KEY:-}" ]] && SECRET_ARGS+=(--from-literal=OPENAI_API_KEY="$OPENAI_API_KEY")
    [[ -n "${ANTHROPIC_API_KEY:-}" ]] && SECRET_ARGS+=(--from-literal=ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY")
    [[ -n "${OPENROUTER_API_KEY:-}" ]] && SECRET_ARGS+=(--from-literal=OPENROUTER_API_KEY="$OPENROUTER_API_KEY")
    [[ -n "${GEMINI_API_KEY:-}" ]] && SECRET_ARGS+=(--from-literal=GEMINI_API_KEY="$GEMINI_API_KEY")

    if [[ ${#SECRET_ARGS[@]} -eq 1 ]]; then
      echo "Warning: No API keys provided. Set OPENAI_API_KEY or another provider key." >&2
    fi

    kubectl create secret generic clawbench-secrets -n "$NS" "${SECRET_ARGS[@]}"
    echo "  Gateway token: $GATEWAY_TOKEN_SOURCE"
    [[ -n "${OPENAI_API_KEY:-}" ]] && echo "  OPENAI_API_KEY: set"
    [[ -n "${ANTHROPIC_API_KEY:-}" ]] && echo "  ANTHROPIC_API_KEY: set"
    [[ -n "${OPENROUTER_API_KEY:-}" ]] && echo "  OPENROUTER_API_KEY: set"
    [[ -n "${GEMINI_API_KEY:-}" ]] && echo "  GEMINI_API_KEY: set"
  else
    echo "Secret clawbench-secrets already exists in '$NS'."
  fi
  return 0
}

# ---------------------------------------------------------------------------
# Step 1: Deploy OpenClaw
# ---------------------------------------------------------------------------
deploy_openclaw() {
  echo ""
  echo "Step 1: Deploying OpenClaw gateway (image: $OPENCLAW_IMG)..."

  kubectl apply -f "$SCRIPT_DIR/openclaw/configmap.yaml" -n "$NS"

  # Patch gateway config with custom OpenAI-compatible base URL
  if [[ -n "${OPENAI_API_BASE:-}" ]]; then
    echo "  Patching gateway config: models.providers.openai.baseUrl = $OPENAI_API_BASE"
    EXISTING_JSON=$(kubectl get configmap openclaw-config -n "$NS" -o jsonpath='{.data.openclaw\.json}')
    PATCHED_JSON=$(echo "$EXISTING_JSON" | python3 -c "
import json, sys, os
cfg = json.load(sys.stdin)
openai_cfg = cfg.setdefault('models', {}).setdefault('providers', {}).setdefault('openai', {})
openai_cfg['baseUrl'] = os.environ['OPENAI_API_BASE']
openai_cfg.setdefault('models', [])
json.dump(cfg, sys.stdout, indent=2)
")
    kubectl create configmap openclaw-config -n "$NS" \
      --from-literal="openclaw.json=$PATCHED_JSON" \
      --dry-run=client -o yaml | kubectl apply -f - -n "$NS" >/dev/null
  fi

  kubectl apply -f "$SCRIPT_DIR/openclaw/pvc.yaml" -n "$NS"
  kubectl apply -f "$SCRIPT_DIR/openclaw/service.yaml" -n "$NS"

  if [[ "$OPENCLAW_IMG" != "ghcr.io/openclaw/openclaw:latest" ]]; then
    kubectl apply -f "$SCRIPT_DIR/openclaw/deployment.yaml" -n "$NS"
    kubectl set image "deploy/openclaw" "gateway=$OPENCLAW_IMG" -n "$NS"
  else
    kubectl apply -f "$SCRIPT_DIR/openclaw/deployment.yaml" -n "$NS"
  fi

  echo "Waiting for OpenClaw rollout..."
  kubectl rollout status deploy/openclaw -n "$NS" --timeout=180s || \
    echo "  (rollout still in progress)"
  echo "OpenClaw deployed."
}

# ---------------------------------------------------------------------------
# Step 2: Deploy MLflow
# ---------------------------------------------------------------------------
deploy_mlflow() {
  if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
    echo ""
    echo "Step 2: Skipping MLflow deploy (MLFLOW_TRACKING_URI is set: $MLFLOW_TRACKING_URI)"
    return
  fi

  echo ""
  echo "Step 2: Deploying MLflow (namespace: $MLFLOW_NS, image: $MLFLOW_IMG)..."

  if ! kubectl get namespace "$MLFLOW_NS" &>/dev/null; then
    kubectl create namespace "$MLFLOW_NS"
  fi

  kubectl apply -f "$SCRIPT_DIR/mlflow/pvc.yaml" -n "$MLFLOW_NS"
  kubectl apply -f "$SCRIPT_DIR/mlflow/service.yaml" -n "$MLFLOW_NS"

  if [[ "$MLFLOW_IMG" != "ghcr.io/mlflow/mlflow:v3.15.2" ]]; then
    kubectl apply -f "$SCRIPT_DIR/mlflow/deployment.yaml" -n "$MLFLOW_NS"
    kubectl set image "deploy/mlflow" "mlflow=$MLFLOW_IMG" -n "$MLFLOW_NS"
  else
    kubectl apply -f "$SCRIPT_DIR/mlflow/deployment.yaml" -n "$MLFLOW_NS"
  fi

  echo "Waiting for MLflow rollout..."
  kubectl rollout status deploy/mlflow -n "$MLFLOW_NS" --timeout=120s || \
    echo "  (rollout still in progress)"

  MLFLOW_TRACKING_URI="http://mlflow-service.${MLFLOW_NS}.svc.cluster.local:5000"
  echo "MLflow deployed: $MLFLOW_TRACKING_URI"
}

# ---------------------------------------------------------------------------
# Step 3: Add clawbench sidecar (starts eval)
# ---------------------------------------------------------------------------
add_sidecar() {
  echo ""
  echo "Step 3: Adding clawbench eval sidecar..."

  echo "Applying clawbench ConfigMap..."
  kubectl apply -f "$SCRIPT_DIR/manifests/configmap.yaml" -n "$NS" >/dev/null

  if [[ -n "${CLAWBENCH_MODEL:-}" ]]; then
    kubectl patch configmap clawbench-config -n "$NS" \
      --type merge -p "{\"data\":{\"CLAWBENCH_MODEL\":\"$CLAWBENCH_MODEL\"}}" >/dev/null
    echo "  Model: $CLAWBENCH_MODEL"
  fi

  if [[ -n "${OPENAI_API_BASE:-}" ]]; then
    kubectl patch configmap clawbench-config -n "$NS" \
      --type merge -p "{\"data\":{\"OPENAI_API_BASE\":\"$OPENAI_API_BASE\"}}" >/dev/null
    echo "  OpenAI API base: $OPENAI_API_BASE"
  fi

  # Patch MLflow settings into ConfigMap
  PATCH_DATA=""
  MLFLOW_URI="${MLFLOW_TRACKING_URI:-http://mlflow-service.${MLFLOW_NS}.svc.cluster.local:5000}"
  PATCH_DATA="\"MLFLOW_TRACKING_URI\":\"$MLFLOW_URI\""
  if [[ -n "${MLFLOW_EXPERIMENT_ID:-}" ]]; then
    PATCH_DATA="$PATCH_DATA,\"MLFLOW_EXPERIMENT_ID\":\"$MLFLOW_EXPERIMENT_ID\""
  fi
  if [[ -n "${MLFLOW_EXPERIMENT_NAME:-}" ]]; then
    PATCH_DATA="$PATCH_DATA,\"MLFLOW_EXPERIMENT_NAME\":\"$MLFLOW_EXPERIMENT_NAME\""
  fi
  kubectl patch configmap clawbench-config -n "$NS" \
    --type merge -p "{\"data\":{$PATCH_DATA}}" >/dev/null
  echo "  MLflow URI: $MLFLOW_URI"
  [[ -n "${MLFLOW_EXPERIMENT_ID:-}" ]] && echo "  MLflow experiment ID: $MLFLOW_EXPERIMENT_ID"
  [[ -n "${MLFLOW_EXPERIMENT_NAME:-}" ]] && echo "  MLflow experiment name: $MLFLOW_EXPERIMENT_NAME"

  # Check if sidecar already exists
  HAS_SIDECAR=$(kubectl get deploy/openclaw -n "$NS" -o json \
    | python3 -c "import json,sys; cs=json.load(sys.stdin)['spec']['template']['spec']['containers']; print('yes' if any(c['name']=='clawbench' for c in cs) else 'no')")

  if [[ "$HAS_SIDECAR" == "yes" ]]; then
    echo "Removing existing clawbench sidecar..."
    INDEX=$(kubectl get deploy/openclaw -n "$NS" -o json \
      | python3 -c "import json,sys; cs=json.load(sys.stdin)['spec']['template']['spec']['containers']; print(next(i for i,c in enumerate(cs) if c['name']=='clawbench'))")
    kubectl patch deploy/openclaw -n "$NS" --type=json \
      -p "[{\"op\":\"remove\",\"path\":\"/spec/template/spec/containers/$INDEX\"}]" >/dev/null
  fi

  # Find the OpenClaw home volume, and capture existing volumes so add-sidecar
  # also works with bring-your-own deployments that lack this repo's PVC layout.
  VOLUME_INFO=$(kubectl get deploy/openclaw -n "$NS" -o json \
    | python3 -c "
import json, sys
spec = json.load(sys.stdin)['spec']['template']['spec']
volume_names = [v.get('name') for v in spec.get('volumes', []) if v.get('name')]
home_volume = 'openclaw-home'
for c in spec['containers']:
    if c['name'] == 'gateway':
        for vm in c.get('volumeMounts', []):
            if vm['mountPath'] == '/home/node/.openclaw':
                home_volume = vm['name']
                break
print(json.dumps({
    'home_volume': home_volume,
    'volumes_present': 'volumes' in spec,
    'volume_names': volume_names,
}))
")

  echo "Adding clawbench sidecar (image: $CLAWBENCH_IMG)..."

  PATCH=$(VOLUME_INFO="$VOLUME_INFO" CLAWBENCH_IMG="$CLAWBENCH_IMG" python3 - <<'PY'
import json
import os

info = json.loads(os.environ["VOLUME_INFO"])
home_volume = info["home_volume"]

command = r"""echo "Waiting for gateway on localhost:18789..."
for i in $(seq 1 90); do
  python3 -c "import socket; s=socket.create_connection((\"127.0.0.1\",18789),2); s.close()" 2>/dev/null && echo "Gateway ready" && break
  sleep 2
done

if [ -n "${MLFLOW_TRACKING_URI:-}" ]; then
  echo "Checking MLflow at ${MLFLOW_TRACKING_URI}..."
  python3 -c "import httpx,os; r=httpx.get(os.environ[\"MLFLOW_TRACKING_URI\"]+\"/health\"); print(\"MLflow OK:\",r.status_code)" 2>&1 || echo "MLflow pre-check failed (will retry at log time)"
fi

echo "Starting eval..."
clawbench run \
  --model "${CLAWBENCH_MODEL}" \
  --gateway-token "${OPENCLAW_GATEWAY_TOKEN}" \
  --runs "${CLAWBENCH_RUNS}" \
  --concurrency "${CLAWBENCH_CONCURRENCY}" \
  ${CLAWBENCH_JUDGE_MODEL:+--judge-model "${CLAWBENCH_JUDGE_MODEL}"} \
  $([ -n "${CLAWBENCH_TASKS:-}" ] && for t in ${CLAWBENCH_TASKS}; do printf -- "-t %s " "$t"; done) \
  -o /results/benchmark.json
RC=$?
if [ $RC -eq 0 ] && [ -n "${MLFLOW_TRACKING_URI:-}" ]; then
  python scripts/log_to_mlflow.py /results/benchmark.json
fi
echo "ClawBench finished (exit=$RC)"
sleep infinity"""

container = {
    "name": "clawbench",
    "image": os.environ["CLAWBENCH_IMG"],
    "imagePullPolicy": "IfNotPresent",
    "command": ["/bin/bash", "-c", command],
    "envFrom": [{"configMapRef": {"name": "clawbench-config"}}],
    "env": [
        {
            "name": "OPENCLAW_GATEWAY_TOKEN",
            "valueFrom": {
                "secretKeyRef": {
                    "name": "clawbench-secrets",
                    "key": "OPENCLAW_GATEWAY_TOKEN",
                }
            },
        }
    ],
    "resources": {
        "requests": {"memory": "1Gi", "cpu": "500m"},
        "limits": {"memory": "4Gi", "cpu": "2"},
    },
    "volumeMounts": [
        {"name": home_volume, "mountPath": "/home/node/.openclaw"},
        {"name": "clawbench-results", "mountPath": "/results"},
        {"name": "tmp-volume", "mountPath": "/tmp"},
    ],
    "securityContext": {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
    },
}

patch = [{"op": "add", "path": "/spec/template/spec/containers/-", "value": container}]

existing_volumes = set(info["volume_names"])
required_volumes = [
    {"name": home_volume, "emptyDir": {}},
    {"name": "clawbench-results", "emptyDir": {}},
    {"name": "tmp-volume", "emptyDir": {}},
]
missing_volumes = []
for volume in required_volumes:
    if volume["name"] not in existing_volumes and volume["name"] not in {
        item["name"] for item in missing_volumes
    }:
        missing_volumes.append(volume)

if missing_volumes:
    if info["volumes_present"]:
        patch.extend(
            {"op": "add", "path": "/spec/template/spec/volumes/-", "value": volume}
            for volume in missing_volumes
        )
    else:
        patch.append(
            {"op": "add", "path": "/spec/template/spec/volumes", "value": missing_volumes}
        )

print(json.dumps(patch))
PY
)

  kubectl patch deploy/openclaw -n "$NS" --type=json -p "$PATCH" >/dev/null

  echo ""
  echo "Waiting for rollout..."
  kubectl rollout status deploy/openclaw -n "$NS" --timeout=300s 2>/dev/null || \
    echo "  (rollout timeout — eval runs for 30-60 min)"

  echo ""
  echo "Eval is running. Follow logs with:"
  echo "  ./scripts/k8s/deploy.sh --logs"
  echo ""
  echo "When finished, remove the sidecar with:"
  echo "  ./scripts/k8s/deploy.sh --remove-sidecar"
}

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
case "$MODE" in
  full)
    ensure_namespace_and_secret
    deploy_openclaw
    deploy_mlflow
    add_sidecar
    ;;
  openclaw-only)
    ensure_namespace_and_secret
    deploy_openclaw
    echo ""
    echo "OpenClaw is running. Next steps:"
    echo "  ./scripts/k8s/deploy.sh --mlflow-only       # Deploy MLflow"
    echo "  ./scripts/k8s/deploy.sh --add-sidecar       # Start eval"
    ;;
  mlflow-only)
    deploy_mlflow
    ;;
  add-sidecar)
    if ! kubectl get deploy/openclaw -n "$NS" &>/dev/null; then
      echo "Deployment 'openclaw' not found in namespace '$NS'." >&2
      echo "Deploy OpenClaw first with: ./scripts/k8s/deploy.sh --openclaw-only" >&2
      exit 1
    fi
    ensure_namespace_and_secret
    add_sidecar
    ;;
esac
