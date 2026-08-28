#!/usr/bin/env bash
set -euo pipefail

HARNESS="${1:-}"
if [[ $# -ne 1 ]] || [[ ! "$HARNESS" =~ ^(openclaw|codex|claude-code|hermes)$ ]]; then
  echo "Usage: $0 {openclaw|codex|claude-code|hermes}" >&2
  exit 2
fi

TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-/opt/shellbench-native}"
NODE_VERSION="${NODE_VERSION:-22.23.1}"
OPENCLAW_VERSION="${OPENCLAW_VERSION:-2026.7.1-2}"
CODEX_VERSION="${CODEX_VERSION:-0.145.0}"
CLAUDE_CODE_VERSION="${CLAUDE_CODE_VERSION:-2.1.220}"
HERMES_COMMIT="${HERMES_COMMIT:-cb06017b1d6e1b9ae0cb35f99a48ffa6bcbaa828}"
LITELLM_VERSION="${LITELLM_VERSION:-1.93.0}"

if [[ "$(id -u)" -ne 0 ]]; then
  exec sudo -E bash "$0" "$@"
fi

export DEBIAN_FRONTEND=noninteractive
install -d -m 0755 "$TOOLCHAIN_ROOT"

install_base_packages() {
  apt-get update
  apt-get install -y --no-install-recommends \
    ca-certificates curl git gnupg jq openssl ripgrep rsync tar xz-utils
  rm -rf /var/lib/apt/lists/*
}

install_docker() {
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    return
  fi

  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  . /etc/os-release
  printf '%s\n' \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${UBUNTU_CODENAME:-$VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list

  printf '#!/bin/sh\nexit 101\n' > /usr/sbin/policy-rc.d
  chmod +x /usr/sbin/policy-rc.d
  apt-get update
  apt-get install -y --no-install-recommends \
    containerd.io docker-buildx-plugin docker-ce docker-ce-cli docker-compose-plugin
  rm -f /usr/sbin/policy-rc.d
  rm -rf /var/lib/apt/lists/*

  cat > /etc/sysctl.d/99-shellbench-docker.conf <<'EOF'
net.ipv4.ip_forward=1
net.ipv6.conf.all.forwarding=1
EOF
  sysctl --system >/dev/null

  install -d -m 0755 /etc/docker
  if [[ -s /etc/docker/daemon.json ]]; then
    jq '. + {"firewall-backend":"nftables"}' /etc/docker/daemon.json \
      > /etc/docker/daemon.json.tmp
    mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json
  else
    printf '%s\n' '{"firewall-backend":"nftables"}' > /etc/docker/daemon.json
  fi

  systemctl enable --now containerd docker
  chmod 0666 /var/run/docker.sock
  docker run --rm hello-world >/dev/null
}

install_node() {
  local node_root="$TOOLCHAIN_ROOT/node"
  if [[ ! -x "$node_root/bin/node" ]] || \
    [[ "$("$node_root/bin/node" --version)" != "v$NODE_VERSION" ]]; then
    rm -rf "$node_root"
    curl -fsSL \
      "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-x64.tar.xz" \
      -o /tmp/shellbench-node.tar.xz
    tar -xJf /tmp/shellbench-node.tar.xz -C "$TOOLCHAIN_ROOT"
    mv "$TOOLCHAIN_ROOT/node-v$NODE_VERSION-linux-x64" "$node_root"
    rm -f /tmp/shellbench-node.tar.xz
  fi

  export PATH="$node_root/bin:$PATH"
}

install_uv() {
  export UV_PYTHON_INSTALL_DIR="$TOOLCHAIN_ROOT/uv-python"
  export UV_CACHE_DIR="$TOOLCHAIN_ROOT/uv-cache"
  install -d -m 0755 "$TOOLCHAIN_ROOT/bin"
  if [[ ! -x "$TOOLCHAIN_ROOT/bin/uv" ]]; then
    curl -LsSf https://astral.sh/uv/install.sh -o /tmp/install-uv.sh
    UV_UNMANAGED_INSTALL="$TOOLCHAIN_ROOT/bin" sh /tmp/install-uv.sh
    rm -f /tmp/install-uv.sh
  fi
}

install_hermes() {
  export HOME="$TOOLCHAIN_ROOT/home"
  export HERMES_HOME="$TOOLCHAIN_ROOT/hermes-home"
  export HERMES_INSTALL_DIR="$TOOLCHAIN_ROOT/hermes-agent"
  export PATH="$TOOLCHAIN_ROOT/node/bin:$TOOLCHAIN_ROOT/bin:$PATH"
  install -d -m 0755 "$HOME" "$HERMES_HOME"

  curl -fsSL \
    "https://raw.githubusercontent.com/NousResearch/hermes-agent/$HERMES_COMMIT/scripts/install.sh" \
    -o /tmp/install-hermes.sh
  bash /tmp/install-hermes.sh \
    --skip-setup \
    --skip-browser \
    --commit "$HERMES_COMMIT" \
    --dir "$HERMES_INSTALL_DIR" \
    --hermes-home "$HERMES_HOME"
  rm -f /tmp/install-hermes.sh
}

install_litellm() {
  local venv="$TOOLCHAIN_ROOT/litellm-venv"
  "$TOOLCHAIN_ROOT/bin/uv" venv --clear --python 3.12 "$venv"
  "$TOOLCHAIN_ROOT/bin/uv" pip install \
    --python "$venv/bin/python" \
    "litellm[proxy]==$LITELLM_VERSION"
}

install_harness() {
  local package
  case "$HARNESS" in
    openclaw) package="openclaw@$OPENCLAW_VERSION" ;;
    codex) package="@openai/codex@$CODEX_VERSION" ;;
    claude-code) package="@anthropic-ai/claude-code@$CLAUDE_CODE_VERSION" ;;
    hermes) install_hermes; return ;;
  esac
  rm -rf "$TOOLCHAIN_ROOT/npm-packages"
  npm install --prefix "$TOOLCHAIN_ROOT/npm-packages" "$package"
}

write_manifest() {
  export PATH="$TOOLCHAIN_ROOT/node/bin:$TOOLCHAIN_ROOT/npm-packages/node_modules/.bin:$TOOLCHAIN_ROOT/home/.local/bin:$PATH"
  local harness_version
  case "$HARNESS" in
    openclaw) harness_version=$(openclaw --version) ;;
    codex) harness_version=$(codex --version) ;;
    claude-code) harness_version=$(claude --version) ;;
    hermes) harness_version=$(hermes version) ;;
  esac
  jq -n \
    --arg created_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg node "$(node --version)" \
    --arg harness "$HARNESS" \
    --arg harness_key "${HARNESS//-/_}" \
    --arg harness_version "${harness_version%%$'\n'*}" \
    --arg litellm "$("$TOOLCHAIN_ROOT/litellm-venv/bin/python" -c 'from importlib.metadata import version; print(version("litellm"))')" \
    --arg hermes_commit "$HERMES_COMMIT" \
    '{
      created_at_utc: $created_at,
      node: $node,
      harness: $harness,
      ($harness_key): $harness_version,
      litellm: $litellm
    } + (if $harness == "hermes" then {hermes_commit: $hermes_commit} else {} end)' \
    > "$TOOLCHAIN_ROOT/manifest.json"
  chmod -R a+rX "$TOOLCHAIN_ROOT"
}

install_base_packages
install_docker
install_node
install_uv
install_harness
install_litellm
write_manifest

docker version --format '{{.Server.Version}}'
docker compose version
jq . "$TOOLCHAIN_ROOT/manifest.json"
