#!/usr/bin/env bash
set -euo pipefail

TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-/opt/shellbench-native}"
NODE_VERSION="${NODE_VERSION:-22.23.1}"
OPENCLAW_VERSION="${OPENCLAW_VERSION:-2026.7.1-2}"
CODEX_VERSION="${CODEX_VERSION:-0.145.0}"
CLAUDE_CODE_VERSION="${CLAUDE_CODE_VERSION:-2.1.220}"
HERMES_COMMIT="${HERMES_COMMIT:-cb06017b1d6e1b9ae0cb35f99a48ffa6bcbaa828}"
LITELLM_VERSION="${LITELLM_VERSION:-1.93.0}"
OPENCLAW_PACKAGE_TARBALL="${OPENCLAW_PACKAGE_TARBALL:-}"
OPENCLAW_PACKAGE_SHA256="${OPENCLAW_PACKAGE_SHA256:-}"
OPENCLAW_PACKAGE_VERSION="${OPENCLAW_PACKAGE_VERSION:-}"
SHELLBENCH_HARNESS="${SHELLBENCH_HARNESS:-all}"

if [[ "$(id -u)" -ne 0 ]]; then
  exec sudo -E env \
    TOOLCHAIN_ROOT="$TOOLCHAIN_ROOT" \
    NODE_VERSION="$NODE_VERSION" \
    OPENCLAW_VERSION="$OPENCLAW_VERSION" \
    CODEX_VERSION="$CODEX_VERSION" \
    CLAUDE_CODE_VERSION="$CLAUDE_CODE_VERSION" \
    HERMES_COMMIT="$HERMES_COMMIT" \
    LITELLM_VERSION="$LITELLM_VERSION" \
    OPENCLAW_PACKAGE_TARBALL="$OPENCLAW_PACKAGE_TARBALL" \
    OPENCLAW_PACKAGE_SHA256="$OPENCLAW_PACKAGE_SHA256" \
    OPENCLAW_PACKAGE_VERSION="$OPENCLAW_PACKAGE_VERSION" \
    SHELLBENCH_HARNESS="$SHELLBENCH_HARNESS" \
    bash "$0" "$@"
fi

export DEBIAN_FRONTEND=noninteractive
install -d -m 0755 "$TOOLCHAIN_ROOT"

harness_enabled() {
  [[ "$SHELLBENCH_HARNESS" == "all" || "$SHELLBENCH_HARNESS" == "$1" ]]
}

case "$SHELLBENCH_HARNESS" in
  all | openclaw | codex | claude-code | hermes) ;;
  *)
    printf 'unsupported SHELLBENCH_HARNESS: %s\n' "$SHELLBENCH_HARNESS" >&2
    exit 2
    ;;
esac

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

install_node_tools() {
  local node_root="$TOOLCHAIN_ROOT/node"
  local openclaw_spec="openclaw@$OPENCLAW_VERSION"
  local -a packages=()
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
  if [[ -n "$OPENCLAW_PACKAGE_TARBALL" ]]; then
    [[ -n "$OPENCLAW_PACKAGE_SHA256" && -n "$OPENCLAW_PACKAGE_VERSION" ]]
    [[ -f "$OPENCLAW_PACKAGE_TARBALL" ]]
    printf '%s  %s\n' \
      "$OPENCLAW_PACKAGE_SHA256" "$OPENCLAW_PACKAGE_TARBALL" \
      | sha256sum -c -
    [[ "$(tar -xOf "$OPENCLAW_PACKAGE_TARBALL" package/package.json | jq -r '.name')" == "openclaw" ]]
    [[ "$(tar -xOf "$OPENCLAW_PACKAGE_TARBALL" package/package.json | jq -r '.version')" == "$OPENCLAW_PACKAGE_VERSION" ]]
    openclaw_spec="$OPENCLAW_PACKAGE_TARBALL"
  fi
  if harness_enabled openclaw; then
    packages+=("$openclaw_spec")
  fi
  if harness_enabled codex; then
    packages+=("@openai/codex@$CODEX_VERSION")
  fi
  if harness_enabled claude-code; then
    packages+=("@anthropic-ai/claude-code@$CLAUDE_CODE_VERSION")
  fi
  rm -rf "$TOOLCHAIN_ROOT/npm-packages"
  install -d -m 0755 "$TOOLCHAIN_ROOT/npm-packages"
  if ((${#packages[@]})); then
    npm install --prefix "$TOOLCHAIN_ROOT/npm-packages" "${packages[@]}"
  fi
}

install_uv() {
  install -d -m 0755 "$TOOLCHAIN_ROOT/bin"
  if [[ ! -x "$TOOLCHAIN_ROOT/bin/uv" ]]; then
    curl -LsSf --retry 5 --retry-delay 2 --retry-all-errors --retry-max-time 60 \
      https://astral.sh/uv/install.sh \
      -o /tmp/install-uv.sh
    UV_UNMANAGED_INSTALL="$TOOLCHAIN_ROOT/bin" sh /tmp/install-uv.sh
    rm -f /tmp/install-uv.sh
  fi
}

install_hermes() {
  export HOME="$TOOLCHAIN_ROOT/home"
  export HERMES_HOME="$TOOLCHAIN_ROOT/hermes-home"
  export HERMES_INSTALL_DIR="$TOOLCHAIN_ROOT/hermes-agent"
  export UV_PYTHON_INSTALL_DIR="$TOOLCHAIN_ROOT/uv-python"
  export UV_CACHE_DIR="$TOOLCHAIN_ROOT/uv-cache"
  export PATH="$TOOLCHAIN_ROOT/node/bin:$TOOLCHAIN_ROOT/bin:$PATH"
  install -d -m 0755 "$HOME" "$HERMES_HOME"

  curl -fsSL --retry 5 --retry-delay 2 --retry-all-errors --retry-max-time 60 \
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
  local python_root="$TOOLCHAIN_ROOT/uv-python"
  local python_bin
  UV_PYTHON_INSTALL_DIR="$python_root" \
    "$TOOLCHAIN_ROOT/bin/uv" python install 3.12
  python_bin="$(
    UV_PYTHON_INSTALL_DIR="$python_root" \
      "$TOOLCHAIN_ROOT/bin/uv" python find --managed-python 3.12
  )"
  "$TOOLCHAIN_ROOT/bin/uv" venv --clear --python "$python_bin" "$venv"
  "$TOOLCHAIN_ROOT/bin/uv" pip install \
    --python "$venv/bin/python" \
    "litellm[proxy]==$LITELLM_VERSION"
}

write_manifest() {
  local openclaw=""
  local codex=""
  local claude_code=""
  local hermes=""
  local openclaw_source_kind="registry"
  local openclaw_package_version="$OPENCLAW_VERSION"
  local openclaw_package_sha256=""
  local openclaw_artifact_filename=""
  if [[ -n "$OPENCLAW_PACKAGE_TARBALL" ]]; then
    openclaw_source_kind="npm_tarball"
    openclaw_package_version="$OPENCLAW_PACKAGE_VERSION"
    openclaw_package_sha256="$OPENCLAW_PACKAGE_SHA256"
    openclaw_artifact_filename="$(basename "$OPENCLAW_PACKAGE_TARBALL")"
  fi
  export PATH="$TOOLCHAIN_ROOT/node/bin:$TOOLCHAIN_ROOT/npm-packages/node_modules/.bin:$TOOLCHAIN_ROOT/home/.local/bin:$PATH"
  if harness_enabled openclaw; then
    openclaw="$(openclaw --version | head -1)"
  fi
  if harness_enabled codex; then
    codex="$(codex --version | head -1)"
  fi
  if harness_enabled claude-code; then
    claude_code="$(claude --version | head -1)"
  fi
  if harness_enabled hermes; then
    hermes="$(hermes version | head -1)"
  fi
  jq -n \
    --arg created_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg node "$(node --version)" \
    --arg harness_scope "$SHELLBENCH_HARNESS" \
    --arg openclaw "$openclaw" \
    --arg codex "$codex" \
    --arg claude_code "$claude_code" \
    --arg hermes "$hermes" \
    --arg litellm "$("$TOOLCHAIN_ROOT/litellm-venv/bin/python" -c 'from importlib.metadata import version; print(version("litellm"))')" \
    --arg hermes_commit "$HERMES_COMMIT" \
    --arg openclaw_source_kind "$openclaw_source_kind" \
    --arg openclaw_package_version "$openclaw_package_version" \
    --arg openclaw_package_sha256 "$openclaw_package_sha256" \
    --arg openclaw_artifact_filename "$openclaw_artifact_filename" \
    '{
      created_at_utc: $created_at,
      harness_scope: $harness_scope,
      node: $node,
      openclaw: (if $openclaw == "" then null else $openclaw end),
      openclaw_package: (if $openclaw == "" then null else {
        source_kind: $openclaw_source_kind,
        package_name: "openclaw",
        package_version: $openclaw_package_version,
        sha256: (if $openclaw_package_sha256 == "" then null else $openclaw_package_sha256 end),
        artifact_filename: (if $openclaw_artifact_filename == "" then null else $openclaw_artifact_filename end)
      } end),
      codex: (if $codex == "" then null else $codex end),
      claude_code: (if $claude_code == "" then null else $claude_code end),
      hermes: (if $hermes == "" then null else $hermes end),
      hermes_commit: (if $hermes == "" then null else $hermes_commit end),
      litellm: $litellm
    }' > "$TOOLCHAIN_ROOT/manifest.json"
  chmod -R a+rX "$TOOLCHAIN_ROOT"
}

install_base_packages
install_docker
install_node_tools
install_uv
if harness_enabled hermes; then
  install_hermes
fi
install_litellm
write_manifest

docker version --format '{{.Server.Version}}'
docker compose version
jq . "$TOOLCHAIN_ROOT/manifest.json"
