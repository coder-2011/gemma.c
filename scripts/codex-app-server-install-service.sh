#!/usr/bin/env bash
set -euo pipefail

NAME="${1:-codex-remote}"
CODEX_BIN="${2:-}"
LISTEN="${3:-ws://0.0.0.0:8765}"
WS_AUTH_MODE="${4:-capability-token}"
TOKEN_FILE="${5:-$HOME/.codex/app-server-control/remote-token}"
RESTART_DELAY="${6:-5}"

if [[ -z "${CODEX_BIN}" ]]; then
  if ! CODEX_BIN="$(command -v codex)"; then
    echo "Could not find codex binary. Set CODEX_BIN as second argument."
    exit 1
  fi
fi

CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SERVICE_PATH="${CONFIG_DIR}/${NAME}@.service"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
RUNNER_PATH="${WORK_DIR}/scripts/codex-app-server-runner.sh"
LOG_DIR="${HOME}/.local/var/codex-app-server"

if [[ "${WS_AUTH_MODE}" != "capability-token" && "${WS_AUTH_MODE}" != "signed-bearer-token" ]]; then
  echo "Unsupported WS auth mode: ${WS_AUTH_MODE}"
  exit 1
fi

if [[ "${WS_AUTH_MODE}" == "capability-token" ]]; then
  mkdir -p "$(dirname "${TOKEN_FILE}")"
  if [[ ! -f "${TOKEN_FILE}" ]]; then
    openssl rand -hex 32 > "${TOKEN_FILE}"
    chmod 600 "${TOKEN_FILE}"
  fi
  if [[ ! -r "${TOKEN_FILE}" ]]; then
    echo "Token file is not readable: ${TOKEN_FILE}"
    exit 1
  fi
fi

mkdir -p "${CONFIG_DIR}" "${LOG_DIR}"

cat > "${SERVICE_PATH}" <<EOF
[Unit]
Description=Codex app-server service (%i)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Restart=always
RestartSec=${RESTART_DELAY}
Environment="CODEX_APP_SERVER_LISTEN=${LISTEN}"
Environment="CODEX_APP_SERVER_WS_AUTH_MODE=${WS_AUTH_MODE}"
Environment="CODEX_APP_SERVER_TOKEN_FILE=${TOKEN_FILE}"
Environment="CODEX_APP_SERVER_BIN=${CODEX_BIN}"
ExecStart=${RUNNER_PATH}
WorkingDirectory=%h
StandardOutput=append:%h/.local/var/codex-app-server/%i-service.out.log
StandardError=append:%h/.local/var/codex-app-server/%i-service.err.log
KillMode=control-group

[Install]
WantedBy=default.target
EOF

if ! systemctl --user daemon-reload; then
  echo "Could not reach user systemd bus. The unit file was written to:"
  echo "  ${SERVICE_PATH}"
  echo "Please run: systemctl --user daemon-reload"
else
  echo "Installed ${SERVICE_PATH}"
fi

echo "Enable + start:"
echo "  systemctl --user enable --now ${NAME}@main.service"
echo
if [[ "${WS_AUTH_MODE}" == "capability-token" ]]; then
  echo "Token file:"
  echo "  ${TOKEN_FILE}"
fi
