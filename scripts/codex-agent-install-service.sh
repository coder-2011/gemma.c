#!/usr/bin/env bash
set -euo pipefail

NAME="${1:-codex-agent}"
WORKDIR="${2:-$HOME/gemma.c}"
CODEX_COMMAND="${3:-}"
if [[ -z "${CODEX_COMMAND}" ]]; then
  if ! CODEX_COMMAND="$(command -v codex)"; then
    echo "Could not find codex binary. Set CODEX_COMMAND as third argument."
    exit 1
  fi
fi
RESTART_DELAY="${4:-5}"

CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SERVICE_PATH="${CONFIG_DIR}/${NAME}@.service"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
RUNNER_PATH="${WORK_DIR}/scripts/codex-agent-runner.sh"

mkdir -p "${CONFIG_DIR}"

cat > "${SERVICE_PATH}" <<EOF
[Unit]
Description=Codex agent service (%i)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Restart=always
RestartSec=${RESTART_DELAY}
Environment="CODEX_AGENT_NAME=%i"
Environment="CODEX_AGENT_WORKDIR=${WORKDIR}"
Environment="CODEX_AGENT_COMMAND=${CODEX_COMMAND} --no-alt-screen --yolo --ask-for-approval never --cd ${WORKDIR}"
Environment="CODEX_AGENT_RESTART_DELAY=${RESTART_DELAY}"
Environment="CODEX_AGENT_LOG_DIR=%h/.local/var/codex-agents"
ExecStart=${RUNNER_PATH}
WorkingDirectory=%h
StandardOutput=append:%h/.local/var/codex-agents/%i-service.out.log
StandardError=append:%h/.local/var/codex-agents/%i-service.err.log
KillMode=control-group

[Install]
WantedBy=default.target
EOF

if ! systemctl --user daemon-reload; then
  echo "Could not reach user systemd bus. The unit file was written to:"
  echo "  ${SERVICE_PATH}"
  echo "Please run 'systemctl --user daemon-reload' from a regular interactive session."
else
  echo "Installed ${SERVICE_PATH}"
fi
echo "Enable + start:"
echo "  systemctl --user enable --now ${NAME}@main.service"
echo
echo "View logs:"
echo "  journalctl --user -fu ${NAME}@main.service"
