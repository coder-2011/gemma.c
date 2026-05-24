#!/usr/bin/env bash
set -euo pipefail

CODEX_BIN="${CODEX_APP_SERVER_BIN:-$(command -v codex)}"
if [[ -z "${CODEX_BIN}" ]]; then
  echo "codex binary not found in PATH"
  exit 1
fi

LISTEN_ADDR="${CODEX_APP_SERVER_LISTEN:-ws://0.0.0.0:8765}"
WS_AUTH_MODE="${CODEX_APP_SERVER_WS_AUTH_MODE:-capability-token}"
TOKEN_FILE="${CODEX_APP_SERVER_TOKEN_FILE:-$HOME/.codex/app-server-control/remote-token}"
SHARED_SECRET_FILE="${CODEX_APP_SERVER_SHARED_SECRET_FILE:-}"
ISSUER="${CODEX_APP_SERVER_WS_ISSUER:-codex-app-server}"
AUDIENCE="${CODEX_APP_SERVER_WS_AUDIENCE:-codex}"
MAX_CLOCK_SKEW_SECONDS="${CODEX_APP_SERVER_WS_MAX_CLOCK_SKEW_SECONDS:-30}"

if [[ "${WS_AUTH_MODE}" != "capability-token" && "${WS_AUTH_MODE}" != "signed-bearer-token" ]]; then
  echo "Unsupported WS_AUTH_MODE: ${WS_AUTH_MODE}. Use capability-token or signed-bearer-token."
  exit 1
fi

mkdir -p "$(dirname "${TOKEN_FILE}")"
if [[ ! -f "${TOKEN_FILE}" && "${WS_AUTH_MODE}" == "capability-token" ]]; then
  openssl rand -hex 32 > "${TOKEN_FILE}"
  chmod 600 "${TOKEN_FILE}"
fi

ARGS=(app-server --remote-control --listen "${LISTEN_ADDR}" --ws-auth "${WS_AUTH_MODE}")

if [[ "${WS_AUTH_MODE}" == "capability-token" ]]; then
  ARGS+=(--ws-token-file "${TOKEN_FILE}")
else
  if [[ ! -f "${SHARED_SECRET_FILE}" ]]; then
    echo "WS shared-secret file missing: ${SHARED_SECRET_FILE}"
    exit 1
  fi
  ARGS+=(--ws-shared-secret-file "${SHARED_SECRET_FILE}")
  ARGS+=(--ws-issuer "${ISSUER}")
  ARGS+=(--ws-audience "${AUDIENCE}")
  ARGS+=(--ws-max-clock-skew-seconds "${MAX_CLOCK_SKEW_SECONDS}")
fi

exec "${CODEX_BIN}" "${ARGS[@]}"
