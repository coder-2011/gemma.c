#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${CODEX_AGENT_WORKDIR:-$HOME}"
COMMAND="${CODEX_AGENT_COMMAND:-}"
NAME="${CODEX_AGENT_NAME:-agent}"
LOG_DIR="${CODEX_AGENT_LOG_DIR:-$HOME/.local/var/codex-agents}"
LOG_FILE="${CODEX_AGENT_LOG_FILE:-$LOG_DIR/runner.log}"
RESTART_DELAY="${CODEX_AGENT_RESTART_DELAY:-5}"

if [[ -z "${COMMAND}" ]]; then
  echo "CODEX_AGENT_COMMAND is required. Example:"
  echo "  CODEX_AGENT_COMMAND='codex --no-alt-screen --yolo --ask-for-approval never' $0"
  exit 1
fi

mkdir -p "${LOG_DIR}"
PSEUDOTERM_DIR="${LOG_DIR}/pty"
mkdir -p "${PSEUDOTERM_DIR}"

shutdown=0
child_pid=""

log() {
  printf '[%s][%s] %s\n' "$(date -Iseconds)" "$NAME" "$*" >>"${LOG_FILE}"
}

shutdown_handler() {
  shutdown=1
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill "${child_pid}" 2>/dev/null || true
  fi
}
trap shutdown_handler SIGINT SIGTERM

while true; do
  if (( shutdown )); then
    log "shutdown requested; exiting"
    break
  fi

  run_log="$PSEUDOTERM_DIR/${NAME}-$(date +%s).log"
  log "starting codex command in ${WORKDIR}: ${COMMAND}"

  (
    cd "${WORKDIR}"
    # shellcheck disable=SC2091
    script -q -f -c "${COMMAND}" "${run_log}"
  ) >>"${LOG_FILE}" 2>&1 &

  child_pid=$!
  wait "${child_pid}"
  status=$?
  child_pid=""

  if (( shutdown )); then
    log "shutdown requested; exiting"
    break
  fi

  log "command exited with status ${status}; restarting in ${RESTART_DELAY}s"
  sleep "${RESTART_DELAY}"
done
