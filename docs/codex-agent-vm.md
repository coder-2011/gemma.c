# Run Codex Agents on a VM Without tmux

This repo now includes a small watchdog wrapper so your Codex agent keeps running after SSH disconnect and
restarts automatically when it crashes.

## 1) Install the service

From the project root:

```bash
cd /home/ubuntu/gemma.c
./scripts/codex-agent-install-service.sh
```

This creates `~/.config/systemd/user/codex-agent@.service` with a default command using your current
`codex` binary plus:

- `--yolo`
- `--ask-for-approval never`
- `--no-alt-screen`
- `--cd /home/ubuntu/gemma.c`

If you want a different startup command, edit the generated service line:

```ini
Environment="CODEX_AGENT_COMMAND=<your exact command>"
```

Then restart the unit.

## 2) Start it

```bash
systemctl --user enable --now codex-agent@main.service
```

## 3) Manage

- Check status: `systemctl --user status codex-agent@main.service`
- Watch logs: `journalctl --user -fu codex-agent@main.service`
- Stop without losing state: `systemctl --user stop codex-agent@main.service`

All output is also written to:

- `~/.local/var/codex-agents/runner.log` (supervisor loop + exit codes)
- `~/.local/var/codex-agents/pty/main-*.log` (per-session terminal PTY captures)

## 4) Enable Remote Control Endpoint

Use this to control your VM Codex from another device (phone/laptop) over SSH tunnel or LAN.

```bash
cd /home/ubuntu/gemma.c
./scripts/codex-app-server-install-service.sh
```

That writes `~/.config/systemd/user/codex-remote@.service` with:

- `CODEX_APP_SERVER_LISTEN=ws://0.0.0.0:8765`
- `CODEX_APP_SERVER_WS_AUTH_MODE=capability-token`
- `CODEX_APP_SERVER_TOKEN_FILE=~/.codex/app-server-control/remote-token`

Start it:

```bash
systemctl --user enable --now codex-remote@main.service
```

> If `systemctl --user` fails in this shell, run from a normal interactive login shell first (user service bus must be active).

Check it:

```bash
systemctl --user status codex-remote@main.service
journalctl --user -u codex-remote@main.service --since "5 minutes ago"
```

## 5) Confirm remote-control works

From the VM:

```bash
TOKEN="$(cat ~/.codex/app-server-control/remote-token)"
CODEX_REMOTE_AUTH_TOKEN="$TOKEN" \
  codex --remote ws://127.0.0.1:8765 --remote-auth-token-env CODEX_REMOTE_AUTH_TOKEN --version
```

You should see `codex-cli 0.133.0` (or your installed version).  
Then from phone on same network/VPN:

```bash
# 1) fetch token securely from your server
scp ubuntu@VM_IP:~/.codex/app-server-control/remote-token .

# 2) set env and connect with your own CLI/terminal client
export CODEX_REMOTE_AUTH_TOKEN="$(cat remote-token)"
codex --remote ws://VM_IP:8765 --remote-auth-token-env CODEX_REMOTE_AUTH_TOKEN
```

If you expose this endpoint on the public internet, put Codex behind TLS and firewall rules first.

## Note

`codex` interactive mode needs a TTY. The wrapper uses `script` to provide a virtual terminal for the process
before handing it to systemd for restart handling.
