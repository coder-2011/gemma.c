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

## Note

`codex` interactive mode needs a TTY. The wrapper uses `script` to provide a virtual terminal for the process
before handing it to systemd for restart handling.
