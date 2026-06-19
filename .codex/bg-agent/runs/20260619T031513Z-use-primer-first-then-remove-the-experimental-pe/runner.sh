#!/usr/bin/env bash
set +e
cd /home/ubuntu/gemma.c
printf '{"status":"running","startedAt":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > /home/ubuntu/gemma.c/.codex/bg-agent/runs/20260619T031513Z-use-primer-first-then-remove-the-experimental-pe/status.json
codex exec --json -C /home/ubuntu/gemma.c -c 'model_reasoning_effort="high"' -s workspace-write -o /home/ubuntu/gemma.c/.codex/bg-agent/runs/20260619T031513Z-use-primer-first-then-remove-the-experimental-pe/last_message.md - < /home/ubuntu/gemma.c/.codex/bg-agent/runs/20260619T031513Z-use-primer-first-then-remove-the-experimental-pe/prompt.md > /home/ubuntu/gemma.c/.codex/bg-agent/runs/20260619T031513Z-use-primer-first-then-remove-the-experimental-pe/stdout.jsonl 2> /home/ubuntu/gemma.c/.codex/bg-agent/runs/20260619T031513Z-use-primer-first-then-remove-the-experimental-pe/stderr.log
exit_code=$?
printf '{"status":"exited","exitCode":%s,"finishedAt":"%s"}\n' "$exit_code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > /home/ubuntu/gemma.c/.codex/bg-agent/runs/20260619T031513Z-use-primer-first-then-remove-the-experimental-pe/status.json
exit "$exit_code"
