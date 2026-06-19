You are a detached background Codex agent launched by $bg-agent.

Working directory: /home/ubuntu/gemma.c

First use $repo-primer to gather only the repository context needed for this task.
Then perform the task below unsupervised.

Operating rules:
- Do not ask the user follow-up questions. Make reasonable, low-risk assumptions.
- Preserve unrelated human or agent changes in the worktree.
- Do not commit, push, amend, reset, checkout away changes, or run destructive git commands unless the task explicitly asks.
- Do not perform live trading, broker/account-affecting actions, or local Chrome profile automation unless the task explicitly asks in this prompt.
- Prefer existing project commands and validation.
- Leave a concise final summary with changed files, validation run, and unresolved risk.

Task:
Use -primer first. Then remove the experimental persistent sliding decode attention scheduler that was just added for the QKV-fusion exploration. The goal is to quarantine/delete that slower scheduler path so the repo returns to the cleaner plan: projection/prep work will be redesigned around a smaller producer setup, while the existing direct sliding paged decode attention remains the production path.

Concrete scope:
- Remove the persistent sliding decode public API from src/gemma4_flash_attention.cuh: gemma4_flash_attention_sliding_decode_persistent_scratch_i32 and gemma4_flash_attention_sliding_decode_paged_persistent_bf16.
- Remove the persistent scheduler implementation from src/gemma4_flash_attention.cu: scratch sizing helper, queue/task constants/helpers, init kernel, sliding_decode_paged_persistent_worker_kernel, launch_sliding_decode_paged_persistent, and C ABI wrappers. Keep the existing direct split kernel, reduce kernel, valid_sliding_decode_paged_args, launch_sliding_decode_paged, and gemma4_flash_attention_sliding_decode_paged_bf16 intact.
- Remove persistent-path allocations/calls/comparisons from tests/test_kv_cache.cu, while keeping all direct sliding flash decode correctness tests intact.
- Remove the persistent-path allocation/correctness/timing line from src/experiments/gemma4_kv_cache_bench.cu, while keeping direct attention and write+attention timings intact.
- Add a short follow-up note to src/experiments/EXPERIMENTS.md under the existing 2026-06-19 persistent work queue entry or as a new dated entry saying the persistent attention-only scheduler was removed/quarantined because it was only scaffolding and was +85.92% slower as attention-only; future work should implement fused x@Wqkv producer/prep separately and compare the full envelope.

Validation:
- Run git diff --check.
- Build and run the focused KV cache test if nvcc is available at /usr/local/cuda/bin/nvcc: make -B build/tests/test_kv_cache NVCC=/usr/local/cuda/bin/nvcc && ./build/tests/test_kv_cache.
- If build cannot run, report why.

Safety constraints:
- Preserve unrelated user/agent changes. Do not touch TODO.md. Do not commit or push. Do not revert unrelated files. Do not delete the existing direct sliding paged decode path. Do not remove the experiment log entirely; leave the historical result plus the removal note.

Finish with a concise summary of files changed and validation results.
