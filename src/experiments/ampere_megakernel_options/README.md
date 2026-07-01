# Ampere Megakernel Options

Scope: Gemma 4 12B batch-1 decode on RTX A6000-class Ampere GPUs.

This note ranks the Ampere-portable pieces of the Hazy-style megakernel idea.
It is a feasibility filter, not a benchmark result. The current measured anchor
is the July 1 decode audit: the token-level cooperative loop saved only about
`0.15 ms/token` of launch gaps, but made GPU work about `0.8 ms/token` slower.

## Verdict

Global producer/consumer counters are useful only when they narrow a real
dependency. They are not a better replacement for the inter-layer hidden-row
barrier, because the next layer needs the completed hidden row before it can
start correct QKV/FFN math.

## Actually Useful

| Idea | Usefulness | Why |
| --- | --- | --- |
| Keep per-layer cooperative kernels as the default | High | This already beat the token-level loop. Kernel boundaries provide cheap inter-layer ordering without carrying every phase's register pressure in one kernel. |
| CUDA Graph capture for the per-layer path | Medium | It attacks the measured launch gap without adding GPU-side spin loops or `grid.sync()` inside a 48-layer token kernel. The upside is bounded by the small launch-gap number. |
| Chunk-ready counters inside one layer | Medium | Useful only if a consumer can do independent work before the whole producer phase finishes. Candidate areas are split attention reduction scheduling and QKV/KV-ready to attention scheduling. |
| Preserve hidden-column O projection ownership | High | The hidden-column schedule restored decode from the bad head-owned O projection path. Do not trade it away just to remove the BF16 `attention_out` handoff. |
| `cp.async` for reused attention/KV tiles | Medium | Ampere supports global-to-shared async copies. It helps when shared-memory staging creates reuse or overlaps real work. |
| More precise split sizing for sliding/global attention | Medium | This changes ready-work granularity without inventing a persistent runtime. It is easier to benchmark than a full VM. |

## Probably Not Useful

| Idea | Usefulness | Why |
| --- | --- | --- |
| Global counters for whole-layer `l-1 -> l` handoff | Low | This replaces a correct kernel boundary or `grid.sync()` with polling and atomics, but the dependency is still whole-row completion. |
| Full persistent VM on Ampere | Low | Hazy leans on Hopper/Blackwell TMA-style movement. On A6000 this becomes manual `cp.async`, global atomics, and polling with a lot of scheduler code. |
| `cp.async` for one-use M=1 GEMV weights | Low | The worklog already found this adds shared-memory traffic and commit/wait overhead without reducing weight bytes or creating reuse. |
| Head-owned O projection to avoid `attention_out` | Low | This was the path that collapsed throughput. O projection needs hidden-column parallelism more than it needs to avoid the small BF16 handoff. |
| Token-level 48-layer cooperative loop | Low | Measured slower. It combines register footprints and adds explicit inter-layer grid barriers. |
| Per-tile atomics where no consumer overlap exists | Low | Atomics and polling only pay off when they unlock useful work. Otherwise they are just a slower barrier. |

## Maybe Later

| Idea | Usefulness | Why |
| --- | --- | --- |
| Per-SM work queues for one layer | Unknown | Could reduce idle time inside attention or projection phases, but it should be prototyped as a single-layer scheduler before any cross-layer VM. |
| Partial O accumulation from attention heads | Unknown | It could start before all heads finish, but it needs extra partial sums or atomics. The previous head-owned O regression makes this risky. |
| Multi-token or batched decode tiles | Unknown | This is the more natural route to tensor-core utilization, but it changes the latency/throughput contract. |

## Hazy Mechanism Inventory

This table lists Hazy low-latency mechanisms that are at least conceptually
doable on Ampere. "Doable" does not mean their exact code should be copied:
Hazy's implementation leans on Hopper/Blackwell TMA, while A6000 needs ordinary
global loads/stores, `cp.async` global-to-shared copies, shared-memory barriers,
and global atomics.

| Hazy mechanism | We have it? | Ampere path | Usefulness | Smallest useful version |
| --- | --- | --- | --- | --- |
| Host-built DAG of decode tasks | No | Build a static list of layer-local tasks on the host, then replay it from one experimental kernel or host launcher. | Medium | Start with one layer's QKV, attention, O, upgate, and down task graph; do not cross layers first. |
| Per-SM instruction queues | No | Use one resident CTA per SM and give each CTA a compact task range or queue. | Unknown | Prototype only for one layer and record idle time versus the current cooperative layer. |
| Persistent one-CTA-per-SM worker kernel | No | Cooperative launch with one block per SM is already supported on A6000. | Low for full token, unknown for one layer | Use a single-layer persistent worker only if a DAG prototype shows real idle gaps. |
| Warp-role specialization | No | Dedicate warps inside a CTA to load, compute, store, and control work. | Medium | Try inside attention/KV staging, where load and compute overlap can be real. |
| Dynamic instruction ring | No | Keep a tiny ring of task descriptors in shared memory. | Low | Avoid until static task lists become too rigid. |
| Shared-memory page allocator | No | Partition dynamic shared memory into fixed pages and recycle them with per-page state. | Unknown | Useful only if several staged pipelines fight for shared memory inside one kernel. |
| Per-instruction shared-memory semaphores | Partial | Use `cuda::barrier`, cooperative groups, or simple shared counters inside one CTA. | Medium | Use for local load/compute/store staging, not for layer ordering. |
| Global producer/consumer counters | No | Use global `uint32_t` counters, `atomicAdd`, polling, and `__nanosleep`. | Medium | Use only for chunk-ready dependencies where consumers can start before a full phase completes. |
| `__nanosleep` polling loops | No | Same primitive exists on Ampere. | Low by itself | Treat as a necessary cost of global counters, not an optimization. |
| Three-stage weight/input pipeline | No | Replace Hazy TMA loads with `cp.async` or vectorized global loads into shared memory. | Unknown | Test attention/KV tiles before M=1 GEMV weights. |
| Three-stage output pipeline | No | Reduce into shared memory, then store normal global outputs. | Low to Medium | Useful if stores are on the critical path; otherwise it adds coordination. |
| QKV projection chunk signals | Partial | Signal Q/K/V chunk readiness after projection, norm, RoPE, and KV write. | Medium | Let attention wait on only the Q/K/V chunks for its KV head. |
| Attention waits on exact Q/K/V chunks | No | Combine global counters with current split attention task mapping. | Medium to High | Best first counter experiment: QKV/KV-ready to attention for one layer. |
| KV tile prefetch pipeline | Partial | Stage reused K/V cache tiles with `cp.async` and double or triple buffering. | Medium | Try only where each staged tile feeds enough Q/head work to amortize shared-memory traffic. |
| Async Q load into shared memory | No | Use `cp.async` for small Q tiles if it overlaps K/V work. | Low to Medium | Likely useful only inside a staged attention loop, not as a standalone copy trick. |
| Optional attention reduction skip | Yes | The decode split kernel writes final attention output directly when `num_splits == 1`, and both standalone and fused-layer launch paths skip the reducer. | High | Keep this branch covered; do not force full-window sliding into one split unless a benchmark wins. |
| Partial attention output/LSE reduction op | Partial | Existing split attention already has partials and reduction. | Medium | Improve split sizing and reduction placement before adding a new scheduler. |
| O projection as a queued matvec task | Partial | Current O is hidden-column CTA-owned inside the layer kernel. | High to preserve, unknown to queue | If queued, keep hidden-column ownership; do not revert to head-owned O. |
| Partial O accumulation from attention heads | No | Accumulate per-head O partials into hidden-column tiles with atomics or scratch reduction. | Unknown | Only test after measuring enough overlap to pay for partial storage/reduction. |
| O projection plus residual add store | Partial | Current O path already fuses post-attention RMS/residual work. | High | Keep this fused path; only change the schedule if measured idle time remains. |
| Up/gate paired matvec instruction | Partial | Existing FFN decode is fused, but not expressed as queued up/gate tasks. | Medium | Consider only if a layer-local scheduler needs finer FFN task granularity. |
| Down projection reduction-block tasks | Partial | Existing FFN down projection is fused, but not DAG-scheduled by reduction block. | Medium | Hazy notes this could be finer; test only after up/gate chunk readiness exists. |
| Final RMS + LM head as scheduled tasks | Partial | Final sampling path is separate and already specialized. | Low | Leave alone unless the final tail becomes a measured bottleneck. |
| Per-op timing slots inside the kernel | No | Store `clock64()` deltas to a debug buffer. | High for prototypes | Add only to experimental kernels; it would make scheduler mistakes visible. |
| Fixed 16-wide output blocks | Partial | Our BF16 pack and hidden-column tiling already use fixed small tiles. | Medium | Treat block width as a tuning parameter, not a Hazy constant to copy. |
| Fixed 512-wide reduction chunks | Partial | Similar chunking is possible, but Gemma dimensions differ. | Medium | Tune against Gemma 12B hidden `3840`, FFN `15360`, and head dims `256/512`. |
| SMID-based worker identity | No | Inline PTX can read `%smid`, but block ID is simpler and more stable. | Low | Use `blockIdx.x` unless true persistent work stealing needs SM identity. |
| Stacked model parameter layouts | Partial | We already specialize layouts, but Hazy's descriptors are built for its VM. | Medium | Keep layout changes tied to one measured kernel, not a global rewrite. |

## Hazy Mechanisms To Exclude On A6000

| Hazy mechanism | Why not directly useful on Ampere |
| --- | --- |
| TMA tensor loads | Hopper/Blackwell feature; Ampere has `cp.async` global-to-shared instead. |
| TMA async stores | Hopper/Blackwell feature; Ampere stores go through normal store instructions. |
| TMA store-add / reduction | Hopper/Blackwell feature; Ampere would need explicit atomics or reductions. |
| Thread-block clusters | Compute capability 9.0+ feature; A6000 is Ampere `sm_86`. |
| Distributed shared memory | Cluster feature, not available on A6000. |
| Cluster barriers / mbarriers across CTAs | Not available as Hazy uses them; use kernel boundaries, cooperative grid sync, or global counters. |
| Register repartition between warp roles | Do not assume Hazy's role-specific register tricks are available or worthwhile on Ampere. |
| Blackwell tensor allocator paths | Not relevant to A6000. |

## Best First Experiments

| Experiment | Why it is first | Keep condition |
| --- | --- | --- |
| QKV/KV-ready to attention counters | It matches Hazy's best producer/consumer idea without changing the whole decode runtime. | Full decode-step p50 improves by at least `0.5 ms/token`, with no correctness drift. |
| `cp.async` K/V tile staging inside attention | K/V tiles are reused within attention; this is where Ampere async copy has a fair shot. | `ncu` shows lower stall time or higher useful throughput after shared-memory overhead. |
| Single-layer static task queue | It tests per-SM work queues without committing to a full persistent VM. | It beats the current cooperative layer kernel for both sliding and global layers. |
| Partial O accumulation probe | It tests the tempting overlap idea directly. | It preserves hidden-column O parallelism and beats the current BF16 `attention_out` handoff. |

## Smallest Reasonable Prototype

Do not start with a persistent megakernel. First measure one local dependency:

1. Pick one layer-local boundary with existing `grid.sync()` cost.
2. Add the narrowest possible ready counter for the producer chunks.
3. Let consumers wait only on the chunks they mathematically need.
4. Require a full decode-step p50 improvement of at least `0.5 ms/token` before
   keeping the complexity.
5. Check `ncu` for registers, spills, occupancy, DRAM throughput, and warp stalls.

If the consumer still needs the whole producer output, keep the simple barrier.
