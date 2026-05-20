1. Maximize bytes-used / bytes-transferred for global loads; coalescing is still
     one of the biggest wins. Page 61.
  2. Keep warp lanes loading from the same 32-byte segments, even if access order
     is permuted. Page 61.
  3. Use 16-byte vectorized loads/stores where layout permits; our Packed128 style
     is aligned with this. Pages 61, 244.
  4. Prefer SoA/packed layouts that make the hot axis contiguous for the warp.
     Page 61.
  5. Avoid unnecessary global round trips by fusing memory-bound neighbors like
     residual add + RMSNorm. Page 130.
  6. Use shared memory only when it reduces global traffic or enables reuse;
     shared memory reduces usable L1. Page 54.
  7. Use constant memory for small grid-wide read-only tables that all threads
     read. Page 56.
  8. Do not reach for texture/surface memory for normal CUDA compute on current
     GPUs; guide says no benefit. Page 57.
  9. Use __ldg() for read-only global data when the compiler cannot infer read-
     only caching. Page 562.
  10. Add __restrict__ to non-aliasing kernel pointers; it enables load/common-
     subexpression optimizations. Pages 518-519.
  11. Watch the tradeoff: __restrict__ can increase register pressure by caching
     more values. Page 519.
  12. Consider L2 persisting access windows for repeatedly reused weights/tables.
     Pages 352-354.
  13. Reset persisting L2 regions when done, or they can hurt later streaming
     accesses. Page 356.
  14. Tune L2 hit ratio when the window is larger than the set-aside cache. Page
     354.
  15. Query L2 access-policy limits before assuming a persisting window size. Page
     356.

  Shared Memory
  16. Pad shared-memory tiles to avoid bank conflicts, especially transposes.
  Pages 67-69.
  17. Keep shared-memory access stride-1 across warp lanes when possible. Page 69.
  18. Align dynamic shared-memory subarrays manually; misaligned typed slices are
  invalid/slow. Page 55.
  19. Remember shared memory and L1 share physical capacity; more smem can mean
  less L1. Pages 54, 133.
  20. Prefer cudaFuncSetAttribute carveout hints over hard cudaFuncSetCacheConfig
  when tuning L1/smem balance. Page 134.
  21. For SM80+, use LDGSTS / async global-to-shared copies for direct staging.
  Pages 133, 300-302.
  22. Use __pipeline_memcpy_async when you want direct control and guaranteed
  LDGSTS use. Page 306.
  23. Keep async copy sizes/alignment at 4, 8, or 16 bytes; 16-byte alignment is
  best. Pages 244, 608, 616.
  24. Do not read async-copy destination shared memory before wait / wait_prior.
  Pages 608, 616.
  25. Batch multiple async copies before one commit; cooperative-groups memcpy may
  auto-commit too aggressively. Page 306.
  26. Actually overlap copy and compute; immediate wait often limits cp.async
  benefit. Pages 130, 306.
  27. Use multi-stage prefetching/double buffering for iterative copy/compute
  loops. Pages 306-312.
  28. Consider producer-consumer warp specialization only when copy/computation
  chunks are large enough. Pages 300, 312.
  29. On Hopper-plus paths later, consider TMA for larger multidimensional tiles,
  not A6000 SM86. Pages 322, 329.
  30. For TMA-era kernels, shared-memory swizzling can reduce bank conflicts.
  Pages 335-336.

  Occupancy And Scheduling
  31. Tune occupancy as a latency-hiding tool, but not as a goal by itself. Pages
  70-71.
  32. Track registers, smem, max resident blocks, and resident warps together.
  Pages 28, 71, 119.
  33. Use __launch_bounds__ to tell the compiler the real launch shape and desired
  residency. Page 526.
  34. Use --maxrregcount only experimentally; spilling to local memory can erase
  occupancy wins. Page 72.
  35. Inspect local-memory spills because “local” memory is off-chip-backed and
  changes performance. Pages 55, 72.
  36. Keep block sizes warp-multiple unless there is a measured reason otherwise.
  Pages 25, 445.
  37. Avoid warp divergence in hot loops; SIMT masks inactive lanes. Page 25.
  38. Be careful with warp-synchronous assumptions on Volta+ independent thread
  scheduling. Page 117.
  39. Use warp shuffles/cooperative-group reductions instead of shared memory when
  scope is warp-only. Pages 241-242.
  40. Prefer narrower synchronization scopes: warp/group sync where legal, block
  sync only when needed. Page 241.

  Launch And Orchestration
  41. Use CUDA graphs for repeated decode sequences to reduce CPU launch overhead.
  Page 179.
  42. Upload/instantiate graphs outside timing; first graph launch/setup is not
  steady state. Page 212.
  43. Use stream capture for existing launch sequences instead of hand-building
  graphs when practical. Page 188.
  44. Avoid legacy default-stream synchronization when overlapping work; use
  explicit non-blocking streams. Pages 82, 86.
  45. Use CUDA events in streams for dependency edges and timing instead of
  device-wide sync. Pages 76, 79.
  46. Batch small memory transfers or small operations to reduce CPU/driver
  overhead. Page 112.
  47. Use stream-ordered allocation/pools for repeated temp buffers instead of
  repeated global alloc/free. Pages 218-238.
  48. Overlap host/device copies with kernels only when using async copies and
  appropriate streams. Pages 72-86.

  Math / Compiler
  49. Use FMA deliberately where algebra permits; validate numerics because
  floating-point reassociation changes results. Pages 585-586.
  50. Treat --use_fast_math, -fmad, -prec-div, and -prec-sqrt as measured options,
  not defaults. Page 586.
  51. For RMSNorm specifically, rsqrtf is the natural operation; compare exactness
  against reference after compiler-flag changes. Pages 586, 593.
  52. Use BF16/half vector types where they map cleanly to packed arithmetic and
  reduce memory traffic. Page 593.
  53. Keep CPU/GPU reference comparisons tolerant of non-associativity and
  implementation-defined math differences. Page 585.
  54. Avoid blind __forceinline__; the guide has __noinline__, __forceinline__,
  and __inline_hint__, so inlining is a tuning lever. Page 518.
  55. Use __grid_constant__ for large read-only kernel parameter objects to avoid
  per-thread local copies. Pages 519-520.