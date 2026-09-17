# Safe worker-image slice: measurement gate

Verified 2026-09-11: branch `codex/parallel-bindings-prep` is at `4d58b04`.
The focused harness and adapter changes are an uncommitted diff in
`tests/ShaderRecompilerComputeTests.cpp`; they are not part of that commit.
No gameplay build or launch was performed for this measurement review.

## Counters required before a gameplay comparison

`DrainMs` currently accumulates from its first call. Its "window" label does
not indicate a reset interval. Difference consecutive snapshots, including
draw counts and elapsed time, or implement real interval resets.

`phase2b_ACQUIRE` and `phase2c` run only under `TestParallelBindingsEnabled()`.
Keep that unsafe experiment off. Those fields do not isolate caller work for
the safe `WorkerResolveImages()` path.

For the safe path, collect these quantities over the same intervals:

- Total phase-2 wall time, including worker dispatch, resource walking,
  image resolution, and joining. Compare off/on; worker CPU times cannot be
  added to obtain the elapsed cost.
- Attempted image slots, successful worker results, and per-slot tickets.
  Reduce the existing per-draw `vertex_images` / `pixel_images` results on the
  caller after workers finish to avoid adding a shared atomic per lookup.
- Results actually retained by caller `PrepareBindings`, generation-invalidated
  results, and caller fallback resolutions. A worker hit alone is not a saved lookup.
- Caller preparation, image rebinding, and descriptor publication costs on the
  real serial path. Account for nesting; do not sum inclusive parent and child timers.
- End-to-end frame durations, draw counts, and shader-compilation activity.

## Coordinated comparison

Use one verified executable with the focused correctness coverage passing.
Record its SHA-256, source revision plus dirty diff, effective settings, and
log paths. Preserve the original gameplay checkout and saves.

Hold all settings fixed except `worker_resolve_images`: off, on, off.
Keep full parallel binding off, and use the same worker count, draw queue,
resource-walking mode, resolution, save, and dense-city route. Confirm the
worker-image path actually executes with those settings.

Warm the same executable and scene before collecting each marked interval.
Exclude menus, loading, pauses, and compilation-heavy intervals explicitly.
Report frame-time median, p95, p99, samples, and the fraction above 33.33 ms,
alongside matched interval phase costs and retained-hit/ticket rates.

The earlier 0.664 us/draw ResolveTexture measurement is inclusive and from a
different compiled-SRT capture. It establishes neither eligibility nor an
expected speedup. Accept this slice as a performance change only if the paired
results show a repeatable end-to-end gain without rendering regressions.
