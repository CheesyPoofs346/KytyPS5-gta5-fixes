# Section A baseline — two matched routes

Revision `584be50` (exe built 17:11:05; the only later commit is tool-only Python).
Captured 2026-09-05. Raw data preserved: `route1-capture.csv` (md5 `af5282099c5a…`),
`route2-capture.csv` (md5 `16bbc06eb095…`), logs `route1-preserved.log`,
`route2-preserved.log`.

## Effective settings (identical in both runs, diff-verified)

```
draw_profile            false      draw_workers             8
secondary_record        true       pipeline_depth           4
draw_queue              true       eop_flush_interval       1
parallel_resolution     true       stream_repeat_threshold  0
test_parallel_bindings  true       warmup_frames            600
frame_pipelining        false      coalesce_eop_flush       false
light_partial_flush     false      buffer_dedup             true
pipeline_memo           true       dyn_state_cache          true
defer_uploads           false      defer_transitions        false
cache_descriptors       false      hw_check                 true
```

Present mode Mailbox. `frame_pipelining false`, so `Done()` calls `WaitForIdle()` each frame.

## Results

| | run 1 | run 2 |
|---|---|---|
| route duration | 93.16 s | 66.34 s |
| intervals (n) | 1445 | 1024 |
| ROUTE_START consumed at sample | 4194 | 3785 |
| median ms | 61.35 | 62.95 |
| p95 ms | 84.57 | 81.12 |
| p99 ms | 113.91 | 107.95 |
| min / max ms | 5.57 / 1655.70 | 44.80 / 213.04 |
| mean ms | 64.47 | 64.78 |
| draws median / max | 4261 / 6528 | 4339 / 6379 |
| over 16.667 ms (PRIMARY) | 97.9% | 100.0% |
| over 33.333 ms | 97.3% | 100.0% |

Phase accounting per run: run 1 `4194 warmup / 1 straddle / 1445 route / 170 after`;
run 2 `3785 warmup / 1 straddle / 1024 route / …`. The interval straddling ROUTE_START
was excluded in both.

## Observed difference between the two runs

```
median ms        61.35 – 62.95   difference 1.60  (2.6%)
p95 ms           81.12 – 84.57   difference 3.46  (4.3%)
p99 ms          107.95 – 113.91  difference 5.96  (5.5%)
% over 16.667ms  97.85 – 100.00  difference 2.15  (2.2%)
```

This is **the observed difference between two runs, not a statistical variance floor**. Two
samples cannot establish a distribution: they give no confidence interval and no estimate of
how much larger the spread could be across more runs. Treat it as a first indication of
magnitude, and re-establish it with repeated captures before leaning on it to judge an A/B.

Median draw counts agree to 1.8% (4261 vs 4339), which **suggests** similar scene density.
It does not prove equivalent workloads: the routes differed in duration (93.16 s vs 66.34 s),
run 1 contains a segment run 2 does not (below), and equal medians are compatible with
different distributions of work.

## Run 1 contains low-draw intervals that run 2 does not

Run 1 has **43 route intervals with fewer than 1000 draws** (all 31 of its sub-16.667 ms
intervals are among them, at 60–312 draws; its 1655.70 ms outlier is at 369 draws).
Run 2 has **zero**.

Low draw counts are consistent with menu, pause or transition frames, but that is an
**inference from the draw counter, not a confirmed observation** — no route annotation or
state marker was recorded to establish what was on screen. It is unconfirmed.

**The unfiltered result stands as the primary figure.** The filtered view below is reported
as an explicit, reversible exclusion (`draws >= 1000`, dropping 43 of 1445), not a correction:

| run 1 | n | med | p95 | p99 | min | max | over 16.667 | over 33.333 |
|---|---|---|---|---|---|---|---|---|
| unfiltered (primary) | 1445 | 61.35 | 84.57 | 113.91 | 5.57 | 1655.70 | 97.9% | 97.3% |
| draws >= 1000 | 1402 | 61.64 | 84.67 | 112.49 | 38.13 | 336.87 | 100.0% | 100.0% |

Recalculated, not assumed: excluding those intervals moves the median +0.47%, p95 +0.12%
and p99 -1.25%. Those shifts are smaller than the 2.6-5.5% difference observed between the
two runs, so the central and tail statistics are **not very sensitive** to this exclusion --
but they are not literally unaffected, and the max and min move a great deal.

The over-threshold statistic is what the exclusion actually changes: 97.9% to 100.0%,
matching run 2 exactly.

## The gap to target

Median interval ~62 ms against a 16.667 ms budget:

- Bringing the **median** interval to 33.333 ms needs a **1.9x reduction**.
- Bringing the **median** interval to 16.667 ms needs a **3.8x reduction**.

These ratios describe the median only. A median at 33.333 ms is **not** a 30 fps floor: by
definition half the intervals would still be slower.

Looking at the tail instead, the measured p99 (107.95-113.91 ms) would need roughly a
**3.2x-3.4x** reduction to fall under 33.333 ms and **6.5x-6.8x** to fall under 16.667 ms.

These are **the improvement needed to bring the measured p99 of these two captures under
budget**. They are not a guaranteed uniform speedup requirement, and meeting them would not
establish a steady 60 fps: a p99 under budget still leaves 1% of intervals above it, the
remaining tail beyond p99 is unbounded here, and these are guest submission intervals rather
than displayed frames.

Not a matter of trimming a few percent, on any of these measures.

## What these numbers are not

- **Guest submission intervals**, one sample per `GuestGpu::Done()` after
  `WaitForIdle`/`WaitForPipelineDepth`. Not presentation intervals, not displayed frames,
  not host submissions.
- Flip ratio 1.000 in both runs is a **count consistency check only**. It does not establish
  one-to-one correspondence between a flip and a submission, nor matching timing. Displayed
  FPS is not derivable from these intervals.

## Capture buckets — what is usable

```
                    run 1                        run 2
pm4-exec     10541532.4 ms / 745977 calls   8279863.7 ms / 571691 calls   INVALID
worker-exec    273513.9 ms / 1789606        215507.2 ms / 1380176         overlapping
wait-idle      251727.6 ms / 5642           211236.4 ms / 4812
wait-workers     5378.3 ms / 227752           4188.8 ms / 175751          overlapping
wait-pipeline  0 calls                      0 calls
wait-stream    N/A (no site wired)          N/A (no site wired)
```

**`pm4-exec` is discarded from both runs.** `ProcessPm4` is recursive
(`graphicsRun.cpp:961`, via `ProcessIndirectBuffer`), so a non-reentrant timer re-counts
enclosing time on every nested call. Against the matching process window — 318.0 s of
submission intervals in run 1 — its 10541.5 s is ~33× inflated.

These are **process-wide accumulated elapsed times and they overlap**: eight workers busy
for 10 ms contribute 80 ms of `worker-exec`. Do not sum them, and do not subtract `wait-*`
from `pm4-exec`; that subtraction needs thread identity, nesting and capture windows
established, and they are not.

`wait-idle` averages 44.62 ms (run 1) and 43.90 ms (run 2) per call, consistent across runs.
This is a **process-cumulative mean** and must not be compared against the route median,
which is a different window and a different statistic. `WaitForIdle` has exactly one call
site (`graphicsRun.cpp:400`, inside `Done()`); the small call/boundary gap (5642 vs 5640) is
a sampling artifact — the first interval has no predecessor to subtract, and the FINAL report
snapshots the sample count while counters keep advancing.

`wait-pipeline` reading 0 calls is consistent with `frame_pipelining false`. It does not
independently validate the instrumentation.

## Next: instrumentation patch (not an optimization)

1. Outermost-only PM4 timing via a thread-local RAII depth guard. Outermost elapsed time
   still **includes blocking** — it is not exclusive CPU execution and must not be labelled
   as such.
2. Route-scoped counter deltas, so buckets are snapshotted at ROUTE_START/ROUTE_END rather
   than reported as process-cumulative totals.
3. A breakdown of what `WaitForIdle` actually waits on.
4. Wire `wait-stream`, or keep reporting it N/A.
