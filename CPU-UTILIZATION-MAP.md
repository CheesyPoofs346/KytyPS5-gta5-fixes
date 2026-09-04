# Why the CPU sits at 20–30%

Every prior measurement in this project was taken *inside a draw*. None of it could explain a
16-thread machine at 20–30%, because none of it looked at the threads. This is that map.

**Bottom line: at most two threads do anything, and one of them is asleep for most of the frame.**
That is ~1.3 busy cores out of 16 ≈ 8%, plus intermittent worker activity — which is exactly the
20–33% observed.

---

## The thread inventory

| thread | what it does | state |
|---|---|---|
| **guest thread(s)** | game logic, builds PM4 command buffers | **blocks once per frame** (§1) |
| **GPU thread** (one) | PM4 parse → draw translate → Vulkan record → submit | the whole pipeline, serial (§2) |
| draw worker pool (7) | *only* phase-2 resolve, and only when `--parallel-resolution` | idle in the default config |
| `m_priority_thread` | deferred priority ops (readback writebacks) | occasional |
| present thread | flip queue | occasional |
| `ResolveWorker` | VS resolve — **spin-waits**, see §6 | never created by default |

`GuestGpu` owns exactly one `std::jthread`, and its loop takes **one submission at a time**
(`graphicsRun.cpp` `ThreadRun`). PM4 parsing, draw translation and Vulkan recording all happen on
that one thread, in order.

---

## §1 The per-frame barrier — the big one

`AgcSuspendPoint()` → `GuestGpu::Done()` → `WaitForIdle()`, which blocks until
`m_processing == false && m_commands.empty() && m_submission_count == 0` — a **full drain of the
GPU thread**.

It runs **once per frame**. Proof from the code, not assumption: `Done()` is the only place
`m_done_num` is incremented, `GetFrameNum()` returns it, and that is the frame counter the entire
profiler uses (`Frame:` lines, `ShaderCensus`, `DrawCensus`).

So each frame: the guest builds commands → hits the suspend point → **sleeps through the GPU
thread's entire workload** (~40 ms of `DrawIndex`) → resumes. The two threads never overlap.

**Why the barrier exists** — and why it is removable:

```cpp
struct Submission {
    std::span<const uint32_t> commands;   // a VIEW into guest memory, not a copy
```

The submission does not own its PM4 data. Without the drain, the guest could overwrite its command
buffer while the GPU thread is still parsing it. Copy the stream at `Enqueue` and the barrier is no
longer needed.

Cost of copying: ~4150 draws × ~5 packets × ~8 dwords ≈ **640 KB/frame, ~50 µs** at typical memcpy
bandwidth — 0.09% of a 57 ms frame, to buy cross-frame pipelining.

**Ceiling is unknown until measured.** `FrameThreads` (committed, `f0220d4`) now prints:

```
FrameThreads: N ms/frame | guest blocked N ms (N%) | gpu busy N ms (N%) | overlap headroom N ms
```

`overlap headroom` is the ceiling on what pipelining can recover. Large ⇒ this is the 30 fps path.
Small ⇒ it is not, and the copy is not worth its risk.

---

## §2 The GPU thread is the whole pipeline, serially

For one submission, on one thread, in order:

1. parse PM4 packets (~20k/frame)
2. for each draw: derive state, resolve bindings, look up pipeline, record Vulkan
3. submit

Only step 2's *resolve* slice can currently go to workers, and in the default config
(`--test-parallel-bindings false`) `phase2_walk` is **10.4%** of drain. The other ~90% is one
thread.

Three attempts to widen this measured **zero fps** (descriptor caching, texture-cache shared lock,
bindings-in-phase-2 at both 2 and 8 workers — 17.10 fps identically). The reason found:
work handed to workers re-serialises on shared locks, and CPU stayed at 21–33% with seven threads
*parked*, not busy. See `gta5-bindings-blocked-by-range-lock`.

---

## §3 Measurement that was lying: PM4 non-draw

`PM4 non-draw packets` read 10.7–15.7 µs/draw — **larger than all of `DrawIndex`**:

```
DrawIndex : 39.9 ms/frame
PM4nonDraw: 44.6 ms/frame
sum       : 84.5 ms  vs an actual 57.2 ms frame     ← impossible
```

`IT_INDIRECT_BUFFER` recurses into a nested command buffer containing draws, so the timer wrapped
them. Fixed in `6517df5` by subtracting the `DrawIndex` cycles accumulated inside the handler.

**Consequence: the true cost of PM4 packet processing has never been measured**, on a path handling
~20k packets/frame. It could be a few ms/frame of pure serial GPU-thread time. The next drive
prints the real number for the first time.

---

## §4 Blocking points inventory

Every wait reachable from the graphics path, and whether it matters:

| site | trigger | verdict |
|---|---|---|
| `WaitForIdle` (§1) | every frame | **the main serializer** |
| `CommandScheduler::Finish` | texture/buffer CPU readback | full GPU stall; frequency unmeasured |
| `CommandScheduler::FlushAndWait` | `SetPredication` w/ wait, HDR probe | rare |
| `StreamBuffer` wrap → `Scheduler().Wait` | ring exhausted | ~4 MB/frame vs 64 MiB ring ⇒ ~1 frame in 16 |
| `FlipQueue::WaitForSubmitSlot` | queue full | capacity **16**; cannot bind at 17 fps — ruled out |
| `MasterSemaphore::Wait` | `waitSemaphores(UINT64_MAX)` | the mechanism behind Finish/Wait |
| `queue.submit` | per submit | **async, does not block** — ruled out |

---

## §5 Shader compilation is synchronous on the draw path

`pipelineCache.cpp:220` calls `CompileProgram(...)` **inline** during shader lookup, on the GPU
thread, inside the program-cache lock. Every new permutation stalls the draw *and* any worker
waiting on that lock.

693 shaders compiled in one session. Mostly front-loaded, but the logs show it continuing during
driving (560→562 spread out), so it is a live stutter source rather than a throughput one.

---

## §6 Landmine: `ResolveWorker` busy-spins

`renderDraw.cpp` `ResolveWorker` spins 2000 iterations before yielding — in its worker loop *and*
in `Wait()`. It burns a core and adds context switches for at best 2-way parallelism.

**Not active by default** (`parallel_resolve = false`, and the `&&` short-circuits before
`Instance()` is ever constructed, so the thread is never created). Do not enable
`--parallel-resolve` expecting a win.

---

## What to do, in order

1. **Drive once.** `FrameThreads` (§1) and the corrected PM4 number (§3) both print. Those two
   numbers decide everything below, and neither has ever been seen.
2. **If `overlap headroom` is large** → copy the PM4 stream at `Enqueue`, drop the per-frame drain,
   behind a default-off flag. This is the only identified change with 30 fps-scale headroom.
3. **If PM4 non-draw is large** → ~20k packets/frame on one thread is its own target, independent
   of draws.
4. **Only then revisit worker parallelism.** It has measured zero three times; it needs the lock
   contention fixed first, and §1 fixed to have threads free to use.

## Honest ceiling

17.5 fps = 57 ms; 30 fps = 33 ms. That needs 24 ms/frame removed. `DrawIndex` is ~40 ms of the 57.
Nothing measured so far is that size — the largest single identified win (tracker query fusion) is
~3%. §1 is the only item whose ceiling is plausibly in the right range, and its ceiling is exactly
what has not yet been measured.
