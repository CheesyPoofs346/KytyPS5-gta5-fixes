# Lock-free draw pipeline — design

Target: run draw translation across 8 cores so 4,200 draws/frame at ~14 µs each stop costing 59 ms
of one core. No visual sacrifice, no culling.

Read §0 first. Two previous attempts at this failed for reasons that are not obvious.

---

## §0 What already failed, and why the obvious fix is wrong

| attempt | result |
|---|---|
| Parallel SRT walk, 8 workers | even work split, **0 fps** |
| `PrepareGraphicsBindings` into phase 2 | **17.10 fps at 2 workers, 17.10 at 8.** Identical. |
| `VirtualRanges` → `std::shared_mutex` | **froze the game at boot** |

**The freeze is the important one.** Windows `std::shared_mutex` is SRWLock, which is *not
starvation-free*. Eight workers continuously holding read access starved the guest allocator's
`unique_lock` in `Add`/`Remove`/`Protect`; the game never left the loading screen — 44 draws/frame
against a normal 4,850.

So the design rule that follows is not stylistic:

> **No reader on the draw path may take a lock of any kind.** Not exclusive, not shared. A reader
> lock is a writer starvation risk, and the writer here is the guest's memory allocator.

RCU and atomic loads satisfy this. `shared_mutex` does not. That is why this design uses the former
everywhere on the hot path.

---

## §1 Complete inventory of shared state a worker touches per draw

Anything missing from this table is a crash or a corruption when workers go wide.

| # | object | current lock | worker needs | plan |
|---|---|---|---|---|
| 1 | `VirtualRanges::m_ranges` | `Common::Mutex`, **exclusive** | read only | **RCU snapshot** (§2) |
| 2 | `RegionManager::lock` (per region) | `TrackingSpinLock`, **exclusive** | read only | **atomic word loads** (§3) |
| 3 | `BufferCache::m_page_table_lock` | `shared_mutex`, engaged when concurrent | shared | keep — but see §6 risk |
| 4 | `TextureCache::m_lock` | `TrackingSharedLock` | shared | keep (already shared for workers) |
| 5 | `PipelineCache` mutex | mutex | — | phase 1 is serial by design, fine |
| 6 | `m_scheduler.Current()` | none | **must never touch** | **§4 — this is the blocker** |
| 7 | `m_renderer.GetMutex()` | mutex | — | only `EmitGlobalBarrier`, main thread |
| 8 | `graphics.queue_mutex` | mutex | — | submit only, main thread |

Measured cost of #1 and #2 on the hot path: `ClampRangeSize` **0.72 µs/draw** and tracker queries
**0.67 µs/draw** — together **~1.4 µs/draw, 10% of a draw**, and they are the two exclusive ones.

---

## §2 VirtualRanges — RCU by immutable snapshot

`m_ranges` is a sorted `std::vector<Range>`. Six readers (`ClampRangeSize`, `Query`, `QuerySpan`,
`QueryOverlap`, `HasOverlap`, `CountPageTableEntries`) only read it and write through a
caller-supplied `out` pointer — verified.

```cpp
struct RangeTable { std::vector<Range> ranges; };
std::atomic<std::shared_ptr<const RangeTable>> m_table;   // published snapshot
std::mutex                                    m_write_mutex;  // writers vs writers ONLY
```

**Reader** — no lock, cannot block, cannot starve anyone:
```cpp
const auto table = m_table.load(std::memory_order_acquire);
// identical binary search over table->ranges; the shared_ptr keeps it alive mid-search
```

**Writer** — copy, mutate the copy with the existing `*Unlocked` helpers, publish:
```cpp
std::lock_guard w(m_write_mutex);
auto next = std::make_shared<RangeTable>(*m_table.load(std::memory_order_relaxed));
EditUnlocked(next->ranges, ...);
m_generation.fetch_add(1, std::memory_order_release);
m_table.store(std::move(next), std::memory_order_release);
```

A writer never waits for a reader. That is the property `shared_mutex` lacked.

**Reclamation must be `shared_ptr`, not an epoch scheme tied to the batch flush.** `Query` and
`QuerySpan` are called from **guest threads** on the `mmap`/`munmap` paths (`memory.cpp:2488, 2685`),
so there is no point in the frame where every reader is known idle.

**Known cost:** `std::atomic<std::shared_ptr>` is not lock-free on MSVC — a short internal spinlock
plus a refcount bump, tens of cycles. Acceptable only because the thread_local fast path already
absorbs 85–93% of `ClampRangeSize` calls, so this is paid on roughly one call in seven. If that
refcount line turns out hot, the fallback is a raw `atomic<const RangeTable*>` with epoch
reclamation — better performance, much more machinery. **Do not start there.**

---

## §3 RegionManager — atomic dirty-bit reads, no lock

`IsRegionCpuModified` / `IsRegionGpuModified` currently do:

```cpp
Iterate<true>(vaddr, size, [](RegionManager* m, uint64_t off, uint64_t bytes) {
    std::scoped_lock lock(m->lock);              // EXCLUSIVE, for a pure read
    return m->IsModified<DirtySource::Cpu>(off, bytes);
});
```

`RegionBits` is `Common::BitArray<N>` — a plain `std::array<uint64_t, N/64>`. The read is a scan
for any set bit in a page range.

**The lock is not buying correctness for the reader.** Guest writes arrive asynchronously through
the page-fault handler, so a write can land the instant after the check returns. The lock cannot
prevent that and never did — it only protects the bitmap's internal consistency and its pairing with
the `mprotect` calls in `UpdateCpuProtection`.

So: make the words `std::atomic<uint64_t>` with relaxed loads for readers, keep the spinlock for
`ChangeState` / `ForEachModifiedRange` (which flip bits **and** call `UpdatePageWatchersForRegion`,
and must stay atomic with respect to each other).

A reader then sees any bit set before its load. A bit set during the load is the same race that
already exists. **No false negative is introduced that was not already possible.**

Direction of error matters and is safe: seeing a stale *set* bit means we re-upload unnecessarily
(slow, correct). The dangerous direction — missing a newly set bit — is unchanged from today.

---

## §4 The actual blocker: `m_scheduler.Current()`

This, not the locks, is why parallel bindings crashed. A worker has its own secondary command
buffer; it must **never** reach for the scheduler's current primary.

Known sites reachable from a worker:

| site | state |
|---|---|
| `bufferCache.cpp:617` `ObtainBuffer` | fixed — gated on `!MustStageForWorker()` |
| `textureCache.cpp:1347` `FindImage` | **aborts a run today** |
| `textureCache.cpp:1691` image clear | unguarded |
| `textureCache.cpp:1784, 2105, 2367` | **no guard at all** — records into whatever buffer is current, i.e. silent corruption |

**Rule:** every function reachable from the draw path gets classified as one of

- **pure** — reads only, safe on a worker;
- **records** — must take an explicit `vk::CommandBuffer` parameter, never `Current()`;
- **mutates shared state** — either deferred to a per-worker list flushed on the main thread, or
  bails the draw out for serial retry (`RequestWorkerBailout`, already built and measured at
  0 bailouts in steady state).

The audit must be exhaustive — `grep m_scheduler.Current()` across the whole draw path and classify
every hit. Three of the five sites above are the silent kind, and those do not announce themselves.

---

## §5 Worker pool — what exists and what changes

Already built and working: per-worker `VkCommandPool` + secondary buffers, `SecondaryRenderingFormats`,
`ParallelFor`, per-worker stream rings (`GetUtilityBuffer(MemoryUsage::Stream)`), per-worker deferred
LRU touches and page-watcher re-arms, per-worker recording generation, thread_local register view,
thread_local push constants and descriptor writes.

What changes:

1. **Partition at the state-run level, not the queue level.** A drained queue can span several
   render-state changes; a secondary's inheritance fixes its attachments. `RecordCensus` measured
   **exactly 1.00 retargets per secondary**, which says secondary boundary and state-run boundary
   already coincide — so runs are the natural unit.
2. **Each worker records its whole chunk into its own secondary.** No shared command buffer, ever.
3. **The primary replays them in guest order** with a single `vkCmdExecuteCommands` per run.
   Ordering is preserved because chunks are contiguous and replayed in index order.
4. **Barriers and layout transitions stay on the primary**, emitted before the pass opens — already
   how `FlushPendingTransitions` works.

Requires `inheritedQueries` for `vkCmdExecuteCommands` while an occlusion query is active.

---

## §6 Risks

1. **`BufferCache::m_page_table_lock` is a `std::shared_mutex`** engaged exactly when workers are
   active. That is the same class of object that froze the game in `VirtualRanges`. It has not yet
   been run at 8 workers with the rest of this in place. **Check it before trusting a wide run**, and
   convert it to the §2 pattern if the guest allocator is on the writer side of it.
2. **Silent corruption beats crashes here.** The `m_scheduler.Current()` sites that have no guard do
   not abort — they record into another thread's buffer. Any wide test must run with validation on
   at least once.
3. **The tests prove nothing about this.** `shader_recompiler_compute_tests` does not exercise these
   paths concurrently, and `GpuCommandLane` in it is flaky ~1 run in 5, pre-existing.
4. **This is guest memory infrastructure.** A fault in §2 or §3 is a guest hang or save corruption,
   not a rendering glitch.

---

## §7 Order of work, each independently testable

1. **§3 atomic dirty bits** — smallest, self-contained, no reclamation problem. Expect ~0.67 µs/draw
   back and, more importantly, removes an exclusive lock from every buffer resolution.
2. **§2 RCU** — larger. Verify at `--draw-workers 2` first: **boot must complete.** 44 draws/frame is
   the starvation signature.
3. **§4 audit** — exhaustive `Current()` classification. Nothing wide is safe until this is done.
4. **§5 chunked recording** — only after 1–3.

After each: one drive, compare **µs/draw**, not fps. Frame-rate comparisons across runs are
confounded by scene draw count — that mistake was made twice.

**Expected ceiling, stated honestly:** phase 2 currently measures ~10% of drain. Steps 1–3 remove
the serialization that made 8 workers equal 2, but the prize is only realised once §5 moves
*recording* off the main thread. If §5 lands and fps still does not move, the conclusion is that
per-draw work does not parallelise on this architecture, and that result should be accepted rather
than attacked a fourth time.
