# VirtualRanges: lock-free reads

Measured problem, from the 8-worker run:

```
ClampCensus: hits=85.3%  hit=40 cyc  miss=6848 cyc (lock=6230 work=619)  resets=50254
RunnerCensus: 8 runners, items even to within 1%   (not starved)
Task Manager: 21% CPU on 16 logical cores          (7 threads parked, not stalled)
```

**91% of a clamp miss is waiting for `m_mutex`.** `work` has measured 550-620 cycles in every run
today, so the binary search was never the cost. Phase 2 costs what it would cost fully serial
(579 us observed per batch vs 634 us serial vs 79 us ideal), which is why 2 workers and 8 workers
both produce exactly 17.10 fps.

Two lock changes are already ruled out **by measurement, not argument**:

| approach | result |
|---|---|
| exclusive `Common::Mutex` (today) | 6230 cycles of acquisition per miss |
| `std::shared_mutex` | **froze the game.** Windows SRWLock is not starvation-free: 8 readers starved the guest allocator's `unique_lock` in Add/Remove/Protect, and the game never left boot - 44 draws/frame vs 4850, black screen at a live 30 fps. Reverted in 599d115. |

The lock *type* is not the lever. The requirement is a structure where **readers never block and
writers never wait for readers.**

---

## Step 0 first: stop mutating when nothing changes

Cheapest possible change, and it may remove most of the problem before any architecture moves.

`generation` reached **159,178** over one drive, and every bump resets every thread's fast path
(`resets=50254` on the main thread alone). The dominant mutator is `Protect`, called constantly by
page-watcher re-arming. But `Protect` only actually changes the table when the requested protection
differs from what the range already has - `EditUnlocked` splits a range only to apply a *different*
attribute to part of it.

So: in `Protect`, `SetMemoryType` and `Rename`, scan the affected span first and return without
touching the table or bumping the generation when every covered range already holds the requested
value.

- Costs one extra read-only scan on the write path, which is rare.
- If most re-arms are no-ops, the miss rate collapses on its own (misses rose 7% -> 9% -> 14.7% as
  workers increased, and resets are what drives that).
- **Measurable on its own**: `generation` and `resets` in `ClampCensus` must fall. If they don't,
  the mutations are real and Step 1 is required.

Do this first and measure it. It is a dozen lines and cannot starve anything.

---

## Step 1: RCU by immutable snapshot

### The shape

```cpp
struct RangeTable {
    std::vector<Range> ranges;      // sorted by start, exactly as m_ranges is today
};

std::atomic<std::shared_ptr<const RangeTable>> m_table;   // published snapshot
std::mutex                                    m_write_mutex;  // serialises WRITERS ONLY
```

`std::atomic<std::shared_ptr<T>>` is C++20 and this project builds with
`CMAKE_CXX_STANDARD 20`, so it is available.

### Reader path - no lock, no blocking

```cpp
uint64_t ClampRangeSize(uint64_t vaddr, uint64_t size) {
    // ... thread_local fast path first, unchanged ...
    const auto table = m_table.load(std::memory_order_acquire);   // one atomic load
    const auto& ranges = table->ranges;
    auto vma = std::upper_bound(ranges.begin(), ranges.end(), vaddr, ...);
    // ... identical search, against a snapshot that cannot change under us ...
}
```

The `shared_ptr` copy keeps that table alive for as long as this reader holds it, so a writer
publishing a new table mid-search cannot pull the vector out from under the binary search. The
reader never waits for anything.

All six readers convert the same way: `ClampRangeSize`, `Query`, `QuerySpan`, `QueryOverlap`,
`HasOverlap`, `CountPageTableEntries`. Each already only reads `m_ranges` and writes through a
caller-supplied `out` pointer - verified when they were converted to `shared_lock` in 75ac708.

### Writer path - copy, mutate, publish

```cpp
void Protect(uint64_t start, uint64_t size, int protection) {
    std::lock_guard writers(m_write_mutex);            // writers vs writers only
    auto next = std::make_shared<RangeTable>(*m_table.load(std::memory_order_relaxed));
    // mutate next->ranges with the existing *Unlocked helpers, unchanged
    EditUnlocked(next->ranges, start, size, ...);
    m_generation.fetch_add(1, std::memory_order_release);
    m_table.store(std::move(next), std::memory_order_release);   // publish
}
```

Writers serialise against each other, which is correct and cheap - they are rare relative to the
6 reads per draw per thread. **A writer never waits for a reader**, which is the entire point:
that is the property `std::shared_mutex` failed to provide and that froze the game.

The existing `*Unlocked` helpers (`EditUnlocked`, `MergeUnlocked`, `MergeAroundUnlocked`,
`RemoveUnlocked`, `LowerBound`, `FindOverlap`) keep their logic; they take the vector to mutate as
a parameter instead of touching `m_ranges` directly.

### Reclamation - handled, and why it has to be

`shared_ptr` frees the old table when the last reader releases it. This is not optional
sophistication: **the readers are called from guest threads**, not only from graphics workers -
`Query` and `QuerySpan` are reached from the guest `mmap`/`munmap` paths (memory.cpp:2488, 2685).
So there is no point in the frame where every possible reader is known idle, which rules out the
cheaper scheme of retiring old tables at the batch flush the way deferred touches are drained.
Epoch-based reclamation would need every guest thread to publish an epoch, which is far more
invasive than accepting the refcount.

### Why the reader cost is acceptable

`std::atomic<std::shared_ptr<>>` is not lock-free on MSVC - it uses a short internal spinlock plus
a refcount bump, so a read is on the order of tens of cycles under contention rather than a single
load. That is the honest downside, and it is fine here for one reason: **the thread_local fast path
absorbs 85-93% of calls before reaching it.** Only misses touch the snapshot, so this replaces
~6230 cycles of parked-thread lock wait with tens of cycles on roughly one call in seven.

If that refcount cache line turns out to be hot, the fallback is a raw
`std::atomic<const RangeTable*>` with epoch-based reclamation - strictly better performance,
strictly more machinery. Do not start there.

---

## Step 2: stop nuking the fast path on every mutation

Independent of the lock, and possibly worth as much.

`ClampFastPath::Reset` clears all 8 entries whenever the generation moves, so one unrelated
`Protect` anywhere in the address space discards a thread's whole cache. The census shows the cost:
`cached=2` of 8 slots in steady state, and a miss rate that climbs with worker count.

With a lock-free snapshot, re-validation becomes cheap: on a generation change, instead of clearing,
re-check each cached entry against the current snapshot and drop only the entries that actually
moved. A cached `[start, end)` that still exists unchanged stays valid.

This is only worth doing **after** Step 1, because today re-validation would mean taking the
contended lock - which is the thing being fixed.

---

## Safety properties

| property | how it is guaranteed |
|---|---|
| readers never block | one atomic load, then a search over an immutable snapshot |
| writers never wait for readers | writers only take `m_write_mutex`, held against other writers |
| no writer starvation | follows from the above - this is the exact failure of `std::shared_mutex` |
| no torn reads | the snapshot is immutable once published; mutation happens on a private copy |
| no use-after-free | `shared_ptr` keeps a table alive while any reader holds it |
| no deadlock | one lock, never held across a call into other subsystems |

---

## Risks

1. **Copy cost per mutation.** Every write clones the whole range vector. At ~159k mutations per
   drive this is the main new cost, and it is why Step 0 comes first - eliding no-op writes removes
   most of them. If mutations stay high after Step 0, measure the clone before proceeding.
2. **`m_ranges` is touched in more places than the 14 locking methods.** Every `*Unlocked` helper
   and anything else naming `m_ranges` has to be routed through the table parameter. Grep for
   `m_ranges` and convert exhaustively - a missed one reads a stale or freed vector.
3. **This is guest memory infrastructure, not graphics code.** Allocation, protection and the page
   tracker all run through it. A fault here is a hang or corruption in the guest, not a rendering
   glitch, and it will not be obvious which change caused it.
4. **`Common::Mutex` may be used for more than mutual exclusion** - check whether anything relies on
   its recursion or ownership behaviour before swapping it out.

---

## Verification plan

Nothing here is proven by the tests: `shader_recompiler_compute_tests` does not exercise
`VirtualRanges` concurrently, and `GpuCommandLane` in it is already flaky (~1 run in 5, pre-existing
on 435cba4). So run the suite several times, but do not read it as evidence about this change.

1. **Step 0 alone.** Drive. `generation` and `resets` in `ClampCensus` must fall. If they do not,
   the no-op elision hypothesis is wrong - say so and go to Step 1 without it.
2. **Step 1 at `--draw-workers 2`.** Boot must complete (44 draws/frame is the starvation
   signature). `lock` in `ClampCensus` must collapse toward zero, since there is no reader lock left
   to wait on.
3. **Step 1 at `--draw-workers 8`.** The real test. If `phase2_walk` finally scales, fps moves. If
   fps is again identical to the 2-worker run, then something *else* serialises phase 2 and the
   bindings move should come out rather than be defended.
4. Watch for guest-side symptoms specifically - hangs during load, allocation failures, save
   corruption - not just rendering.

## Rollback

Step 0 and Step 1 land as separate commits so either can be reverted alone. Neither changes the
`--test-parallel-bindings` default, so a revert restores today's behaviour exactly.
